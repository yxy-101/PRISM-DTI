"""Dirichlet evidential classification losses.

The model heads output positive Dirichlet parameters (`alpha_*`). These helpers
train those parameters directly instead of treating the normalized probability
as an ordinary softmax output.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F


SUPPORTED_EVIDENTIAL_LOSSES = ("digamma", "mse")


def evidential_loss(
    alpha: torch.Tensor,
    targets: torch.Tensor,
    loss_type: str = "digamma",
    kl_weight: float = 0.0,
    kl_annealing_factor: float = 1.0,
    sample_weights: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
    return_components: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
    """Return a weighted Dirichlet evidential loss.

    Parameters
    ----------
    alpha:
        Dirichlet parameters shaped ``[batch, num_classes]``.
    targets:
        Integer class labels shaped ``[batch]``.
    loss_type:
        ``digamma`` for expected cross entropy under the Dirichlet posterior, or
        ``mse`` for the Sensoy-style evidential MSE risk.
    kl_weight:
        Weight for the KL penalty toward a uniform Dirichlet prior.
    kl_annealing_factor:
        Multiplicative annealing factor in ``[0, 1]``.
    sample_weights:
        Optional per-sample weights used by prior-balancing probes.
    """
    if alpha.dim() != 2:
        raise ValueError("Expected alpha with shape [batch, num_classes].")
    if alpha.size(0) != targets.reshape(-1).size(0):
        raise ValueError("alpha batch size and target count do not match.")

    alpha = alpha.clamp_min(float(eps))
    targets = targets.reshape(-1).long()
    num_classes = int(alpha.size(1))
    one_hot = F.one_hot(targets, num_classes=num_classes).to(dtype=alpha.dtype)

    normalized_type = str(loss_type).strip().lower()
    if normalized_type in {"edl", "evidential", "expected_nll", "expected_ce"}:
        normalized_type = "digamma"
    if normalized_type not in SUPPORTED_EVIDENTIAL_LOSSES:
        raise ValueError(
            "Unsupported evidential loss %r. Supported losses: %s"
            % (loss_type, ", ".join(SUPPORTED_EVIDENTIAL_LOSSES))
        )

    if normalized_type == "digamma":
        per_sample_data = _digamma_risk(alpha=alpha, one_hot=one_hot)
    else:
        per_sample_data = _mse_risk(alpha=alpha, one_hot=one_hot)

    kl = dirichlet_kl_to_uniform(remove_true_class_evidence(alpha, one_hot))
    annealed_kl_weight = float(kl_weight) * float(kl_annealing_factor)
    per_sample_total = per_sample_data + annealed_kl_weight * kl
    loss = _weighted_mean(per_sample_total, sample_weights)

    if not bool(torch.isfinite(loss).all().item()):
        raise ValueError("Evidential loss is non-finite.")

    if not return_components:
        return loss

    components = {
        "loss_type": normalized_type,
        "data_loss": float(_weighted_mean(per_sample_data, sample_weights).detach().cpu().item()),
        "kl_loss": float(_weighted_mean(kl, sample_weights).detach().cpu().item()),
        "kl_weight": float(kl_weight),
        "kl_annealing_factor": float(kl_annealing_factor),
        "annealed_kl_weight": annealed_kl_weight,
    }
    return loss, components


def _digamma_risk(alpha: torch.Tensor, one_hot: torch.Tensor) -> torch.Tensor:
    """Expected cross entropy under a Dirichlet posterior."""
    alpha_sum = alpha.sum(dim=-1, keepdim=True)
    return torch.sum(one_hot * (torch.digamma(alpha_sum) - torch.digamma(alpha)), dim=-1)


def _mse_risk(alpha: torch.Tensor, one_hot: torch.Tensor) -> torch.Tensor:
    """Evidential MSE risk with Dirichlet variance."""
    alpha_sum = alpha.sum(dim=-1, keepdim=True)
    probabilities = alpha / alpha_sum
    squared_error = torch.sum((one_hot - probabilities).pow(2), dim=-1)
    variance = torch.sum(
        probabilities * (1.0 - probabilities) / (alpha_sum + 1.0),
        dim=-1,
    )
    return squared_error + variance


def remove_true_class_evidence(alpha: torch.Tensor, one_hot: torch.Tensor) -> torch.Tensor:
    """Keep false-class evidence for KL regularization and neutralize the label."""
    return one_hot + (1.0 - one_hot) * alpha


def dirichlet_kl_to_uniform(alpha: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return KL(Dir(alpha) || Dir(1)) for each sample."""
    alpha = alpha.clamp_min(float(eps))
    num_classes = int(alpha.size(-1))
    alpha_sum = alpha.sum(dim=-1, keepdim=True)
    log_norm = (
        torch.lgamma(alpha_sum).squeeze(-1)
        - torch.lgamma(alpha).sum(dim=-1)
        - torch.lgamma(alpha.new_tensor(float(num_classes)))
    )
    digamma_gap = torch.digamma(alpha) - torch.digamma(alpha_sum)
    return log_norm + torch.sum((alpha - 1.0) * digamma_gap, dim=-1)


def _weighted_mean(values: torch.Tensor, sample_weights: Optional[torch.Tensor]) -> torch.Tensor:
    """Average per-sample losses with optional positive weights."""
    if sample_weights is None:
        return values.mean()
    weights = sample_weights.reshape(-1).to(device=values.device, dtype=values.dtype)
    if weights.size(0) != values.size(0):
        raise ValueError("sample weight count does not match loss count.")
    denominator = weights.sum().clamp_min(1e-8)
    return (values * weights).sum() / denominator

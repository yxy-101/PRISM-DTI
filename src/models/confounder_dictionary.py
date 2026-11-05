"""Target-prior confounder dictionary and evidence pooling helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import nn


class ConfounderDictionary(nn.Module):
    """Load target-prior clusters and pool evidential posteriors.

    TODO: Add versioned metadata once cluster dictionaries are produced from
    multiple datasets or ablations.
    """

    def __init__(
        self,
        cluster_centers_path: Optional[str] = None,
        p_ci_path: Optional[str] = None,
        num_fallback_confounders: int = 1,
    ) -> None:
        """Create an optional file-backed confounder dictionary."""
        super().__init__()
        self.num_fallback_confounders = max(1, int(num_fallback_confounders))
        centers = self._load_optional_tensor(cluster_centers_path)
        p_ci = self._load_optional_tensor(p_ci_path)
        if cluster_centers_path or p_ci_path:
            if centers.dim() != 2 or centers.size(0) == 0 or not torch.isfinite(centers).all():
                raise ValueError("Context centers must be a finite nonempty [M, D] tensor.")
            _normalize_prior(p_ci, centers.size(0), torch.device("cpu"))
        self.register_buffer("cluster_centers", centers)
        self.register_buffer("p_ci", p_ci)

    def has_loaded_dictionary(self) -> bool:
        """Return whether file-backed confounders are available."""
        return self.cluster_centers.numel() > 0

    def get_bank(self, reference_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return confounder centers and prior weights on the reference device."""
        if self.has_loaded_dictionary():
            centers = self.cluster_centers.to(reference_features.device).float()
            weights = self._normalized_prior(centers.size(0), reference_features.device)
            return centers, weights

        fallback = reference_features.detach()
        if fallback.size(0) > self.num_fallback_confounders:
            fallback = fallback[: self.num_fallback_confounders]
        weights = torch.ones(fallback.size(0), device=reference_features.device)
        weights = weights / weights.sum().clamp_min(1e-8)
        return fallback, weights

    def backdoor_evidence_pooling(
        self,
        alpha_by_confounder: torch.Tensor,
        p_ci: torch.Tensor,
    ) -> torch.Tensor:
        """Approximate the causal evidential posterior by pooling evidence.

        `alpha_by_confounder` is shaped `[batch, num_clusters, num_classes]`.
        The method pools evidence (`alpha - 1`) under `p_ci`, then reconstructs
        Dirichlet parameters as `1 + pooled_evidence`.
        """
        if alpha_by_confounder.dim() != 3:
            raise ValueError(
                "alpha_by_confounder must be [batch, num_clusters, num_classes]."
            )
        num_clusters = alpha_by_confounder.size(1)
        weights = _normalize_prior(p_ci, num_clusters, alpha_by_confounder.device)
        evidence = (alpha_by_confounder - 1.0).clamp_min(0.0)
        pooled = torch.sum(evidence * weights.view(1, num_clusters, 1), dim=1)
        return pooled + 1.0

    def backdoor_alpha_pooling(
        self,
        alpha_by_confounder: torch.Tensor,
        p_ci: torch.Tensor,
    ) -> torch.Tensor:
        """Pool Dirichlet parameters directly under the confounder prior.

        This is the direct ``sum_m pi_m alpha_m`` comparator. It is kept
        separate from :meth:`backdoor_evidence_pooling`, whose canonical
        intervention pools ``alpha - 1`` and reconstructs ``1 + evidence``.
        """
        self._validate_alpha_bank(alpha_by_confounder)
        num_clusters = alpha_by_confounder.size(1)
        weights = _normalize_prior(p_ci, num_clusters, alpha_by_confounder.device)
        return torch.sum(
            alpha_by_confounder * weights.view(1, num_clusters, 1),
            dim=1,
        )

    def backdoor_probability_pooling(
        self,
        alpha_by_confounder: torch.Tensor,
        p_ci: torch.Tensor,
    ) -> torch.Tensor:
        """Pool normalized class probabilities and preserve mean concentration.

        The pooled probability is ``sum_m pi_m p_m``. Because the model
        contract also exposes evidential uncertainty, the weighted mean
        Dirichlet concentration is retained and the pooled probability is
        reconstructed as ``p_pool * sum_pool``. This makes the probability
        comparator explicit while keeping ``p_do`` and ``u_do`` well-defined.
        """
        self._validate_alpha_bank(alpha_by_confounder)
        num_clusters = alpha_by_confounder.size(1)
        weights = _normalize_prior(p_ci, num_clusters, alpha_by_confounder.device)
        concentration = alpha_by_confounder.sum(dim=-1, keepdim=True)
        probabilities = alpha_by_confounder / concentration.clamp_min(1e-8)
        pooled_probability = torch.sum(
            probabilities * weights.view(1, num_clusters, 1),
            dim=1,
        )
        pooled_concentration = torch.sum(
            concentration * weights.view(1, num_clusters, 1),
            dim=1,
        )
        return pooled_probability * pooled_concentration

    @staticmethod
    def _validate_alpha_bank(alpha_by_confounder: torch.Tensor) -> None:
        """Validate the common per-confounder Dirichlet tensor shape."""
        if alpha_by_confounder.dim() != 3:
            raise ValueError(
                "alpha_by_confounder must be [batch, num_clusters, num_classes]."
            )

    def _normalized_prior(self, num_clusters: int, device: torch.device) -> torch.Tensor:
        """Return normalized file-backed or uniform prior weights."""
        if self.p_ci.numel() == num_clusters:
            return _normalize_prior(self.p_ci, num_clusters, device)
        weights = torch.ones(num_clusters, device=device)
        return weights / weights.sum().clamp_min(1e-8)

    @staticmethod
    def _load_optional_tensor(path: Optional[str]) -> torch.Tensor:
        """Load an optional tensor from disk, or return an empty tensor."""
        if not path:
            return torch.empty(0)
        tensor_path = Path(path).expanduser()
        if not tensor_path.exists():
            raise FileNotFoundError("Confounder tensor not found: %s" % tensor_path)
        loaded = torch.load(str(tensor_path), map_location="cpu")
        if isinstance(loaded, dict):
            for key in ("cluster_centers", "p_ci", "weights", "tensor"):
                if key in loaded:
                    loaded = loaded[key]
                    break
        if not torch.is_tensor(loaded):
            loaded = torch.as_tensor(loaded)
        return loaded.float()


def _normalize_prior(
    p_ci: torch.Tensor,
    num_clusters: int,
    device: torch.device,
) -> torch.Tensor:
    """Normalize valid cluster mass; reject incompatible or invalid weights."""
    prior = p_ci.reshape(-1).float().to(device)
    if prior.numel() != num_clusters:
        raise ValueError("Cluster mass length must match the number of centers.")
    if not torch.isfinite(prior).all() or bool((prior < 0).any()) or float(prior.sum()) <= 0:
        raise ValueError("Cluster masses must be finite, nonnegative and have positive total.")
    return prior / prior.sum().clamp_min(1e-8)

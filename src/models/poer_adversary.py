"""Prior adversary used by POER confidence regularization."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn


class _GradientReversal(torch.autograd.Function):
    """Identity in the forward pass and sign-reversed gradients in backward."""

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.scale * grad_output, None


def gradient_reverse(inputs: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal with a configurable backward scale."""
    return _GradientReversal.apply(inputs, float(scale))


class POERPriorAdversary(nn.Module):
    """Small MLP that predicts target-prior group from confidence features."""

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 32,
        num_groups: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("POERPriorAdversary input_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("POERPriorAdversary hidden_dim must be positive.")
        if num_groups <= 1:
            raise ValueError("POERPriorAdversary num_groups must be >= 2.")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_groups = int(num_groups)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.num_groups),
        )

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]]) -> "POERPriorAdversary":
        """Build the adversary from a project config dictionary."""
        config = config or {}
        poer_config = config.get("poer", {})
        confidence_features = poer_config.get("confidence_features") or [
            "logit_p_raw_positive",
            "u_raw",
            "pcs",
            "evidence_strength",
        ]
        return cls(
            input_dim=int(poer_config.get("adversary_input_dim", len(confidence_features))),
            hidden_dim=int(poer_config.get("adversary_hidden_dim", 32)),
            num_groups=int(poer_config.get("num_prior_groups", 3)),
            dropout=float(poer_config.get("adversary_dropout", 0.1)),
        )

    def forward(self, confidence_features: torch.Tensor, grl_scale: float = 1.0) -> torch.Tensor:
        """Return prior-group logits after gradient reversal."""
        if confidence_features.dim() != 2:
            raise ValueError(
                "POER confidence features must have shape [batch, feature_dim], "
                "got %s." % (tuple(confidence_features.shape),)
            )
        if confidence_features.size(1) != self.input_dim:
            raise ValueError(
                "POER adversary expected feature_dim=%d, got %d."
                % (self.input_dim, confidence_features.size(1))
            )
        reversed_features = gradient_reverse(confidence_features, scale=grl_scale)
        return self.network(reversed_features)

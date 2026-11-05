"""Evidential prediction head for the raw branch."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class EvidentialHead(nn.Module):
    """Convert fused features into Dirichlet evidence statistics.

    The raw branch returns:
    - `alpha_raw`: positive Dirichlet parameters
    - `p_raw`: normalized class probabilities
    - `u_raw`: evidential uncertainty scalar per sample

    TODO: Add calibration hooks for bias-focused uncertainty analysis.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_classes: int = 2,
        eps: float = 1e-8,
    ) -> None:
        """Create the evidential raw prediction head."""
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.softplus = nn.Softplus()

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return `alpha_raw`, `p_raw`, and `u_raw` for fused embeddings."""
        evidence = self.softplus(self.mlp(features))
        alpha_raw = evidence + 1.0
        alpha_sum = torch.sum(alpha_raw, dim=-1, keepdim=True).clamp_min(self.eps)
        p_raw = alpha_raw / alpha_sum
        u_raw = float(self.num_classes) / alpha_sum
        return {"alpha_raw": alpha_raw, "p_raw": p_raw, "u_raw": u_raw}

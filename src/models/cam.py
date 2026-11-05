"""Confounder alignment module for TAPB-inspired target-prior debiasing."""

from __future__ import annotations

from math import sqrt
from typing import Dict, Optional

import torch
from torch import nn


class CausalAttentionModule(nn.Module):
    """Align target features to target-prior confounder centers.

    TODO: Compare this lightweight alignment against the original TAPB CAM once
    the bias experiments are running.
    """

    def __init__(
        self,
        feature_dim: int,
        confounder_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        temperature: float = 1.0,
    ) -> None:
        """Create query/key/value projections for target-confounder alignment."""
        super().__init__()
        confounder_dim = confounder_dim or feature_dim
        hidden_dim = hidden_dim or feature_dim
        self.temperature = max(float(temperature), 1e-6)
        self.query_projection = nn.Linear(feature_dim, hidden_dim)
        self.key_projection = nn.Linear(confounder_dim, hidden_dim)
        self.value_projection = nn.Linear(confounder_dim, feature_dim)
        self.gate = nn.Linear(feature_dim * 2, feature_dim)
        self.output_norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        target_features: torch.Tensor,
        confounder_centers: torch.Tensor,
        p_ci: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return aligned target features and confounder attention weights."""
        if target_features.dim() != 2:
            raise ValueError("target_features must be shaped [batch, feature_dim].")
        if confounder_centers.dim() != 2:
            raise ValueError("confounder_centers must be shaped [num_clusters, dim].")

        keys = self.key_projection(confounder_centers.float())
        values = self.project_confounders(confounder_centers)
        queries = self.query_projection(target_features.float())
        scores = torch.matmul(queries, keys.transpose(0, 1))
        scores = scores / (sqrt(float(keys.size(-1))) * self.temperature)

        if p_ci is not None:
            prior = _normalize_prior(p_ci, keys.size(0), scores.device)
            scores = scores + torch.log(prior.clamp_min(1e-8)).unsqueeze(0)

        alignment = torch.softmax(scores, dim=-1)
        confounder_context = torch.matmul(alignment, values)
        gate = torch.sigmoid(self.gate(torch.cat([target_features, confounder_context], dim=-1)))
        t_ci = self.output_norm(gate * target_features + (1.0 - gate) * confounder_context)
        return {
            "T_ci": t_ci,
            "alignment": alignment,
            "confounder_context": confounder_context,
            "confounder_values": values,
        }

    def project_confounders(self, confounder_centers: torch.Tensor) -> torch.Tensor:
        """Project confounder centers into the target feature space."""
        return self.value_projection(confounder_centers.float())


def _normalize_prior(
    p_ci: torch.Tensor,
    num_clusters: int,
    device: torch.device,
) -> torch.Tensor:
    """Normalize cluster prior weights for attention scoring."""
    prior = p_ci.reshape(-1).float().to(device)
    if prior.numel() != num_clusters:
        raise ValueError("Attention cluster mass length must match its centers.")
    if not torch.isfinite(prior).all() or bool((prior < 0).any()) or float(prior.sum()) <= 0:
        raise ValueError("Attention cluster masses must be finite, nonnegative and have positive total.")
    return prior / prior.sum().clamp_min(1e-8)

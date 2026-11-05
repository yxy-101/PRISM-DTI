"""Feature fusion block for the raw EviDTI-style backbone."""

from __future__ import annotations

import torch
from torch import nn


class FeatureFusion(nn.Module):
    """Fuse two or more modality embeddings."""

    def __init__(
        self,
        input_dims: tuple,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        mode: str = "concat",
    ) -> None:
        """Create the raw feature fusion block."""
        super().__init__()
        self.mode = str(mode).strip().lower()
        if self.mode not in {"concat", "interaction"}:
            raise ValueError(
                "FeatureFusion mode must be 'concat' or 'interaction'."
            )
        self.num_modalities = len(input_dims)
        if self.num_modalities < 2:
            raise ValueError("FeatureFusion expects at least two modalities.")
        total_dim = int(sum(input_dims))
        if self.mode == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(total_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim),
            )
            self.modality_projections = None
            self.interaction_fusion = None
        else:
            self.fusion = None
            self.modality_projections = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(int(input_dim), hidden_dim),
                        nn.GELU(),
                        nn.LayerNorm(hidden_dim),
                    )
                    for input_dim in input_dims
                ]
            )
            interaction_dim = hidden_dim * (self.num_modalities ** 2)
            self.interaction_fusion = nn.Sequential(
                nn.Linear(interaction_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim),
            )

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        """Fuse modality embeddings with concat or explicit interactions."""
        if len(features) != self.num_modalities:
            raise ValueError(
                "FeatureFusion expected %d modalities, got %d."
                % (self.num_modalities, len(features))
            )
        if self.mode == "concat":
            return self.fusion(torch.cat(features, dim=-1))

        projected = [
            projection(feature)
            for projection, feature in zip(self.modality_projections, features)
        ]
        pairwise_products = []
        pairwise_differences = []
        for left_index, left in enumerate(projected):
            for right in projected[left_index + 1 :]:
                pairwise_products.append(left * right)
                pairwise_differences.append(torch.abs(left - right))
        interaction_features = torch.cat(
            projected + pairwise_products + pairwise_differences,
            dim=-1,
        )
        return self.interaction_fusion(interaction_features)

"""3D drug encoder wrapper for precomputed GeoGNN features."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from torch import nn

from src.data.feature_io import load_feature_tensor
from src.models.encoder_blocks import (
    AttentiveStatsPool,
    ResidualConvBlock,
    masked_max,
    masked_mean,
)


class GeoGNNFeatureLoader:
    """Load GeoGNN node or graph features from tensors or feature files.

    Expected batch keys:
    - `drug3d_features`: `[batch, nodes, dim]` or pooled `[batch, dim]`
    - `drug3d_feature_path`: path or list of paths to `.pt`, `.pth`, or `.npy`

    TODO: Support multi-conformer packing and geometry masks.
    """

    def __call__(self, batch: Dict[str, object]) -> torch.Tensor:
        """Return a batched GeoGNN tensor."""
        source = batch.get("drug3d_features", batch.get("drug3d_feature_path"))
        if source is None:
            raise KeyError(
                "Missing 3D drug input. Provide `drug3d_features` or "
                "`drug3d_feature_path`."
            )
        tensor = load_feature_tensor(source, feature_name="GeoGNN feature")
        single_file = isinstance(source, (str, Path))
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() == 2 and single_file:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() not in {2, 3}:
            raise ValueError(
                "GeoGNN feature must be 1D, 2D, or 3D, got %s."
                % (tuple(tensor.shape),)
            )
        return tensor.float()


class Drug3DEncoder(nn.Module):
    """Encode 3D drug features with the selected checkpoint architecture."""

    def __init__(
        self,
        input_dim: int = 300,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        num_conv_blocks: int = 2,
        conv_kernel_size: int = 5,
        use_attention_pool: bool = True,
        architecture: str = "enhanced",
    ) -> None:
        """Create the raw 3D drug feature encoder."""
        super().__init__()
        self.loader = GeoGNNFeatureLoader()
        self.architecture = _normalize_architecture(architecture)
        if self.architecture == "legacy":
            self.projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
            )
            return

        self.use_attention_pool = bool(use_attention_pool)
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.node_blocks = nn.ModuleList(
            [
                ResidualConvBlock(
                    hidden_dim=hidden_dim,
                    kernel_size=conv_kernel_size,
                    dropout=dropout,
                )
                for _ in range(max(0, int(num_conv_blocks)))
            ]
        )
        self.attention_pool = AttentiveStatsPool(hidden_dim, dropout=dropout)
        pooled_dim = (
            self.attention_pool.output_dim
            if self.use_attention_pool
            else hidden_dim * 2
        )
        self.output = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, batch: Dict[str, object]) -> torch.Tensor:
        """Return a pooled 3D drug representation."""
        features = self.loader(batch)
        if self.architecture == "legacy":
            if features.dim() == 3:
                features = torch.mean(features, dim=1)
            return self.projection(features)

        if features.dim() == 2:
            features = features.unsqueeze(1)
        mask = batch.get("drug3d_mask")
        mask_tensor = mask if torch.is_tensor(mask) else None
        encoded = self.input_projection(self.input_norm(features))
        for block in self.node_blocks:
            encoded = block(encoded, mask=mask_tensor)
        if self.use_attention_pool:
            pooled = self.attention_pool(encoded, mask=mask_tensor)
        else:
            pooled = torch.cat(
                [
                    masked_max(encoded, mask=mask_tensor),
                    masked_mean(encoded, mask=mask_tensor),
                ],
                dim=-1,
            )
        return self.output(pooled)


def _normalize_architecture(value: str) -> str:
    """Validate the encoder architecture used by a checkpoint."""
    architecture = str(value).strip().lower()
    if architecture not in {"enhanced", "legacy"}:
        raise ValueError(
            "encoder architecture must be 'enhanced' or 'legacy', got %r."
            % value
        )
    return architecture

"""Protein encoder for precomputed ProtTrans features."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn

from src.data.feature_io import load_feature_tensor
from src.models.encoder_blocks import (
    AttentiveStatsPool,
    FeatureTokenMixer,
    ResidualConvBlock,
    masked_mean,
    normalize_sequence_mask,
)
from src.models.light_attention import LightAttention


class ProtTransFeatureLoader:
    """Load ProtTrans features from tensors or external feature files.

    Expected batch keys:
    - `protein_features`: tensor shaped `[batch, seq_len, dim]`
    - `protein_feature_path`: path or list of paths to `.pt`, `.pth`, or `.npy`

    TODO: Add sequence-length metadata and memory-mapped caching.
    """

    def __call__(self, batch: Dict[str, object]) -> torch.Tensor:
        """Return a batched ProtTrans tensor."""
        source = batch.get("protein_features", batch.get("protein_feature_path"))
        if source is None:
            raise KeyError(
                "Missing protein input. Provide `protein_features` or "
                "`protein_feature_path`."
            )
        tensor = load_feature_tensor(source, feature_name="ProtTrans feature")
        return _ensure_batched_sequence(tensor, name="ProtTrans feature")


class ProteinEncoder(nn.Module):
    """Encode ProtTrans features with the selected checkpoint architecture."""

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 128,
        kernel_size: int = 9,
        dropout: float = 0.1,
        num_conv_blocks: int = 2,
        use_stats_pool: bool = True,
        feature_token_count: int = 1,
        transformer_layers: int = 0,
        transformer_heads: int = 4,
        architecture: str = "enhanced",
        pooling_mode: str = "attention",
    ) -> None:
        """Create the raw protein feature encoder."""
        super().__init__()
        self.loader = ProtTransFeatureLoader()
        self.architecture = _normalize_architecture(architecture)
        self.pooling_mode = _normalize_pooling_mode(pooling_mode)
        if self.architecture == "legacy":
            if self.pooling_mode != "attention":
                raise ValueError(
                    "legacy protein encoder supports attention pooling only."
                )
            self.input_projection = nn.Linear(input_dim, hidden_dim)
            self.attention = LightAttention(
                input_dim=hidden_dim,
                output_dim=hidden_dim,
                kernel_size=kernel_size,
                dropout=dropout,
            )
            self.output_norm = nn.LayerNorm(hidden_dim)
            return

        self.use_stats_pool = bool(use_stats_pool)
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.feature_mixer = FeatureTokenMixer(
            hidden_dim=hidden_dim,
            token_count=feature_token_count,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            dropout=dropout,
        )
        self.sequence_blocks = nn.ModuleList(
            [
                ResidualConvBlock(
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(max(0, int(num_conv_blocks)))
            ]
        )
        if self.pooling_mode == "masked_mean":
            self.mean_output = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim),
            )
            return
        self.attention = LightAttention(
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.stats_pool = AttentiveStatsPool(hidden_dim, dropout=dropout)
        pooled_dim = hidden_dim + (
            self.stats_pool.output_dim if self.use_stats_pool else 0
        )
        self.output = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, batch: Dict[str, object]) -> torch.Tensor:
        """Return a pooled protein representation."""
        encoded, mask_tensor = self.encode_sequence(batch)
        return self.pool_sequence(encoded, mask=mask_tensor)

    def encode_sequence(
        self,
        batch: Dict[str, object],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return encoded residue tokens before protein sequence pooling."""
        features = self.loader(batch)
        mask = batch.get("protein_mask")
        mask_tensor = mask if torch.is_tensor(mask) else None
        if self.architecture == "legacy":
            return self.input_projection(features), mask_tensor

        encoded = self.input_projection(self.input_norm(features))
        encoded, mask_tensor = self.feature_mixer(encoded, mask=mask_tensor)
        for block in self.sequence_blocks:
            encoded = block(encoded, mask=mask_tensor)
        return encoded, mask_tensor

    def pool_sequence(
        self,
        encoded: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pool already encoded residue tokens using the configured architecture."""
        if self.architecture == "legacy":
            pooled = self.attention(encoded, mask=mask)
            return self.output_norm(pooled)

        if self.pooling_mode == "masked_mean":
            normalized_mask = None
            if mask is not None:
                normalized_mask = normalize_sequence_mask(
                    mask,
                    seq_len=encoded.size(1),
                    device=encoded.device,
                )
            return self.mean_output(masked_mean(encoded, normalized_mask))

        light_attention = self.attention(encoded, mask=mask)
        if self.use_stats_pool:
            pooled = torch.cat(
                [light_attention, self.stats_pool(encoded, mask=mask)],
                dim=-1,
            )
        else:
            pooled = light_attention
        return self.output(pooled)


def _ensure_batched_sequence(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Normalize a sequence tensor to `[batch, seq_len, dim]`."""
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 3:
        raise ValueError("%s must be 2D or 3D, got %s." % (name, tuple(tensor.shape)))
    return tensor.float()


def _normalize_architecture(value: str) -> str:
    """Validate the encoder architecture used by a checkpoint."""
    architecture = str(value).strip().lower()
    if architecture not in {"enhanced", "legacy"}:
        raise ValueError(
            "encoder architecture must be 'enhanced' or 'legacy', got %r."
            % value
        )
    return architecture


def _normalize_pooling_mode(value: str) -> str:
    """Validate the optional cold-target protein pooling path."""
    mode = str(value).strip().lower()
    if mode not in {"attention", "masked_mean"}:
        raise ValueError(
            "protein pooling mode must be 'attention' or 'masked_mean', got %r."
            % value
        )
    return mode

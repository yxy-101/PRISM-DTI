"""2D drug encoder for precomputed MG-BERT features."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from src.data.feature_io import load_feature_tensor
from src.models.encoder_blocks import (
    AttentiveStatsPool,
    FeatureTokenMixer,
    ResidualConvBlock,
    masked_max,
    masked_mean,
)


class MGBERTFeatureLoader:
    """Load MG-BERT token features from tensors or external feature files.

    Expected batch keys:
    - `drug2d_features`: tensor shaped `[batch, seq_len, dim]`
    - `drug2d_feature_path`: path or list of paths to `.pt`, `.pth`, or `.npy`

    TODO: Align token masks with the future molecular tokenizer pipeline.
    """

    def __call__(self, batch: Dict[str, object]) -> torch.Tensor:
        """Return a batched MG-BERT tensor."""
        source = batch.get("drug2d_features", batch.get("drug2d_feature_path"))
        if source is None:
            raise KeyError(
                "Missing 2D drug input. Provide `drug2d_features` or "
                "`drug2d_feature_path`."
            )
        tensor = load_feature_tensor(source, feature_name="MG-BERT feature")
        return _ensure_batched_sequence(tensor, name="MG-BERT feature")


class Drug2DEncoder(nn.Module):
    """Encode 2D drug features with the selected checkpoint architecture."""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        num_conv_blocks: int = 2,
        conv_kernel_size: int = 5,
        use_attention_pool: bool = True,
        feature_token_count: int = 1,
        transformer_layers: int = 0,
        transformer_heads: int = 4,
        architecture: str = "enhanced",
    ) -> None:
        """Create the raw 2D drug feature encoder."""
        super().__init__()
        self.loader = MGBERTFeatureLoader()
        self.architecture = _normalize_architecture(architecture)
        if self.architecture == "legacy":
            self.input_projection = nn.Linear(input_dim, hidden_dim)
            self.cnn = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.dropout = nn.Dropout(dropout)
            self.output = nn.Linear(hidden_dim * 2, hidden_dim)
            return

        self.use_attention_pool = bool(use_attention_pool)
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
        self.conv_blocks = nn.ModuleList(
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
        """Return a pooled 2D drug representation."""
        features = self.loader(batch)
        if self.architecture == "legacy":
            projected = self.input_projection(features).transpose(1, 2)
            encoded = self.cnn(projected)
            max_pooled = torch.amax(encoded, dim=-1)
            mean_pooled = torch.mean(encoded, dim=-1)
            combined = torch.cat([max_pooled, mean_pooled], dim=-1)
            return self.output(self.dropout(combined))

        mask = batch.get("drug2d_mask")
        mask_tensor = mask if torch.is_tensor(mask) else None
        encoded = self.input_projection(self.input_norm(features))
        encoded, mask_tensor = self.feature_mixer(encoded, mask=mask_tensor)
        for block in self.conv_blocks:
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

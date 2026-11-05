"""Reusable sequence encoder blocks for protein and drug feature heads."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


class ResidualConvBlock(nn.Module):
    """Depthwise-separable residual convolution over sequence positions."""

    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int = 5,
        dropout: float = 0.1,
        dilation: int = 1,
    ) -> None:
        """Create a residual temporal/node convolution block."""
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        if dilation <= 0:
            raise ValueError("dilation must be positive.")
        padding = (kernel_size // 2) * dilation
        self.norm = nn.LayerNorm(hidden_dim)
        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=hidden_dim,
        )
        self.pointwise_in = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=1)
        self.pointwise_out = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return residual-enhanced `[batch, seq_len, hidden_dim]` features."""
        if features.dim() != 3:
            raise ValueError(
                "ResidualConvBlock expects [batch, seq_len, dim], got %s."
                % (tuple(features.shape),)
            )
        normalized = self.norm(features).transpose(1, 2)
        encoded = self.depthwise(normalized)
        encoded = self.pointwise_in(encoded)
        encoded = self.activation(encoded)
        encoded = self.dropout(encoded)
        encoded = self.pointwise_out(encoded).transpose(1, 2)
        encoded = self.dropout(encoded)
        output = features + encoded
        if mask is None:
            return output
        broadcast_mask = normalize_sequence_mask(
            mask,
            seq_len=output.size(1),
            device=output.device,
        ).unsqueeze(-1)
        return output.masked_fill(~broadcast_mask, 0.0)


class FeatureTokenMixer(nn.Module):
    """Expand pooled features into learned tokens and mix them with attention."""

    def __init__(
        self,
        hidden_dim: int,
        token_count: int = 8,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        """Create a lightweight Transformer mixer for encoder feature heads."""
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.token_count = max(1, int(token_count))
        self.num_layers = max(0, int(num_layers))
        self.disabled = self.token_count <= 1 and self.num_layers == 0
        self.expand_norm = nn.LayerNorm(hidden_dim)
        self.token_expand = nn.Linear(hidden_dim, self.token_count * hidden_dim)
        self.token_embedding = nn.Parameter(
            torch.zeros(1, self.token_count, hidden_dim)
        )
        nn.init.trunc_normal_(self.token_embedding, std=0.02)
        if self.num_layers > 0:
            resolved_heads = _resolve_num_heads(hidden_dim, int(num_heads))
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=resolved_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=self.num_layers)
        else:
            self.encoder = None
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return mixed features and the corresponding normalized mask."""
        if features.dim() != 3:
            raise ValueError(
                "FeatureTokenMixer expects [batch, seq_len, dim], got %s."
                % (tuple(features.shape),)
            )
        batch_size, seq_len, hidden_dim = features.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                "FeatureTokenMixer hidden_dim mismatch: expected %d, got %d."
                % (self.hidden_dim, hidden_dim)
            )
        if self.disabled:
            normalized_mask = None
            if mask is not None:
                normalized_mask = normalize_sequence_mask(
                    mask,
                    seq_len=seq_len,
                    device=features.device,
                )
            return features, normalized_mask

        if seq_len == 1 and self.token_count > 1:
            expanded = self.token_expand(self.expand_norm(features[:, 0, :]))
            features = expanded.view(batch_size, self.token_count, hidden_dim)
            features = features + self.token_embedding
            normalized_mask = torch.ones(
                batch_size,
                self.token_count,
                dtype=torch.bool,
                device=features.device,
            )
        else:
            normalized_mask = None
            if mask is not None:
                normalized_mask = normalize_sequence_mask(
                    mask,
                    seq_len=features.size(1),
                    device=features.device,
                )

        if self.encoder is not None:
            key_padding_mask = None
            if normalized_mask is not None:
                key_padding_mask = ~normalized_mask
            features = self.encoder(features, src_key_padding_mask=key_padding_mask)

        features = self.output_norm(features)
        if normalized_mask is None:
            return features, normalized_mask
        return features.masked_fill(~normalized_mask.unsqueeze(-1), 0.0), normalized_mask


class AttentiveStatsPool(nn.Module):
    """Attention, mean, max, and uncertainty-aware stats pooling."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        """Create a scalar attention pooler for sequence/node features."""
        super().__init__()
        score_hidden_dim = max(32, hidden_dim // 2)
        self.score = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, score_hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(score_hidden_dim, 1),
        )
        self.output_dim = hidden_dim * 4

    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return pooled `[batch, hidden_dim * 4]` representation."""
        if features.dim() != 3:
            raise ValueError(
                "AttentiveStatsPool expects [batch, seq_len, dim], got %s."
                % (tuple(features.shape),)
            )
        normalized_mask = None
        if mask is not None:
            normalized_mask = normalize_sequence_mask(
                mask,
                seq_len=features.size(1),
                device=features.device,
            )

        scores = self.score(features).squeeze(-1)
        values = features
        if normalized_mask is not None:
            scores = scores.masked_fill(~normalized_mask, -1e9)
            values = values.masked_fill(~normalized_mask.unsqueeze(-1), 0.0)

        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        attention_mean = torch.sum(values * weights, dim=1)
        centered = values - attention_mean.unsqueeze(1)
        variance = torch.sum(centered.pow(2) * weights, dim=1)
        attention_std = torch.sqrt(variance.clamp_min(1e-8))
        arithmetic_mean = masked_mean(values, normalized_mask)
        max_pooled = masked_max(features, normalized_mask)
        return torch.cat(
            [attention_mean, arithmetic_mean, max_pooled, attention_std],
            dim=-1,
        )


def masked_mean(
    features: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return mean pooled features with optional boolean mask."""
    if mask is None:
        return torch.mean(features, dim=1)
    weights = mask.to(dtype=features.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return torch.sum(features * weights, dim=1) / denom


def masked_max(
    features: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return max pooled features with optional boolean mask."""
    if mask is None:
        return torch.amax(features, dim=1)
    masked = features.masked_fill(~mask.unsqueeze(-1), -torch.inf)
    pooled = torch.amax(masked, dim=1)
    return torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))


def normalize_sequence_mask(
    mask: torch.Tensor,
    seq_len: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Normalize sequence masks to boolean `[batch, seq_len]` tensors."""
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    if mask.dim() == 3 and mask.size(-1) == 1:
        mask = mask.squeeze(-1)
    if mask.dim() != 2 or mask.size(-1) != seq_len:
        raise ValueError("Mask must be shaped [batch, seq_len].")
    mask = mask.to(device=device, dtype=torch.bool)
    valid_rows = mask.any(dim=-1, keepdim=True)
    return torch.where(valid_rows, mask, torch.ones_like(mask, dtype=torch.bool))


def _resolve_num_heads(hidden_dim: int, requested_heads: int) -> int:
    """Return a valid attention-head count that divides hidden_dim."""
    requested_heads = max(1, int(requested_heads))
    hidden_dim = int(hidden_dim)
    for heads in range(min(requested_heads, hidden_dim), 0, -1):
        if hidden_dim % heads == 0:
            return heads
    return 1

"""Light attention pooling for the raw EviDTI-style protein branch."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class LightAttention(nn.Module):
    """Pool sequence features with lightweight convolutional attention.

    TODO: Tune kernel size and dropout after aligning against reference EviDTI.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: Optional[int] = None,
        kernel_size: int = 9,
        dropout: float = 0.1,
    ) -> None:
        """Create value and score convolutions over sequence positions."""
        super().__init__()
        hidden_dim = output_dim or input_dim
        padding = kernel_size // 2
        self.value_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=padding)
        self.score_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=padding)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(
        self, features: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Return a pooled representation for `[batch, seq_len, dim]` features."""
        if features.dim() != 3:
            raise ValueError(
                "LightAttention expects [batch, seq_len, dim], got %s."
                % (tuple(features.shape),)
            )

        sequence = features.transpose(1, 2)
        values = torch.tanh(self.value_conv(sequence))
        scores = self.score_conv(sequence)

        if mask is not None:
            broadcast_mask = _normalize_mask(mask, scores.size(-1)).unsqueeze(1)
            scores = scores.masked_fill(~broadcast_mask, -1e9)
            values = values.masked_fill(~broadcast_mask, 0.0)

        attention = torch.softmax(scores, dim=-1)
        attended = torch.sum(values * attention, dim=-1)
        max_pooled = torch.amax(values, dim=-1)
        combined = torch.cat([attended, max_pooled], dim=-1)
        return self.output(self.dropout(combined))


def _normalize_mask(mask: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Normalize sequence masks to boolean `[batch, seq_len]` tensors."""
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    if mask.dim() != 2 or mask.size(-1) != seq_len:
        raise ValueError("Mask must be shaped [batch, seq_len].")
    mask = mask.to(dtype=torch.bool)
    valid_rows = mask.any(dim=-1, keepdim=True)
    return torch.where(valid_rows, mask, torch.ones_like(mask, dtype=torch.bool))

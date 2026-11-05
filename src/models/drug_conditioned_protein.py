"""Drug-conditioned residue attention for strict unseen-target transfer."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn

from src.models.encoder_blocks import normalize_sequence_mask


class DrugConditionedProteinAttention(nn.Module):
    """Use pooled drug modalities to select informative protein residues."""

    def __init__(
        self,
        protein_dim: int,
        drug2d_dim: int,
        drug3d_dim: int,
        dropout: float = 0.1,
        update_mode: str = "gated_blend",
    ) -> None:
        """Create one lightweight cross-modal residue-attention block."""
        super().__init__()
        self.protein_dim = int(protein_dim)
        self.update_mode = str(update_mode).strip().lower()
        if self.update_mode not in {"gated_blend", "identity_residual"}:
            raise ValueError(
                "Unsupported drug-conditioned protein update mode: %s"
                % update_mode
            )
        self.drug_query = nn.Sequential(
            nn.Linear(int(drug2d_dim) + int(drug3d_dim), self.protein_dim),
            nn.GELU(),
            nn.LayerNorm(self.protein_dim),
        )
        self.key = nn.Linear(self.protein_dim, self.protein_dim, bias=False)
        self.value = nn.Linear(self.protein_dim, self.protein_dim, bias=False)
        self.attention_dropout = nn.Dropout(dropout)
        self.candidate = nn.Sequential(
            nn.Linear(self.protein_dim * 4, self.protein_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.protein_dim * 2, self.protein_dim),
            nn.LayerNorm(self.protein_dim),
        )
        self.gate = nn.Linear(self.protein_dim * 3, self.protein_dim)
        nn.init.constant_(self.gate.bias, -1.5)
        if self.update_mode == "gated_blend":
            self.output_norm = nn.LayerNorm(self.protein_dim)
            self.residual_projection = None
        else:
            self.output_norm = None
            self.residual_projection = nn.Linear(
                self.protein_dim,
                self.protein_dim,
                bias=False,
            )
            nn.init.zeros_(self.residual_projection.weight)

    def forward(
        self,
        residue_tokens: torch.Tensor,
        pooled_protein: torch.Tensor,
        drug2d: torch.Tensor,
        drug3d: torch.Tensor,
        residue_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return drug-conditioned protein vectors and residue weights."""
        if residue_tokens.dim() != 3:
            raise ValueError("residue_tokens must be [batch, residues, dim].")
        if residue_tokens.size(-1) != self.protein_dim:
            raise ValueError(
                "Residue dimension mismatch: expected %d, got %d."
                % (self.protein_dim, residue_tokens.size(-1))
            )
        query = self.drug_query(torch.cat([drug2d, drug3d], dim=-1))
        keys = self.key(residue_tokens)
        values = self.value(residue_tokens)
        logits = torch.sum(keys * query.unsqueeze(1), dim=-1) / math.sqrt(
            float(self.protein_dim)
        )
        if residue_mask is not None:
            normalized_mask = normalize_sequence_mask(
                residue_mask,
                seq_len=residue_tokens.size(1),
                device=residue_tokens.device,
            )
            logits = logits.masked_fill(~normalized_mask, -1e9)
        weights = torch.softmax(logits, dim=-1)
        dropped_weights = self.attention_dropout(weights)
        context = torch.sum(values * dropped_weights.unsqueeze(-1), dim=1)
        candidate = self.candidate(
            torch.cat(
                [
                    pooled_protein,
                    context,
                    pooled_protein * context,
                    torch.abs(pooled_protein - context),
                ],
                dim=-1,
            )
        )
        gate = torch.sigmoid(
            self.gate(torch.cat([pooled_protein, context, query], dim=-1))
        )
        if self.update_mode == "gated_blend":
            conditioned = self.output_norm(
                (1.0 - gate) * pooled_protein + gate * candidate
            )
        else:
            conditioned = pooled_protein + gate * self.residual_projection(candidate)
        return conditioned, weights

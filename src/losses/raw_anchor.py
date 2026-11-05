"""Reference-anchor loss for preserving the raw evidential prediction path."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def raw_anchor_loss(
    outputs: Dict[str, torch.Tensor],
    reference_outputs: Dict[str, torch.Tensor],
    probability_weight: float = 1.0,
    uncertainty_weight: float = 1.0,
    evidence_weight: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Measure raw-output drift from a frozen pre-intervention reference.

    The function deliberately anchors only ``alpha_raw``, ``p_raw`` and
    ``u_raw``.  It does not constrain the adjusted branch, which remains free
    to respond to the target-prior intervention.
    """
    required = {"alpha_raw", "p_raw", "u_raw"}
    missing = sorted(
        (required - set(outputs)) | (required - set(reference_outputs))
    )
    if missing:
        raise KeyError("raw anchor is missing output keys: %s" % missing)

    weights = {
        "probability": float(probability_weight),
        "uncertainty": float(uncertainty_weight),
        "evidence": float(evidence_weight),
    }
    if any(value < 0.0 for value in weights.values()):
        raise ValueError("raw anchor component weights must be non-negative")
    if sum(weights.values()) <= 0.0:
        raise ValueError("raw anchor requires at least one positive component weight")

    probability = F.mse_loss(
        outputs["p_raw"], reference_outputs["p_raw"].detach()
    )
    uncertainty = F.smooth_l1_loss(
        outputs["u_raw"], reference_outputs["u_raw"].detach()
    )
    evidence_current = outputs["alpha_raw"].sum(dim=-1).clamp_min(1e-8)
    evidence_reference = (
        reference_outputs["alpha_raw"].detach().sum(dim=-1).clamp_min(1e-8)
    )
    evidence = F.smooth_l1_loss(
        torch.log(evidence_current), torch.log(evidence_reference)
    )
    total = (
        weights["probability"] * probability
        + weights["uncertainty"] * uncertainty
        + weights["evidence"] * evidence
    )
    return {
        "loss_total": total,
        "probability": probability,
        "uncertainty": uncertainty,
        "log_evidence": evidence,
    }

"""Derived raw-versus-adjusted evidential audit quantities."""

from __future__ import annotations

from typing import Dict

import torch


def derive_evidential_contrasts(
    outputs: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Return probability, uncertainty and evidence-strength contrasts.

    Positive ``delta_probability`` means the adjusted positive probability is
    lower than the raw positive probability. Positive ``delta_uncertainty``
    means uncertainty increases after adjustment. Positive
    ``delta_log_evidence`` means total Dirichlet concentration contracts.
    """
    required = {"alpha_raw", "p_raw", "u_raw"}
    missing = sorted(required - set(outputs))
    if missing:
        raise KeyError("evidential audit is missing raw keys: %s" % missing)

    alpha_do = outputs.get("alpha_do", outputs["alpha_raw"])
    p_do = outputs.get("p_do", outputs["p_raw"])
    u_do = outputs.get("u_do", outputs["u_raw"])
    evidence_raw = outputs["alpha_raw"].sum(dim=-1).clamp_min(1e-8)
    evidence_do = alpha_do.sum(dim=-1).clamp_min(1e-8)
    p_raw_positive = outputs["p_raw"][:, -1]
    p_do_positive = p_do[:, -1]
    u_raw = outputs["u_raw"].reshape(-1)
    u_do = u_do.reshape(-1)
    return {
        "delta_probability": p_raw_positive - p_do_positive,
        "delta_uncertainty": u_do - u_raw,
        "delta_log_evidence": torch.log(evidence_raw) - torch.log(evidence_do),
        # For a K-class Dirichlet, u=K/S.  Therefore the uncertainty change
        # is exactly the negative concentration change (up to the log scale):
        # delta_uncertainty > 0 iff delta_log_evidence > 0.  Keep this flag in
        # the machine-readable audit so downstream reports cannot count both
        # quantities as independent evidence axes.
        "uncertainty_concentration_equivalent": torch.eq(
            u_do > u_raw,
            evidence_do < evidence_raw,
        ),
        "independent_evidence_axis": torch.zeros_like(u_raw, dtype=torch.bool),
        "evidence_raw": evidence_raw,
        "evidence_do": evidence_do,
    }

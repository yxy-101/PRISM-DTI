"""Prior-Orthogonal Evidential Regularization losses."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


DEFAULT_CONFIDENCE_FEATURES = (
    "logit_p_raw_positive",
    "u_raw",
    "pcs",
    "evidence_strength",
)
PRIOR_GROUP_TO_INDEX = {
    "low": 0,
    "0": 0,
    0: 0,
    "mid": 1,
    "middle": 1,
    "medium": 1,
    "1": 1,
    1: 1,
    "high": 2,
    "2": 2,
    2: 2,
}


def build_confidence_features(
    outputs: Mapping[str, torch.Tensor],
    confidence_features: Optional[Sequence[str]] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Build z_conf from required raw evidential outputs.

    The default feature order is:
    [logit(p_raw_positive), u_raw, pcs, evidence_strength].
    """
    feature_names = tuple(confidence_features or DEFAULT_CONFIDENCE_FEATURES)
    required = {"p_raw", "u_raw", "pcs", "alpha_raw"}
    missing = sorted(name for name in required if name not in outputs)
    if missing:
        raise KeyError("POER requires model outputs missing from batch: %s" % missing)

    p_raw = outputs["p_raw"]
    u_raw = outputs["u_raw"]
    pcs = outputs["pcs"]
    alpha_raw = outputs["alpha_raw"]
    if p_raw.dim() != 2 or p_raw.size(1) < 2:
        raise ValueError("POER expects p_raw shape [batch, num_classes>=2].")
    if alpha_raw.dim() != 2 or alpha_raw.size(0) != p_raw.size(0):
        raise ValueError("POER expects alpha_raw shape [batch, num_classes].")

    p_pos = p_raw[:, -1].clamp(float(eps), 1.0 - float(eps))
    feature_values = {
        "p_raw_positive": p_pos,
        "logit_p_raw_positive": torch.logit(p_pos),
        "u_raw": u_raw.reshape(-1),
        "pcs": pcs.reshape(-1),
        "evidence_strength": alpha_raw.sum(dim=-1),
    }

    columns: List[torch.Tensor] = []
    for feature_name in feature_names:
        if feature_name not in feature_values:
            raise ValueError(
                "Unsupported POER confidence feature %r. Supported features: %s"
                % (feature_name, sorted(feature_values))
            )
        value = feature_values[feature_name]
        if value.size(0) != p_raw.size(0):
            raise ValueError(
                "POER feature %s batch size mismatch: %d vs %d."
                % (feature_name, value.size(0), p_raw.size(0))
            )
        columns.append(value)

    z_conf = torch.stack(columns, dim=-1)
    if not bool(torch.isfinite(z_conf).all().item()):
        raise ValueError("POER confidence features contain non-finite values.")
    return z_conf


def ppce_proxy_loss(
    outputs: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    tau: float = 0.8,
) -> torch.Tensor:
    """Penalize high raw positive confidence and positive PCS on negatives."""
    required = {"p_raw", "pcs"}
    missing = sorted(name for name in required if name not in outputs)
    if missing:
        raise KeyError("POER PPCE proxy requires missing outputs: %s" % missing)

    p_pos = outputs["p_raw"][:, -1]
    pcs = outputs["pcs"].reshape(-1)
    labels = labels.reshape(-1).to(device=p_pos.device)
    negative_mask = labels == 0
    if not bool(negative_mask.any().item()):
        return p_pos.sum() * 0.0

    negative_p_pos = p_pos[negative_mask]
    negative_pcs = pcs[negative_mask]
    raw_conf_penalty = F.relu(negative_p_pos - float(tau)).pow(2).mean()
    pcs_penalty = F.relu(negative_pcs).pow(2).mean()
    return raw_conf_penalty + pcs_penalty


def prior_adversary_loss(
    prior_logits: torch.Tensor,
    target_prior_group: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy for the target-prior group adversary."""
    target_prior_group = target_prior_group.reshape(-1).to(
        device=prior_logits.device,
        dtype=torch.long,
    )
    if prior_logits.size(0) != target_prior_group.size(0):
        raise ValueError(
            "POER adversary batch mismatch: logits=%d groups=%d."
            % (prior_logits.size(0), target_prior_group.size(0))
        )
    return F.cross_entropy(prior_logits, target_prior_group)


def poer_total_loss(
    original_loss: torch.Tensor,
    outputs: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    target_prior_group: torch.Tensor,
    prior_adversary: torch.nn.Module,
    lambda_adv: float = 0.01,
    lambda_ppce: float = 0.05,
    tau: float = 0.8,
    confidence_features: Optional[Sequence[str]] = None,
    grl_scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Return total POER loss and its components."""
    z_conf = build_confidence_features(outputs, confidence_features=confidence_features)
    prior_logits = prior_adversary(z_conf, grl_scale=grl_scale)
    loss_prior_adv = prior_adversary_loss(prior_logits, target_prior_group)
    loss_ppce = ppce_proxy_loss(outputs, labels, tau=tau)
    loss_total = (
        original_loss
        + float(lambda_adv) * loss_prior_adv
        + float(lambda_ppce) * loss_ppce
    )
    if not bool(torch.isfinite(loss_total).all().item()):
        raise ValueError("POER total loss is non-finite.")
    return {
        "loss_total": loss_total,
        "loss_original": original_loss,
        "loss_prior_adv": loss_prior_adv,
        "loss_ppce_proxy": loss_ppce,
        "z_conf": z_conf,
        "prior_logits": prior_logits,
    }


def target_prior_groups_to_tensor(
    groups: Iterable[Any],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Convert low/mid/high target-prior groups into class indices."""
    indices = []
    for group in groups:
        key = str(group).strip().lower() if not isinstance(group, int) else group
        if key not in PRIOR_GROUP_TO_INDEX:
            raise ValueError("Unknown target_prior_group for POER: %r" % (group,))
        indices.append(PRIOR_GROUP_TO_INDEX[key])
    return torch.as_tensor(indices, dtype=torch.long, device=device)


def build_target_prior_group_mapping(
    records: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, str], Dict[str, float], Dict[str, Any]]:
    """Construct target high/mid/low groups from train positive rates."""
    target_counts: Dict[str, int] = defaultdict(int)
    target_positive_counts: Dict[str, int] = defaultdict(int)
    for record in records:
        target_id = str(record["target_id"])
        label = int(record["label"])
        target_counts[target_id] += 1
        target_positive_counts[target_id] += label
    if not target_counts:
        raise ValueError("Cannot build POER target-prior groups from empty records.")

    rates = {
        target_id: target_positive_counts[target_id] / float(count)
        for target_id, count in target_counts.items()
    }
    rate_values = sorted(rates.values())
    low_cut = _quantile(rate_values, 1.0 / 3.0)
    high_cut = _quantile(rate_values, 2.0 / 3.0)

    groups = {}
    for target_id, rate in rates.items():
        if rate <= low_cut:
            group = "low"
        elif rate >= high_cut:
            group = "high"
        else:
            group = "mid"
        groups[target_id] = group

    group_counts = dict(Counter(groups.values()))
    summary = {
        "source": "constructed_from_train_positive_rate",
        "target_count": len(groups),
        "sample_count": len(records),
        "low_cut": low_cut,
        "high_cut": high_cut,
        "group_counts": group_counts,
    }
    return groups, rates, summary


def attach_target_prior_groups(
    records: Sequence[Dict[str, Any]],
    prior_target: str = "target_prior_group",
    prior_records: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Ensure records have a POER target_prior_group field.

    If the field is absent, groups are constructed from P_train(label=1 |
    target_id). Pass full training records as prior_records when the current
    batch/dataset is sampled.
    """
    if not records:
        raise ValueError("Cannot attach POER target-prior groups to empty records.")

    existing = [
        str(record.get(prior_target, "")).strip().lower()
        for record in records
        if str(record.get(prior_target, "")).strip()
    ]
    if len(existing) == len(records) and all(value in PRIOR_GROUP_TO_INDEX for value in existing):
        group_counts = dict(Counter(existing))
        return {
            "prior_target": prior_target,
            "source": "batch_or_dataset_field",
            "constructed": False,
            "sample_count": len(records),
            "group_counts": group_counts,
        }

    source_records = list(prior_records) if prior_records is not None else list(records)
    mapping, rates, summary = build_target_prior_group_mapping(source_records)
    missing_targets = []
    for record in records:
        target_id = str(record["target_id"])
        if target_id not in mapping:
            missing_targets.append(target_id)
            continue
        record[prior_target] = mapping[target_id]
        record["target_prior_train_positive_rate"] = float(rates[target_id])
    if missing_targets:
        raise ValueError(
            "Cannot attach POER target-prior groups; targets absent from train prior source: %s"
            % sorted(set(missing_targets))[:10]
        )

    summary.update(
        {
            "prior_target": prior_target,
            "constructed": True,
            "source": "constructed_from_train_csv_positive_rate"
            if prior_records is not None
            else "constructed_from_dataset_positive_rate",
            "attached_sample_count": len(records),
        }
    )
    return summary


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    """Return a linearly interpolated quantile from sorted values."""
    if not sorted_values:
        raise ValueError("Cannot compute quantile of empty values.")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * float(q)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)

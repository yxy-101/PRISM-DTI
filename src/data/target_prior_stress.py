"""Train-only target-prior stress weights for controlled bias audits.

The stress intervention changes only the contribution of training examples.
Validation and test records are never consulted.  A positive strength amplifies
labels that agree with each target's smoothed deviation from the global
training positive rate; strength zero is an exact no-op control.
"""

from __future__ import annotations

from collections import defaultdict
import random
from math import exp, log
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple


def attach_target_prior_stress_weights(
    records: Sequence[MutableMapping[str, Any]],
    prior_records: Sequence[Mapping[str, Any]],
    strength: float,
    smoothing: float = 10.0,
    min_weight: float = 0.2,
    max_weight: float = 5.0,
    normalize_mean: bool = True,
    assignment_mode: str = "target_prior",
    shuffle_seed: int = 7,
) -> Dict[str, Any]:
    """Attach exponential-tilting weights derived from training labels only.

    For target ``j``, the smoothed training positive rate is ``theta_j`` and
    the global training rate is ``theta_global``.  A sample with binary label
    ``y`` receives an unnormalised weight

    ``exp(strength * (2*y - 1) * (logit(theta_j) - logit(theta_global)))``.

    Thus the intervention strengthens the direction already present in the
    training target prior without relabelling any pair.  It is intended as a
    controlled stress test, not as a performance-improvement reweighting rule.
    """
    strength = float(strength)
    smoothing = float(smoothing)
    min_weight = float(min_weight)
    max_weight = float(max_weight)
    assignment_mode = str(assignment_mode)
    if assignment_mode not in {"target_prior", "shuffled_within_class"}:
        raise ValueError(
            "target-prior stress assignment_mode must be target_prior or "
            "shuffled_within_class"
        )
    if strength < 0.0:
        raise ValueError("target-prior stress strength must be non-negative")
    if smoothing < 0.0:
        raise ValueError("target-prior stress smoothing must be non-negative")
    if min_weight <= 0.0 or max_weight < min_weight:
        raise ValueError("target-prior stress weights require 0 < min <= max")
    if not prior_records:
        raise ValueError("target-prior stress requires non-empty training records")

    target_total: Dict[str, int] = defaultdict(int)
    target_positive: Dict[str, int] = defaultdict(int)
    global_positive = 0
    for record in prior_records:
        target_id, label = _target_and_label(record)
        target_total[target_id] += 1
        target_positive[target_id] += label
        global_positive += label

    global_rate = _bounded_probability(global_positive / len(prior_records))
    global_logit = _logit(global_rate)
    target_bias: Dict[str, float] = {}
    for target_id, total in target_total.items():
        smoothed_rate = (
            target_positive[target_id] + smoothing * global_rate
        ) / (total + smoothing)
        target_bias[target_id] = _logit(_bounded_probability(smoothed_rate)) - global_logit

    weights: List[float] = []
    for record in records:
        target_id, label = _target_and_label(record)
        if target_id not in target_bias:
            raise KeyError(
                "target-prior stress record is absent from the training prior: %s"
                % target_id
            )
        signed_label = 1.0 if label == 1 else -1.0
        weights.append(exp(strength * signed_label * target_bias[target_id]))

    if normalize_mean and weights:
        mean_weight = sum(weights) / len(weights)
        if mean_weight > 0.0:
            weights = [weight / mean_weight for weight in weights]
    weights = [min(max(weight, min_weight), max_weight) for weight in weights]

    # Matched negative control: shuffle the treated weights among examples
    # within each label. This preserves each class's exact empirical weight
    # distribution, weighted mass, and effective sample size while breaking
    # the assignment between target identity and its prior-derived weights.
    if assignment_mode == "shuffled_within_class":
        rng = random.Random(int(shuffle_seed))
        class_indices: Dict[int, List[int]] = defaultdict(list)
        for index, record in enumerate(records):
            _target_id, label = _target_and_label(record)
            class_indices[label].append(index)
        for indices in class_indices.values():
            shuffled_weights = [weights[index] for index in indices]
            rng.shuffle(shuffled_weights)
            for index, weight in zip(indices, shuffled_weights):
                weights[index] = weight

    for record, weight in zip(records, weights):
        if "sample_weight" in record:
            raise ValueError(
                "target-prior stress must be isolated from another sample-weight intervention"
            )
        record["sample_weight"] = float(weight)

    absolute_biases = [abs(value) for value in target_bias.values()]
    class_weight_sums = {
        str(label): sum(
            weight
            for record, weight in zip(records, weights)
            if int(record["label"]) == label
        )
        for label in (0, 1)
    }
    class_ess = {}
    for label in (0, 1):
        values = [
            weight
            for record, weight in zip(records, weights)
            if int(record["label"]) == label
        ]
        denominator = sum(value * value for value in values)
        class_ess[str(label)] = (
            (sum(values) ** 2 / denominator) if denominator > 0.0 else 0.0
        )
    sum_squared = sum(weight * weight for weight in weights)
    return {
        "enabled": True,
        "definition": "train_only_exponential_target_prior_tilt",
        "assignment_mode": assignment_mode,
        "shuffle_seed": int(shuffle_seed) if assignment_mode == "shuffled_within_class" else None,
        "strength": strength,
        "smoothing": smoothing,
        "normalize_mean": bool(normalize_mean),
        "min_weight": min_weight,
        "max_weight": max_weight,
        "sample_count": len(weights),
        "target_count": len(target_total),
        "global_positive_rate": global_rate,
        "mean_absolute_target_logit_deviation": (
            sum(absolute_biases) / len(absolute_biases)
            if absolute_biases
            else 0.0
        ),
        "weight_min": min(weights) if weights else None,
        "weight_max": max(weights) if weights else None,
        "weight_mean": sum(weights) / len(weights) if weights else None,
        "effective_sample_size": (
            (sum(weights) ** 2 / sum_squared) if sum_squared > 0.0 else 0.0
        ),
        "class_weight_sums": class_weight_sums,
        "class_effective_sample_size": class_ess,
    }


def _target_and_label(record: Mapping[str, Any]) -> Tuple[str, int]:
    target_id = str(record.get("target_id", "")).strip()
    if not target_id:
        raise ValueError("target-prior stress record is missing target_id")
    label = int(record.get("label", -1))
    if label not in (0, 1):
        raise ValueError("target-prior stress requires binary labels")
    return target_id, label


def _bounded_probability(value: float, epsilon: float = 1e-6) -> float:
    return min(max(float(value), epsilon), 1.0 - epsilon)


def _logit(value: float) -> float:
    return log(value / (1.0 - value))

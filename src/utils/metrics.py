"""Metric utilities for pseudo-confidence and uncertainty evaluation.

The helpers intentionally use only the Python standard library so evaluation
can run on lightweight environments before the full experiment stack lands.
"""

from __future__ import annotations

from math import ceil, sqrt
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


Number = Optional[float]


def auroc(labels: Sequence[int], scores: Sequence[float]) -> Number:
    """Compute binary AUROC with average ranks for tied scores."""
    pairs = _valid_pairs(labels, scores)
    positives = sum(label for label, _score in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return None

    sorted_pairs = sorted(enumerate(pairs), key=lambda item: item[1][1])
    rank_sum_positive = 0.0
    index = 0
    while index < len(sorted_pairs):
        end = index + 1
        score = sorted_pairs[index][1][1]
        while end < len(sorted_pairs) and sorted_pairs[end][1][1] == score:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for tied_index in range(index, end):
            if sorted_pairs[tied_index][1][0] == 1:
                rank_sum_positive += average_rank
        index = end

    return (rank_sum_positive - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def auprc(labels: Sequence[int], scores: Sequence[float]) -> Number:
    """Compute binary average precision / AUPRC with threshold-level ties.

    All examples sharing a score are incorporated before precision and recall
    are evaluated.  This makes the result invariant to row order inside a tie
    and matches the average-precision convention used by report_metrics.py.
    """
    pairs = _valid_pairs(labels, scores)
    positives = sum(label for label, _score in pairs)
    if positives == 0:
        return None

    ordered = sorted(pairs, key=lambda item: item[1], reverse=True)
    true_positive = 0
    false_positive = 0
    previous_recall = 0.0
    area = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        score = ordered[index][1]
        while end < len(ordered) and ordered[end][1] == score:
            end += 1
        for label, _score in ordered[index:end]:
            if label == 1:
                true_positive += 1
            else:
                false_positive += 1
        recall = true_positive / positives
        precision = true_positive / max(true_positive + false_positive, 1)
        area += precision * (recall - previous_recall)
        previous_recall = recall
        index = end
    return area


def mcc(
    labels: Sequence[int], scores: Sequence[float], threshold: float = 0.5
) -> Number:
    """Compute Matthews correlation coefficient for binary scores."""
    counts = confusion_counts(labels, scores, threshold=threshold)
    if counts["total"] == 0:
        return None
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    denominator = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return 0.0
    return (tp * tn - fp * fn) / denominator


def f1_score(
    labels: Sequence[int], scores: Sequence[float], threshold: float = 0.5
) -> Number:
    """Compute binary F1 for scores thresholded into predictions."""
    counts = confusion_counts(labels, scores, threshold=threshold)
    if counts["total"] == 0:
        return None
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    denominator = 2 * tp + fp + fn
    if denominator == 0:
        return 0.0
    return (2 * tp) / denominator


def ece(
    labels: Sequence[int],
    scores: Sequence[float],
    n_bins: int = 10,
    threshold: float = 0.5,
) -> Number:
    """Compute expected calibration error from binary confidence values."""
    pairs = _valid_pairs(labels, scores)
    if not pairs:
        return None
    n_bins = max(1, int(n_bins))
    confidences = [_confidence(score, threshold=threshold) for _label, score in pairs]
    predictions = [1 if score >= threshold else 0 for _label, score in pairs]

    total = len(pairs)
    error = 0.0
    for bin_index in range(n_bins):
        lower = bin_index / n_bins
        upper = (bin_index + 1) / n_bins
        members = []
        for row_index, confidence in enumerate(confidences):
            if bin_index == n_bins - 1:
                in_bin = lower <= confidence <= upper
            else:
                in_bin = lower <= confidence < upper
            if in_bin:
                members.append(row_index)
        if not members:
            continue
        accuracy = sum(
            1
            for row_index in members
            if predictions[row_index] == pairs[row_index][0]
        ) / len(members)
        mean_confidence = sum(confidences[row_index] for row_index in members) / len(
            members
        )
        error += (len(members) / total) * abs(accuracy - mean_confidence)
    return error


def brier_score(labels: Sequence[int], scores: Sequence[float]) -> Number:
    """Compute the binary Brier score."""
    pairs = _valid_pairs(labels, scores)
    if not pairs:
        return None
    return sum((score - label) ** 2 for label, score in pairs) / len(pairs)


def ofr(
    labels: Sequence[int],
    scores: Sequence[float],
    confidence_threshold: float = 0.9,
    threshold: float = 0.5,
) -> Number:
    """Compute overconfidence failure rate.

    OFR is defined here as high-confidence errors divided by all errors.
    """
    counts = confidence_error_counts(
        labels,
        scores,
        confidence_threshold=confidence_threshold,
        threshold=threshold,
    )
    if counts["errors"] == 0:
        return 0.0
    return counts["high_confidence_errors"] / counts["errors"]


def confidence_error_counts(
    labels: Sequence[int],
    scores: Sequence[float],
    confidence_threshold: float = 0.9,
    threshold: float = 0.5,
) -> Dict[str, int]:
    """Count high-confidence errors and related totals."""
    pairs = _valid_pairs(labels, scores)
    high_confidence_errors = 0
    high_confidence_total = 0
    errors = 0
    for label, score in pairs:
        prediction = 1 if score >= threshold else 0
        confidence = _confidence(score, threshold=threshold)
        is_error = prediction != label
        if is_error:
            errors += 1
        if confidence >= confidence_threshold:
            high_confidence_total += 1
            if is_error:
                high_confidence_errors += 1
    return {
        "total": len(pairs),
        "errors": errors,
        "high_confidence_total": high_confidence_total,
        "high_confidence_errors": high_confidence_errors,
    }


def confusion_counts(
    labels: Sequence[int], scores: Sequence[float], threshold: float = 0.5
) -> Dict[str, int]:
    """Return binary confusion-matrix counts."""
    pairs = _valid_pairs(labels, scores)
    tp = tn = fp = fn = 0
    for label, score in pairs:
        prediction = 1 if score >= threshold else 0
        if prediction == 1 and label == 1:
            tp += 1
        elif prediction == 0 and label == 0:
            tn += 1
        elif prediction == 1 and label == 0:
            fp += 1
        else:
            fn += 1
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "total": len(pairs)}


def risk_coverage_curve(
    labels: Sequence[int],
    scores: Sequence[float],
    uncertainties: Optional[Sequence[float]] = None,
    coverages: Optional[Iterable[float]] = None,
    threshold: float = 0.5,
) -> List[Dict[str, Number]]:
    """Compute selective-prediction risk at requested coverage levels."""
    if uncertainties is not None:
        uncertainty_values = _as_optional_float_list(uncertainties)
    else:
        uncertainty_values = [None] * len(scores)

    aligned_rows = []
    for row_index, score in enumerate(scores):
        score_value = _to_float(score)
        if score_value is None:
            continue
        label_value = _to_label(labels[row_index])
        if label_value is None:
            continue
        uncertainty = uncertainty_values[row_index]
        confidence = _confidence(score_value, threshold=threshold)
        rank_value = -confidence if uncertainty is None else uncertainty
        aligned_rows.append((rank_value, label_value, score_value))

    if not aligned_rows:
        return []

    ordered = sorted(aligned_rows, key=lambda item: item[0])
    levels = list(coverages) if coverages is not None else [i / 10 for i in range(1, 11)]
    curve = []
    for coverage in levels:
        clipped = min(1.0, max(0.0, float(coverage)))
        k = max(1, int(ceil(len(ordered) * clipped)))
        selected = ordered[:k]
        errors = sum(
            1
            for _rank, label, score in selected
            if (1 if score >= threshold else 0) != label
        )
        curve.append({"coverage": k / len(ordered), "risk": errors / k, "count": k})
    return curve


def top_k_ranking(
    labels: Sequence[int],
    scores: Sequence[float],
    top_ks: Sequence[int],
) -> List[Dict[str, Number]]:
    """Compute positive-label ranking metrics at each K."""
    pairs = _valid_pairs(labels, scores)
    if not pairs:
        return []
    ordered = sorted(pairs, key=lambda item: item[1], reverse=True)
    positives = sum(label for label, _score in pairs)
    rows = []
    for top_k in top_ks:
        k = max(1, min(int(top_k), len(ordered)))
        selected = ordered[:k]
        hits = sum(label for label, _score in selected)
        rows.append(
            {
                "k": k,
                "hits": hits,
                "precision_at_k": hits / k,
                "recall_at_k": None if positives == 0 else hits / positives,
            }
        )
    return rows


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> Number:
    """Compute Pearson correlation for paired scalar values."""
    pairs = [
        (x_value, y_value)
        for x_value, y_value in zip(
            _as_optional_float_list(xs), _as_optional_float_list(ys)
        )
        if x_value is not None and y_value is not None
    ]
    if len(pairs) < 2:
        return None
    x_mean = sum(x for x, _y in pairs) / len(pairs)
    y_mean = sum(y for _x, y in pairs) / len(pairs)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    x_denominator = sqrt(sum((x - x_mean) ** 2 for x, _y in pairs))
    y_denominator = sqrt(sum((y - y_mean) ** 2 for _x, y in pairs))
    denominator = x_denominator * y_denominator
    if denominator == 0:
        return None
    return numerator / denominator


def spearman_correlation(xs: Sequence[float], ys: Sequence[float]) -> Number:
    """Compute Spearman rank correlation with average ranks for ties."""
    pairs = [
        (x_value, y_value)
        for x_value, y_value in zip(
            _as_optional_float_list(xs), _as_optional_float_list(ys)
        )
        if x_value is not None and y_value is not None
    ]
    if len(pairs) < 2:
        return None
    x_ranks = _average_ranks([x for x, _y in pairs])
    y_ranks = _average_ranks([y for _x, y in pairs])
    return pearson_correlation(x_ranks, y_ranks)


def branch_classification_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float = 0.5,
    ece_bins: int = 10,
    confidence_threshold: float = 0.9,
) -> Dict[str, Number]:
    """Return the standard binary metrics requested for a branch."""
    counts = confidence_error_counts(
        labels,
        scores,
        confidence_threshold=confidence_threshold,
        threshold=threshold,
    )
    return {
        "auroc": auroc(labels, scores),
        "auprc": auprc(labels, scores),
        "mcc": mcc(labels, scores, threshold=threshold),
        "f1": f1_score(labels, scores, threshold=threshold),
        "ece": ece(labels, scores, n_bins=ece_bins, threshold=threshold),
        "brier": brier_score(labels, scores),
        "ofr": ofr(
            labels,
            scores,
            confidence_threshold=confidence_threshold,
            threshold=threshold,
        ),
        "high_confidence_errors": counts["high_confidence_errors"],
        "high_confidence_total": counts["high_confidence_total"],
        "errors": counts["errors"],
        "total": counts["total"],
    }


def _valid_pairs(
    labels: Sequence[int], scores: Sequence[float]
) -> List[Tuple[int, float]]:
    """Return valid binary label and score pairs."""
    pairs = []
    for label, score in zip(labels, scores):
        label_value = _to_label(label)
        score_value = _to_float(score)
        if label_value is None or score_value is None:
            continue
        pairs.append((label_value, score_value))
    return pairs


def _to_label(value: object) -> Optional[int]:
    """Convert a raw value into a binary label."""
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return 1 if numeric >= 0.5 else 0


def _to_float(value: object) -> Optional[float]:
    """Convert a raw value into a finite float."""
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return numeric


def _as_optional_float_list(values: Sequence[float]) -> List[Optional[float]]:
    """Convert a sequence into optional floats."""
    return [_to_float(value) for value in values]


def _confidence(score: float, threshold: float = 0.5) -> float:
    """Return confidence of the thresholded binary prediction."""
    return score if score >= threshold else 1.0 - score


def _average_ranks(values: Sequence[float]) -> List[float]:
    """Return average ranks for a list of values."""
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        value = indexed[index][1]
        while end < len(indexed) and indexed[end][1] == value:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for tied_index in range(index, end):
            ranks[indexed[tied_index][0]] = average_rank
        index = end
    return ranks

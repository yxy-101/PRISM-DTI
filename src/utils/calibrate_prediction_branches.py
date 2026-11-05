#!/usr/bin/env python
"""Fit validation-only temperature scaling and apply it unchanged to test."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.metrics import auprc, auroc, brier_score, ece


BRANCH_COLUMNS = {"raw": "p_raw", "do": "p_do"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit one scalar temperature on validation labels and apply it "
            "unchanged to aligned test predictions. Original probabilities "
            "and PCS/PPCE columns are never overwritten."
        )
    )
    parser.add_argument("--valid-pred", required=True)
    parser.add_argument("--test-pred", required=True)
    parser.add_argument(
        "--branch",
        action="append",
        choices=sorted(BRANCH_COLUMNS),
        dest="branches",
        help="Branch to calibrate; repeat for both. Defaults to every available branch.",
    )
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--minimum-temperature", type=float, default=0.05)
    parser.add_argument("--maximum-temperature", type=float, default=20.0)
    parser.add_argument("--grid-points", type=int, default=401)
    parser.add_argument("--refinement-rounds", type=int, default=3)
    parser.add_argument(
        "--minimum-valid-nll-improvement", type=float, default=1e-6
    )
    return parser.parse_args()


def resolve(path: str) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = ROOT / value
    return value.resolve()


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError("Prediction file not found: %s" % path)
    frame = pd.read_csv(path)
    required = {"drug_id", "target_id", "label"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("%s is missing columns: %s" % (path, missing))
    frame = frame.copy()
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if not frame["label"].isin([0, 1]).all():
        raise ValueError("Labels must be binary in %s" % path)
    return frame


def validate_alignment(valid: pd.DataFrame, test: pd.DataFrame) -> None:
    for name, frame in (("valid", valid), ("test", test)):
        duplicated = frame.duplicated(["drug_id", "target_id"]).sum()
        if duplicated:
            raise ValueError(
                "%s predictions contain %d duplicate drug-target pairs"
                % (name, duplicated)
            )


def clipped_probabilities(values: Iterable[float]) -> np.ndarray:
    scores = np.asarray(list(values), dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Prediction probabilities contain non-finite values")
    if ((scores < 0.0) | (scores > 1.0)).any():
        raise ValueError("Prediction probabilities must be in [0, 1]")
    return np.clip(scores, 1e-7, 1.0 - 1e-7)


def logits_from_probabilities(scores: np.ndarray) -> np.ndarray:
    return np.log(scores) - np.log1p(-scores)


def sigmoid(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=float)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def apply_temperature(scores: np.ndarray, temperature: float) -> np.ndarray:
    logits = logits_from_probabilities(scores)
    return sigmoid(logits / max(float(temperature), 1e-8))


def binary_nll(labels: np.ndarray, scores: np.ndarray) -> float:
    clipped = np.clip(scores, 1e-7, 1.0 - 1e-7)
    return float(
        -np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))
    )


def fit_temperature(
    labels: np.ndarray,
    scores: np.ndarray,
    minimum: float,
    maximum: float,
    grid_points: int,
    refinement_rounds: int,
) -> Tuple[float, float]:
    if not 0.0 < minimum <= 1.0 <= maximum:
        raise ValueError(
            "Temperature interval must satisfy 0 < minimum <= 1 <= maximum"
        )
    points = max(21, int(grid_points))
    low = math.log(minimum)
    high = math.log(maximum)
    best_log = 0.0
    best_nll = binary_nll(labels, scores)
    for _ in range(max(1, int(refinement_rounds))):
        grid = np.linspace(low, high, points)
        losses = np.asarray(
            [
                binary_nll(labels, apply_temperature(scores, math.exp(value)))
                for value in grid
            ],
            dtype=float,
        )
        best_index = int(np.argmin(losses))
        best_log = float(grid[best_index])
        best_nll = float(losses[best_index])
        step = (high - low) / max(points - 1, 1)
        low = max(math.log(minimum), best_log - 3.0 * step)
        high = min(math.log(maximum), best_log + 3.0 * step)
    return float(math.exp(best_log)), best_nll


def accuracy(labels: Sequence[int], scores: Sequence[float]) -> float:
    labels_array = np.asarray(labels, dtype=int)
    scores_array = np.asarray(scores, dtype=float)
    return float(np.mean((scores_array >= 0.5).astype(int) == labels_array))


def summarize(
    branch: str,
    split: str,
    state: str,
    labels: np.ndarray,
    scores: np.ndarray,
    temperature: float,
    used_identity_fallback: bool,
    ece_bins: int,
) -> Dict[str, object]:
    return {
        "branch": branch,
        "split": split,
        "calibration_state": state,
        "sample_count": int(len(labels)),
        "positive_count": int(labels.sum()),
        "temperature": float(temperature),
        "used_identity_fallback": bool(used_identity_fallback),
        "nll": binary_nll(labels, scores),
        "roc_auc": auroc(labels.tolist(), scores.tolist()),
        "aupr": auprc(labels.tolist(), scores.tolist()),
        "accuracy": accuracy(labels, scores),
        "brier": brier_score(labels.tolist(), scores.tolist()),
        "ece": ece(labels.tolist(), scores.tolist(), n_bins=ece_bins),
    }


def write_report(
    path: Path,
    summary: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Validation-Locked Prediction Calibration",
        "",
        "## Protocol",
        "",
        "- One scalar temperature is selected by validation negative log-likelihood.",
        "- The selected temperature is applied unchanged to test probabilities.",
        "- Original p_raw, p_do, PCS, and PPCE values are not overwritten.",
        "- Calibration cannot improve ROC-AUC or AUPR ranking.",
        "",
        "## Results",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Interpretation Boundary",
        "",
        "Use calibrated columns for calibration and decision-threshold analyses only. ",
        "Keep uncalibrated predictions for the primary PCS/PPCE mechanism analysis.",
        "",
        "## Reproducibility",
        "",
        "- ECE bins: `%d`." % args.ece_bins,
        "- Temperature range: `[%s, %s]`."
        % (args.minimum_temperature, args.maximum_temperature),
        "- Minimum validation NLL improvement: `%s`."
        % args.minimum_valid_nll_improvement,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    valid_path = resolve(args.valid_pred)
    test_path = resolve(args.test_pred)
    valid = load_predictions(valid_path)
    test = load_predictions(test_path)
    validate_alignment(valid, test)

    requested = args.branches or [
        branch
        for branch, column in BRANCH_COLUMNS.items()
        if column in valid.columns and column in test.columns
    ]
    if not requested:
        raise ValueError("No calibratable p_raw or p_do columns were found")

    rows: List[Dict[str, object]] = []
    decisions: List[Dict[str, object]] = []
    valid_output = valid.copy()
    test_output = test.copy()
    valid_labels = valid["label"].to_numpy(dtype=int)
    test_labels = test["label"].to_numpy(dtype=int)

    for branch in requested:
        column = BRANCH_COLUMNS[branch]
        if column not in valid.columns or column not in test.columns:
            raise ValueError("Requested branch %s requires column %s" % (branch, column))
        valid_scores = clipped_probabilities(valid[column])
        test_scores = clipped_probabilities(test[column])
        baseline_nll = binary_nll(valid_labels, valid_scores)
        temperature, fitted_nll = fit_temperature(
            labels=valid_labels,
            scores=valid_scores,
            minimum=args.minimum_temperature,
            maximum=args.maximum_temperature,
            grid_points=args.grid_points,
            refinement_rounds=args.refinement_rounds,
        )
        improvement = baseline_nll - fitted_nll
        identity_fallback = improvement < args.minimum_valid_nll_improvement
        if identity_fallback:
            temperature = 1.0
        valid_calibrated = apply_temperature(valid_scores, temperature)
        test_calibrated = apply_temperature(test_scores, temperature)
        calibrated_column = "%s_calibrated" % column
        valid_output[calibrated_column] = valid_calibrated
        test_output[calibrated_column] = test_calibrated

        for split, labels, original, calibrated in (
            ("valid", valid_labels, valid_scores, valid_calibrated),
            ("test", test_labels, test_scores, test_calibrated),
        ):
            rows.append(
                summarize(
                    branch,
                    split,
                    "uncalibrated",
                    labels,
                    original,
                    temperature,
                    identity_fallback,
                    args.ece_bins,
                )
            )
            rows.append(
                summarize(
                    branch,
                    split,
                    "calibrated",
                    labels,
                    calibrated,
                    temperature,
                    identity_fallback,
                    args.ece_bins,
                )
            )
        decisions.append(
            {
                "branch": branch,
                "temperature": temperature,
                "valid_nll_before": baseline_nll,
                "valid_nll_after": binary_nll(valid_labels, valid_calibrated),
                "valid_nll_improvement": baseline_nll
                - binary_nll(valid_labels, valid_calibrated),
                "used_identity_fallback": identity_fallback,
                "selection_split": "valid",
            }
        )

    output_base = resolve("outputs/metrics/%s" % args.output_prefix)
    summary_path = output_base.with_name(output_base.name + "_summary.csv")
    selection_path = output_base.with_name(output_base.name + "_selection.csv")
    valid_output_path = resolve(
        "outputs/predictions/%s_valid_predictions.csv" % args.output_prefix
    )
    test_output_path = resolve(
        "outputs/predictions/%s_test_predictions.csv" % args.output_prefix
    )
    report_path = resolve("docs/%s_report.md" % args.output_prefix)
    for path in (
        summary_path,
        selection_path,
        valid_output_path,
        test_output_path,
        report_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(rows)
    summary["ece_bins"] = int(args.ece_bins)
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(decisions).to_csv(selection_path, index=False)
    valid_output.to_csv(valid_output_path, index=False)
    test_output.to_csv(test_output_path, index=False)
    write_report(report_path, summary, args)

    print(pd.DataFrame(decisions).to_string(index=False))
    print(summary.to_string(index=False))
    print("Saved:")
    for path in (
        summary_path,
        selection_path,
        valid_output_path,
        test_output_path,
        report_path,
    ):
        print(path)


if __name__ == "__main__":
    main()

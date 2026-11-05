#!/usr/bin/env python
"""Audit target-prior effects on conditional PPCE severity.

The analysis is read-only with respect to checkpoints and predictions. It
restricts the sample domain to high-confidence false-positive candidates,
uses positive PCS as the pre-specified primary outcome, and treats binary PPCE
rate as a secondary outcome. Inference uses target-level bootstrap confidence
intervals and target-level permutation tests, avoiding a statsmodels
dependency and sample-level pseudo-replication.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREFIX = "bindingdb_highauc_conditional_ppce_severity"
REQUIRED_TRAIN_COLUMNS = ("drug_id", "target_id", "label")
REQUIRED_PREDICTION_COLUMNS = (
    "drug_id",
    "target_id",
    "label",
    "p_raw",
    "p_do",
    "pcs",
)
CONTROL_COLUMNS = (
    "mean_p_raw",
    "mean_drug_prior_eb",
    "log_target_count",
    "mean_log_drug_count",
)
OUTCOMES = (
    ("mean_positive_pcs", "primary_continuous_severity"),
    ("mean_pcs", "continuous_sensitivity"),
    ("ppce_rate", "secondary_binary_rate"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether train-derived target prior has an independent "
            "positive association with PPCE severity inside high-confidence "
            "false-positive candidates."
        )
    )
    parser.add_argument(
        "--train-csv",
        default="data/processed/bindingdb_train.csv",
    )
    parser.add_argument(
        "--valid-pred",
        default=(
            "outputs/predictions/"
            "stage5_bindingdb_realprotein_real2d_real3d_valid_predictions.csv"
        ),
    )
    parser.add_argument(
        "--test-pred",
        default=(
            "outputs/predictions/"
            "stage5_bindingdb_realprotein_real2d_real3d_test_predictions.csv"
        ),
    )
    parser.add_argument("--dataset", default="bindingdb")
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--prior-strength", type=float, default=10.0)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--permutation-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20240618)
    parser.add_argument("--minimum-targets", type=int, default=20)
    parser.add_argument("--minimum-raw-auc", type=float, default=0.0)
    parser.add_argument("--minimum-raw-aupr", type=float, default=0.0)
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else ROOT / path


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_csv(path: Path, required: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError("Input file not found: %s" % path)
    frame = pd.read_csv(path)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("%s is missing columns: %s" % (path, missing))
    return frame


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binary_ece(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    predictions = (probabilities >= 0.5).astype(int)
    confidence = np.where(predictions == 1, probabilities, 1.0 - probabilities)
    correctness = (predictions == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        if index == bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if mask.any():
            value += float(mask.mean()) * abs(
                float(correctness[mask].mean()) - float(confidence[mask].mean())
            )
    return float(value)


def prediction_metrics(frame: pd.DataFrame) -> Dict[str, float]:
    labels = frame["label"].to_numpy(dtype=int)
    probabilities = frame["p_raw"].to_numpy(dtype=float)
    return {
        "raw_auc": float(roc_auc_score(labels, probabilities)),
        "raw_aupr": float(average_precision_score(labels, probabilities)),
        "raw_accuracy": float(((probabilities >= 0.5).astype(int) == labels).mean()),
        "raw_brier": float(np.mean((probabilities - labels) ** 2)),
        "raw_ece": binary_ece(labels, probabilities),
    }


def build_train_covariates(
    train: pd.DataFrame,
    prior_strength: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    working = train.copy()
    working["target_id"] = working["target_id"].astype(str)
    working["drug_id"] = working["drug_id"].astype(str)
    working["label"] = pd.to_numeric(working["label"], errors="raise").astype(int)
    global_rate = float(working["label"].mean())

    target = (
        working.groupby("target_id", as_index=False)["label"]
        .agg(["sum", "count"])
        .reset_index()
    )
    target["target_prior_eb"] = (
        target["sum"] + prior_strength * global_rate
    ) / (target["count"] + prior_strength)
    target["log_target_count"] = np.log1p(target["count"])

    count_model = LinearRegression()
    count_model.fit(
        target[["log_target_count"]].to_numpy(dtype=float),
        target["target_prior_eb"].to_numpy(dtype=float),
    )
    target["prior_residual"] = target["target_prior_eb"] - count_model.predict(
        target[["log_target_count"]].to_numpy(dtype=float)
    )

    drug = (
        working.groupby("drug_id", as_index=False)["label"]
        .agg(["sum", "count"])
        .reset_index()
    )
    drug["drug_prior_eb"] = (
        drug["sum"] + prior_strength * global_rate
    ) / (drug["count"] + prior_strength)
    drug["log_drug_count"] = np.log1p(drug["count"])

    return target, drug, global_rate


def prepare_target_detail(
    prediction: pd.DataFrame,
    target_covariates: pd.DataFrame,
    drug_covariates: pd.DataFrame,
    global_rate: float,
    confidence_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frame = prediction.copy()
    frame["target_id"] = frame["target_id"].astype(str)
    frame["drug_id"] = frame["drug_id"].astype(str)
    for column in ("label", "p_raw", "p_do", "pcs"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    frame = frame.merge(
        target_covariates[
            [
                "target_id",
                "target_prior_eb",
                "prior_residual",
                "log_target_count",
            ]
        ],
        on="target_id",
        how="left",
        validate="many_to_one",
    )
    frame = frame.merge(
        drug_covariates[["drug_id", "drug_prior_eb", "log_drug_count"]],
        on="drug_id",
        how="left",
        validate="many_to_one",
    )
    frame["target_prior_eb"] = frame["target_prior_eb"].fillna(global_rate)
    frame["prior_residual"] = frame["prior_residual"].fillna(0.0)
    frame["log_target_count"] = frame["log_target_count"].fillna(0.0)
    frame["drug_prior_eb"] = frame["drug_prior_eb"].fillna(global_rate)
    frame["log_drug_count"] = frame["log_drug_count"].fillna(0.0)

    candidates = frame[
        (frame["label"] == 0) & (frame["p_raw"] >= confidence_threshold)
    ].copy()
    candidates["is_ppce"] = (candidates["pcs"] > 0.0).astype(int)
    candidates["positive_pcs"] = np.maximum(candidates["pcs"], 0.0)

    target_detail = (
        candidates.groupby("target_id", as_index=False)
        .agg(
            target_prior_eb=("target_prior_eb", "first"),
            prior_residual=("prior_residual", "first"),
            log_target_count=("log_target_count", "first"),
            candidate_count=("is_ppce", "size"),
            ppce_count=("is_ppce", "sum"),
            ppce_rate=("is_ppce", "mean"),
            mean_positive_pcs=("positive_pcs", "mean"),
            mean_pcs=("pcs", "mean"),
            mean_p_raw=("p_raw", "mean"),
            mean_drug_prior_eb=("drug_prior_eb", "mean"),
            mean_log_drug_count=("log_drug_count", "mean"),
        )
        .reset_index(drop=True)
    )
    return candidates, target_detail


def fit_standardized_beta(frame: pd.DataFrame, outcome: str) -> Tuple[float, float]:
    features = ["prior_residual"] + list(CONTROL_COLUMNS)
    matrix = frame[features].to_numpy(dtype=float)
    target = frame[outcome].to_numpy(dtype=float)
    if len(frame) < 3 or np.std(target) <= 0 or np.std(matrix[:, 0]) <= 0:
        return math.nan, math.nan
    scaled = StandardScaler().fit_transform(matrix)
    model = LinearRegression().fit(scaled, target)
    return float(model.coef_[0]), float(model.score(scaled, target))


def safe_spearman(left: pd.Series, right: pd.Series) -> Tuple[float, float]:
    left_values = left.to_numpy(dtype=float)
    right_values = right.to_numpy(dtype=float)
    if (
        len(left_values) < 3
        or np.std(left_values) <= 0
        or np.std(right_values) <= 0
    ):
        return math.nan, math.nan
    correlation, p_value = spearmanr(left_values, right_values)
    return float(correlation), float(p_value)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = ["| " + " | ".join(columns) + " |"]
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for record in frame.itertuples(index=False, name=None):
        values = []
        for value in record:
            if isinstance(value, float):
                values.append("%.6g" % value if np.isfinite(value) else "NaN")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def bootstrap_beta(
    frame: pd.DataFrame,
    outcome: str,
    iterations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    values: List[float] = []
    row_count = len(frame)
    for _ in range(iterations):
        indices = rng.integers(0, row_count, size=row_count)
        sample = frame.iloc[indices]
        beta, _ = fit_standardized_beta(sample, outcome)
        if np.isfinite(beta):
            values.append(beta)
    return np.asarray(values, dtype=float)


def permutation_beta(
    frame: pd.DataFrame,
    outcome: str,
    iterations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    values: List[float] = []
    for _ in range(iterations):
        permuted = frame.copy()
        permuted["prior_residual"] = rng.permutation(
            permuted["prior_residual"].to_numpy(dtype=float)
        )
        beta, _ = fit_standardized_beta(permuted, outcome)
        if np.isfinite(beta):
            values.append(beta)
    return np.asarray(values, dtype=float)


def analyze_outcome(
    split: str,
    outcome: str,
    role: str,
    target_detail: pd.DataFrame,
    candidates: pd.DataFrame,
    metrics: Dict[str, float],
    prediction_path: Path,
    args: argparse.Namespace,
    seed_offset: int,
) -> Dict[str, object]:
    rng = np.random.default_rng(args.seed + seed_offset)
    beta, model_r2 = fit_standardized_beta(target_detail, outcome)
    boot = bootstrap_beta(
        target_detail,
        outcome,
        args.bootstrap_iterations,
        rng,
    )
    perm = permutation_beta(
        target_detail,
        outcome,
        args.permutation_iterations,
        rng,
    )
    if len(boot):
        ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    else:
        ci_low, ci_high = math.nan, math.nan
    if len(perm) and np.isfinite(beta):
        permutation_p = float((1 + np.sum(perm >= beta)) / (len(perm) + 1))
    else:
        permutation_p = math.nan

    raw_corr, raw_corr_p = safe_spearman(
        target_detail["target_prior_eb"],
        target_detail[outcome],
    )
    supported = bool(
        np.isfinite(beta)
        and beta > 0
        and np.isfinite(ci_low)
        and ci_low > 0
        and np.isfinite(permutation_p)
        and permutation_p < 0.05
    )
    performance_floor_met = bool(
        metrics["raw_auc"] >= args.minimum_raw_auc
        and metrics["raw_aupr"] >= args.minimum_raw_aupr
    )
    return {
        "dataset": args.dataset,
        "split": split,
        "outcome": outcome,
        "outcome_role": role,
        "confidence_threshold": args.confidence_threshold,
        "candidate_count": int(len(candidates)),
        "candidate_target_count": int(len(target_detail)),
        "ppce_count": int(candidates["is_ppce"].sum()),
        "candidate_ppce_rate": float(candidates["is_ppce"].mean())
        if len(candidates)
        else math.nan,
        "target_prior_standardized_beta": beta,
        "bootstrap_beta_ci_low": float(ci_low),
        "bootstrap_beta_ci_high": float(ci_high),
        "permutation_p_one_sided": permutation_p,
        "target_level_spearman": raw_corr,
        "target_level_spearman_p": raw_corr_p,
        "model_r2": model_r2,
        "bootstrap_valid_iterations": int(len(boot)),
        "permutation_valid_iterations": int(len(perm)),
        "positive_independent_effect": supported,
        "raw_auc": metrics["raw_auc"],
        "raw_aupr": metrics["raw_aupr"],
        "raw_accuracy": metrics["raw_accuracy"],
        "raw_brier": metrics["raw_brier"],
        "raw_ece": metrics["raw_ece"],
        "performance_floor_met": performance_floor_met,
        "prediction_file": relative_path(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
    }


def write_report(
    summary: pd.DataFrame,
    detail_path: Path,
    summary_path: Path,
    report_path: Path,
    args: argparse.Namespace,
) -> None:
    primary = summary[summary["outcome_role"] == "primary_continuous_severity"]
    replicated = bool(
        len(primary) == 2
        and primary["positive_independent_effect"].all()
        and primary["performance_floor_met"].all()
    )
    lines = [
        "# Conditional PPCE Severity Audit",
        "",
        "This analysis freezes the input prediction CSVs and does not modify "
        "the checkpoint, p_raw, p_do, PCS, PPCE, or HCE definitions.",
        "",
        "## Pre-specified analysis",
        "",
        "- Candidate domain: label=0 and p_raw >= %.3f." % args.confidence_threshold,
        "- Primary outcome: target-level mean positive PCS.",
        "- Secondary outcomes: target-level mean PCS and PPCE rate.",
        "- Exposure: empirical-Bayes target prior residualized against target count.",
        "- Controls: raw score, drug prior, target count, and drug count.",
        "- Inference: target-level bootstrap CI and one-sided target permutation.",
        "",
        "## Results",
        "",
        markdown_table(summary),
        "",
        "## Replication decision",
        "",
        "- Primary positive independent effect replicated in valid and test: "
        "**%s**." % replicated,
        "- A positive claim requires beta > 0, bootstrap CI lower bound > 0, "
        "permutation p < 0.05, and the configured raw-performance floor.",
        "- Failure to meet these conditions is evidence insufficiency, not "
        "permission to tune the test split.",
        "",
        "## Outputs",
        "",
        "- `%s`" % relative_path(summary_path),
        "- `%s`" % relative_path(detail_path),
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.bootstrap_iterations < 100:
        raise ValueError("--bootstrap-iterations must be at least 100")
    if args.permutation_iterations < 100:
        raise ValueError("--permutation-iterations must be at least 100")
    if not 0.5 < args.confidence_threshold < 1.0:
        raise ValueError("--confidence-threshold must be between 0.5 and 1.0")

    train_path = resolve_path(args.train_csv)
    prediction_paths = {
        "valid": resolve_path(args.valid_pred),
        "test": resolve_path(args.test_pred),
    }
    train = load_csv(train_path, REQUIRED_TRAIN_COLUMNS)
    target_covariates, drug_covariates, global_rate = build_train_covariates(
        train,
        args.prior_strength,
    )

    summary_rows: List[Dict[str, object]] = []
    detail_frames: List[pd.DataFrame] = []
    for split_index, split in enumerate(("valid", "test")):
        prediction_path = prediction_paths[split]
        prediction = load_csv(prediction_path, REQUIRED_PREDICTION_COLUMNS)
        metrics = prediction_metrics(prediction)
        candidates, target_detail = prepare_target_detail(
            prediction,
            target_covariates,
            drug_covariates,
            global_rate,
            args.confidence_threshold,
        )
        if len(target_detail) < args.minimum_targets:
            raise ValueError(
                "%s has only %d candidate targets; minimum is %d"
                % (split, len(target_detail), args.minimum_targets)
            )
        target_detail.insert(0, "split", split)
        detail_frames.append(target_detail)
        for outcome_index, (outcome, role) in enumerate(OUTCOMES):
            summary_rows.append(
                analyze_outcome(
                    split,
                    outcome,
                    role,
                    target_detail,
                    candidates,
                    metrics,
                    prediction_path,
                    args,
                    seed_offset=split_index * 100 + outcome_index,
                )
            )

    summary = pd.DataFrame(summary_rows)
    primary = summary[summary["outcome_role"] == "primary_continuous_severity"]
    replicated = bool(
        len(primary) == 2
        and primary["positive_independent_effect"].all()
        and primary["performance_floor_met"].all()
    )
    summary["primary_effect_replicated_valid_test"] = replicated

    output_dir = ROOT / "outputs" / "metrics"
    report_dir = ROOT / "docs"
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix
    summary_path = output_dir / ("%s_summary.csv" % prefix)
    detail_path = output_dir / ("%s_target_detail.csv" % prefix)
    report_path = report_dir / ("%s_report.md" % prefix)
    summary.to_csv(summary_path, index=False)
    pd.concat(detail_frames, ignore_index=True).to_csv(detail_path, index=False)
    write_report(summary, detail_path, summary_path, report_path, args)

    print(summary.to_string(index=False))
    print("\nPrimary effect replicated valid/test:", replicated)
    print("Saved:")
    print(summary_path)
    print(detail_path)
    print(report_path)


if __name__ == "__main__":
    main()

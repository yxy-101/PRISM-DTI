#!/usr/bin/env python
"""Audit the independent target-prior association with PPCE incidence.

The primary risk set contains every negative-labeled prediction, avoiding the
selection bias and outcome saturation caused by conditioning first on
high-confidence false positives. The primary endpoint is the grouped-binomial
PPCE incidence per target. Target-level bootstrap confidence intervals and
target-level regressor-residual permutations provide exploratory inference.

This remains an observational association audit. A positive replicated result
supports an independent positive association, not causal identification.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREFIX = "bindingdb_highauc_target_prior_ppce_independent_risk"
REQUIRED_TRAIN_COLUMNS = ("drug_id", "target_id", "label")
REQUIRED_PREDICTION_COLUMNS = (
    "drug_id",
    "target_id",
    "label",
    "p_raw",
    "p_do",
    "pcs",
)
EXPOSURE = "target_prior_eb"
CONTROL_COLUMNS = (
    "log_train_target_count",
    "mean_drug_prior_eb",
    "mean_log_train_drug_count",
    "log_eval_negative_count",
)
OUTCOMES = (
    ("hcfp_count", "grouped_binomial", "primary_hcfp_incidence"),
    ("ppce_count", "grouped_binomial", "primary_ppce_incidence"),
    (
        "mean_positive_pcs_all_negatives",
        "weighted_linear",
        "secondary_ppce_severity",
    ),
    ("mean_pcs_all_negatives", "weighted_linear", "sensitivity_pcs_shift"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether train-derived target prior is independently and "
            "positively associated with PPCE incidence among all negatives."
        )
    )
    parser.add_argument("--dataset", default="bindingdb")
    parser.add_argument("--train-csv", default="data/processed/bindingdb_train.csv")
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
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--prior-strength", type=float, default=10.0)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--permutation-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20240619)
    parser.add_argument("--minimum-targets", type=int, default=20)
    parser.add_argument("--minimum-raw-auc", type=float, default=0.94)
    parser.add_argument("--minimum-raw-aupr", type=float, default=0.91)
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


def build_train_covariates(
    train: pd.DataFrame,
    prior_strength: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    frame = train.copy()
    frame["target_id"] = frame["target_id"].astype(str)
    frame["drug_id"] = frame["drug_id"].astype(str)
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    global_rate = float(frame["label"].mean())

    target = (
        frame.groupby("target_id", as_index=False)["label"]
        .agg(["sum", "count"])
        .reset_index()
    )
    target["target_prior_eb"] = (
        target["sum"] + prior_strength * global_rate
    ) / (target["count"] + prior_strength)
    target["log_train_target_count"] = np.log1p(target["count"])

    drug = (
        frame.groupby("drug_id", as_index=False)["label"]
        .agg(["sum", "count"])
        .reset_index()
    )
    drug["drug_prior_eb"] = (
        drug["sum"] + prior_strength * global_rate
    ) / (drug["count"] + prior_strength)
    drug["log_train_drug_count"] = np.log1p(drug["count"])
    return target, drug, global_rate


def prepare_target_risk_table(
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
            ["target_id", "target_prior_eb", "log_train_target_count"]
        ],
        on="target_id",
        how="left",
        validate="many_to_one",
    )
    frame = frame.merge(
        drug_covariates[
            ["drug_id", "drug_prior_eb", "log_train_drug_count"]
        ],
        on="drug_id",
        how="left",
        validate="many_to_one",
    )
    frame["target_prior_eb"] = frame["target_prior_eb"].fillna(global_rate)
    frame["log_train_target_count"] = frame["log_train_target_count"].fillna(0.0)
    frame["drug_prior_eb"] = frame["drug_prior_eb"].fillna(global_rate)
    frame["log_train_drug_count"] = frame["log_train_drug_count"].fillna(0.0)

    negatives = frame[frame["label"] == 0].copy()
    negatives["is_hce"] = (
        negatives["p_raw"] >= confidence_threshold
    ).astype(int)
    # Legacy is_hce is defined only on negatives: it is HCFP, not all-label HCE.
    negatives["is_hcfp"] = negatives["is_hce"]
    negatives["is_ppce"] = (
        (negatives["p_raw"] >= confidence_threshold)
        & (negatives["pcs"] > 0.0)
    ).astype(int)
    negatives["positive_pcs"] = np.maximum(negatives["pcs"], 0.0)

    target = (
        negatives.groupby("target_id", as_index=False)
        .agg(
            target_prior_eb=("target_prior_eb", "first"),
            log_train_target_count=("log_train_target_count", "first"),
            eval_negative_count=("is_ppce", "size"),
            ppce_count=("is_ppce", "sum"),
            hce_count=("is_hce", "sum"),
            mean_positive_pcs_all_negatives=("positive_pcs", "mean"),
            mean_pcs_all_negatives=("pcs", "mean"),
            mean_p_raw_all_negatives=("p_raw", "mean"),
            mean_drug_prior_eb=("drug_prior_eb", "mean"),
            mean_log_train_drug_count=("log_train_drug_count", "mean"),
        )
        .reset_index(drop=True)
    )
    target["ppce_rate_all_negatives"] = (
        target["ppce_count"] / target["eval_negative_count"]
    )
    target["hce_rate_all_negatives"] = (
        target["hce_count"] / target["eval_negative_count"]
    )
    target["hcfp_count"] = target["hce_count"]
    target["hcfp_rate_all_negatives"] = target["hce_rate_all_negatives"]
    target["log_eval_negative_count"] = np.log1p(target["eval_negative_count"])
    return negatives, target


def weighted_mean_and_scale(
    values: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    mean = float(np.average(values, weights=weights))
    variance = float(np.average((values - mean) ** 2, weights=weights))
    scale = math.sqrt(max(variance, 0.0))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Cannot standardize a constant analysis variable")
    return (values - mean) / scale, mean, scale


def standardized_design(
    frame: pd.DataFrame,
    control_columns: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[float, float]]]:
    weights = frame["eval_negative_count"].to_numpy(dtype=float)
    columns = [EXPOSURE] + list(
        CONTROL_COLUMNS if control_columns is None else control_columns
    )
    standardized: List[np.ndarray] = []
    scaling: Dict[str, Tuple[float, float]] = {}
    for column in columns:
        values = frame[column].to_numpy(dtype=float)
        transformed, mean, scale = weighted_mean_and_scale(values, weights)
        standardized.append(transformed)
        scaling[column] = (mean, scale)
    matrix = np.column_stack(
        [np.ones(len(frame), dtype=float)] + standardized
    )
    return matrix, weights, scaling


def variable_control_columns(frame: pd.DataFrame) -> Tuple[List[str], List[str]]:
    weights = frame["eval_negative_count"].to_numpy(dtype=float)
    active: List[str] = []
    dropped: List[str] = []
    for column in CONTROL_COLUMNS:
        values = frame[column].to_numpy(dtype=float)
        mean = float(np.average(values, weights=weights))
        variance = float(np.average((values - mean) ** 2, weights=weights))
        if math.isfinite(variance) and variance > 0.0:
            active.append(column)
        else:
            dropped.append(column)
    return active, dropped


def fit_grouped_binomial(
    design: np.ndarray,
    successes: np.ndarray,
    totals: np.ndarray,
) -> float:
    successes = successes.astype(float)
    totals = totals.astype(float)

    def objective(beta: np.ndarray) -> Tuple[float, np.ndarray]:
        eta = design @ beta
        penalty = 1e-8 * float(np.dot(beta[1:], beta[1:]))
        value = float(np.sum(totals * np.logaddexp(0.0, eta) - successes * eta))
        value += penalty
        probability = expit(eta)
        gradient = design.T @ (totals * probability - successes)
        gradient[1:] += 2e-8 * beta[1:]
        return value, gradient

    result = minimize(
        fun=lambda beta: objective(beta)[0],
        x0=np.zeros(design.shape[1], dtype=float),
        jac=lambda beta: objective(beta)[1],
        method="BFGS",
        options={"maxiter": 2000, "gtol": 1e-8},
    )
    if not np.all(np.isfinite(result.x)):
        raise RuntimeError("Grouped-binomial optimization produced non-finite values")
    return float(result.x[1])


def fit_weighted_linear(
    design: np.ndarray,
    outcome: np.ndarray,
    weights: np.ndarray,
) -> float:
    model = LinearRegression(fit_intercept=False)
    model.fit(design, outcome.astype(float), sample_weight=weights)
    return float(model.coef_[1])


def fit_effect(
    frame: pd.DataFrame,
    outcome: str,
    method: str,
    exposure_override: Optional[np.ndarray] = None,
    control_columns: Optional[Sequence[str]] = None,
) -> float:
    design, weights, _ = standardized_design(frame, control_columns)
    if exposure_override is not None:
        design[:, 1] = exposure_override
    if method == "grouped_binomial":
        successes = frame[outcome].to_numpy(dtype=float)
        totals = frame["eval_negative_count"].to_numpy(dtype=float)
        return fit_grouped_binomial(design, successes, totals)
    if method == "weighted_linear":
        return fit_weighted_linear(
            design,
            frame[outcome].to_numpy(dtype=float),
            weights,
        )
    raise ValueError("Unknown method: %s" % method)


def bootstrap_effects(
    frame: pd.DataFrame,
    outcome: str,
    method: str,
    iterations: int,
    rng: np.random.Generator,
    control_columns: Sequence[str],
) -> np.ndarray:
    values: List[float] = []
    for _ in range(max(0, int(iterations))):
        indexes = rng.integers(0, len(frame), size=len(frame))
        sampled = frame.iloc[indexes].reset_index(drop=True)
        try:
            value = fit_effect(
                sampled,
                outcome,
                method,
                control_columns=control_columns,
            )
        except Exception:
            continue
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def permutation_effects(
    frame: pd.DataFrame,
    outcome: str,
    method: str,
    iterations: int,
    rng: np.random.Generator,
    control_columns: Sequence[str],
) -> np.ndarray:
    design, weights, _ = standardized_design(frame, control_columns)
    exposure = design[:, 1]
    controls = design[:, [0] + list(range(2, design.shape[1]))]
    exposure_model = LinearRegression(fit_intercept=False)
    exposure_model.fit(controls, exposure, sample_weight=weights)
    fitted = exposure_model.predict(controls)
    residual = exposure - fitted

    values: List[float] = []
    for _ in range(max(0, int(iterations))):
        permuted_exposure = fitted + rng.permutation(residual)
        try:
            value = fit_effect(
                frame,
                outcome,
                method,
                exposure_override=permuted_exposure,
                control_columns=control_columns,
            )
        except Exception:
            continue
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def prediction_metrics(frame: pd.DataFrame) -> Dict[str, float]:
    labels = frame["label"].to_numpy(dtype=int)
    probabilities = frame["p_raw"].to_numpy(dtype=float)
    return {
        "raw_auc": float(roc_auc_score(labels, probabilities)),
        "raw_aupr": float(average_precision_score(labels, probabilities)),
    }


def analyze_split(
    split: str,
    prediction: pd.DataFrame,
    prediction_path: Path,
    target_covariates: pd.DataFrame,
    drug_covariates: pd.DataFrame,
    global_rate: float,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, object]], pd.DataFrame]:
    metrics = prediction_metrics(prediction)
    negatives, target = prepare_target_risk_table(
        prediction=prediction,
        target_covariates=target_covariates,
        drug_covariates=drug_covariates,
        global_rate=global_rate,
        confidence_threshold=args.confidence_threshold,
    )
    if len(target) < args.minimum_targets:
        raise ValueError(
            "%s has %d negative-risk targets, below --minimum-targets=%d"
            % (split, len(target), args.minimum_targets)
        )

    active_controls, dropped_controls = variable_control_columns(target)

    rows: List[Dict[str, object]] = []
    for outcome_index, (outcome, method, role) in enumerate(OUTCOMES):
        rng = np.random.default_rng(args.seed + outcome_index + (0 if split == "valid" else 1000))
        effect = fit_effect(
            target,
            outcome,
            method,
            control_columns=active_controls,
        )
        bootstrap = bootstrap_effects(
            target,
            outcome,
            method,
            args.bootstrap_iterations,
            rng,
            active_controls,
        )
        permutations = permutation_effects(
            target,
            outcome,
            method,
            args.permutation_iterations,
            rng,
            active_controls,
        )
        ci_low = (
            float(np.quantile(bootstrap, 0.025))
            if len(bootstrap)
            else math.nan
        )
        ci_high = (
            float(np.quantile(bootstrap, 0.975))
            if len(bootstrap)
            else math.nan
        )
        p_one_sided = (
            float((1 + np.sum(permutations >= effect)) / (1 + len(permutations)))
            if len(permutations)
            else math.nan
        )
        positive = bool(
            effect > 0.0
            and math.isfinite(ci_low)
            and ci_low > 0.0
            and math.isfinite(p_one_sided)
            and p_one_sided < 0.05
        )
        rows.append(
            {
                "dataset": args.dataset,
                "split": split,
                "analysis_role": role,
                "risk_set": "all_negative_labeled_predictions",
                "outcome": outcome,
                "method": method,
                "exposure": EXPOSURE,
                "controls": ",".join(active_controls),
                "dropped_constant_controls": ",".join(dropped_controls),
                "target_count": int(len(target)),
                "negative_sample_count": int(len(negatives)),
                "ppce_count": int(negatives["is_ppce"].sum()),
                "ppce_rate_all_negatives": float(negatives["is_ppce"].mean()),
                "hce_count": int(negatives["is_hce"].sum()),
                "hce_rate_all_negatives": float(negatives["is_hce"].mean()),
                "target_prior_standardized_beta": effect,
                "target_prior_odds_ratio_per_sd": (
                    float(math.exp(np.clip(effect, -50.0, 50.0)))
                    if method == "grouped_binomial"
                    else math.nan
                ),
                "bootstrap_beta_ci_low": ci_low,
                "bootstrap_beta_ci_high": ci_high,
                "bootstrap_successful_iterations": int(len(bootstrap)),
                "permutation_p_one_sided": p_one_sided,
                "permutation_successful_iterations": int(len(permutations)),
                "positive_independent_effect": positive,
                "raw_auc": metrics["raw_auc"],
                "raw_aupr": metrics["raw_aupr"],
                "performance_floor_met": bool(
                    metrics["raw_auc"] >= args.minimum_raw_auc
                    and metrics["raw_aupr"] >= args.minimum_raw_aupr
                ),
                "prediction_file": relative_path(prediction_path),
                "prediction_sha256": sha256_file(prediction_path),
                "claim_boundary": (
                    "controlled target-level association; causal induction "
                    "requires a pre-specified intervention experiment"
                ),
            }
        )

    target.insert(0, "split", split)
    return rows, target


def write_report(
    path: Path,
    summary: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    primary = summary[
        summary["analysis_role"] == "primary_ppce_incidence"
    ].copy()
    replicated = bool(
        set(primary["split"]) == {"valid", "test"}
        and primary["positive_independent_effect"].all()
        and primary["performance_floor_met"].all()
    )
    lines = [
        "# Target-Prior Independent PPCE Risk Audit",
        "",
        "## Design",
        "",
        "- Risk set: every negative-labeled prediction; no conditioning on HCE.",
        "- Primary outcome: PPCE incidence per target.",
        "- Exposure: fixed-strength (10) shrinkage of the train target positive rate toward the global train rate.",
        "- Controls are standardized per split; zero-variance controls are dropped and recorded in the summary.",
        "- Inference: target bootstrap CI and a regressor-residual permutation of the prior exposure; this is not the classical outcome-residual Freedman-Lane procedure.",
        "- `p_raw` is not controlled because it is a candidate mediator.",
        "",
        "## Primary Result",
        "",
        "```text",
        primary.to_string(index=False),
        "```",
        "",
        "## Decision",
        "",
        "- Replicated positive independent association: `%s`." % replicated,
        "- A positive result supports an independent association, not causal induction.",
        "- Causal wording requires a separate prior intervention with fixed samples and checkpoint.",
        "",
        "## Reproducibility",
        "",
        "- Confidence threshold: `%s`." % args.confidence_threshold,
        "- Bootstrap iterations: `%d`." % args.bootstrap_iterations,
        "- Permutation iterations: `%d`." % args.permutation_iterations,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    train_path = resolve_path(args.train_csv)
    valid_path = resolve_path(args.valid_pred)
    test_path = resolve_path(args.test_pred)
    train = load_csv(train_path, REQUIRED_TRAIN_COLUMNS)
    valid = load_csv(valid_path, REQUIRED_PREDICTION_COLUMNS)
    test = load_csv(test_path, REQUIRED_PREDICTION_COLUMNS)
    target_covariates, drug_covariates, global_rate = build_train_covariates(
        train,
        prior_strength=args.prior_strength,
    )

    summary_rows: List[Dict[str, object]] = []
    detail_frames: List[pd.DataFrame] = []
    for split, prediction, path in (
        ("valid", valid, valid_path),
        ("test", test, test_path),
    ):
        rows, detail = analyze_split(
            split=split,
            prediction=prediction,
            prediction_path=path,
            target_covariates=target_covariates,
            drug_covariates=drug_covariates,
            global_rate=global_rate,
            args=args,
        )
        summary_rows.extend(rows)
        detail_frames.append(detail)

    summary = pd.DataFrame(summary_rows)
    primary_mask = summary["analysis_role"] == "primary_ppce_incidence"
    replicated = bool(
        set(summary.loc[primary_mask, "split"]) == {"valid", "test"}
        and summary.loc[primary_mask, "positive_independent_effect"].all()
        and summary.loc[primary_mask, "performance_floor_met"].all()
    )
    summary["primary_effect_replicated_valid_test"] = replicated

    output_base = resolve_path("outputs/metrics/%s" % args.output_prefix)
    summary_path = output_base.with_name(output_base.name + "_summary.csv")
    detail_path = output_base.with_name(output_base.name + "_target_detail.csv")
    report_path = resolve_path("docs/%s_report.md" % args.output_prefix)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    pd.concat(detail_frames, ignore_index=True).to_csv(detail_path, index=False)
    write_report(report_path, summary, args)

    display_columns = [
        "split",
        "analysis_role",
        "target_prior_standardized_beta",
        "target_prior_odds_ratio_per_sd",
        "bootstrap_beta_ci_low",
        "bootstrap_beta_ci_high",
        "permutation_p_one_sided",
        "positive_independent_effect",
        "raw_auc",
        "raw_aupr",
        "performance_floor_met",
        "primary_effect_replicated_valid_test",
    ]
    print(summary[display_columns].to_string(index=False))
    print("Saved:")
    print(summary_path)
    print(detail_path)
    print(report_path)


if __name__ == "__main__":
    main()

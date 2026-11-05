#!/usr/bin/env python
"""Target-level prior-assignment shuffle negative control for PPCE risk."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_target_prior_ppce_independent_risk import (
    REQUIRED_PREDICTION_COLUMNS,
    REQUIRED_TRAIN_COLUMNS,
    build_train_covariates,
    fit_effect,
    load_csv,
    prepare_target_risk_table,
    relative_path,
    resolve_path,
)
from src.utils.analyze_target_prior_ppce_independent_risk import variable_control_columns, permutation_effects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Shuffle the train-derived target-prior assignment across targets "
            "and refit the controlled grouped-binomial PPCE model."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--valid-pred", required=True)
    parser.add_argument("--test-pred", required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--prior-strength", type=float, default=10.0)
    parser.add_argument("--shuffle-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20240620)
    parser.add_argument("--minimum-targets", type=int, default=20)
    parser.add_argument("--output-prefix", required=True)
    return parser.parse_args()


def shuffled_effects(
    target: pd.DataFrame,
    iterations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Permute exposure residuals after adjustment, as described in manuscript M1."""
    controls, _ = variable_control_columns(target)
    return permutation_effects(target, "ppce_count", "grouped_binomial", iterations, rng, controls)


def analyze_split(
    split: str,
    prediction: pd.DataFrame,
    prediction_path: Path,
    target_covariates: pd.DataFrame,
    drug_covariates: pd.DataFrame,
    global_rate: float,
    args: argparse.Namespace,
) -> tuple[Dict[str, object], pd.DataFrame]:
    negatives, target = prepare_target_risk_table(
        prediction=prediction,
        target_covariates=target_covariates,
        drug_covariates=drug_covariates,
        global_rate=global_rate,
        confidence_threshold=args.confidence_threshold,
    )
    if len(target) < args.minimum_targets:
        raise ValueError(
            "%s has %d targets, below --minimum-targets=%d"
            % (split, len(target), args.minimum_targets)
        )

    controls, _ = variable_control_columns(target)
    observed = fit_effect(
        target,
        outcome="ppce_count",
        method="grouped_binomial",
        control_columns=controls,
    )
    rng = np.random.default_rng(
        args.seed + (0 if split == "valid" else 1000)
    )
    shuffled = shuffled_effects(
        target=target,
        iterations=args.shuffle_iterations,
        rng=rng,
    )
    null_low = (
        float(np.quantile(shuffled, 0.025)) if len(shuffled) else math.nan
    )
    null_high = (
        float(np.quantile(shuffled, 0.975)) if len(shuffled) else math.nan
    )
    p_one_sided = (
        float((1 + np.sum(shuffled >= observed)) / (1 + len(shuffled)))
        if len(shuffled)
        else math.nan
    )
    null_centered = bool(
        math.isfinite(null_low)
        and math.isfinite(null_high)
        and null_low <= 0.0 <= null_high
    )
    passed = bool(
        observed > 0.0
        and math.isfinite(null_high)
        and observed > null_high
        and math.isfinite(p_one_sided)
        and p_one_sided < 0.05
        and null_centered
    )

    distribution = pd.DataFrame(
        {
            "dataset": args.dataset,
            "split": split,
            "shuffle_index": np.arange(len(shuffled), dtype=int),
            "shuffled_target_prior_beta": shuffled,
        }
    )
    row: Dict[str, object] = {
        "dataset": args.dataset,
        "split": split,
        "confidence_threshold": args.confidence_threshold,
        "target_count": int(len(target)),
        "negative_sample_count": int(len(negatives)),
        "hce_count": int(negatives["is_hce"].sum()),
        "ppce_count": int(negatives["is_ppce"].sum()),
        "observed_target_prior_beta": observed,
        "observed_target_prior_or_per_sd": float(math.exp(observed)),
        "shuffle_beta_mean": (
            float(np.mean(shuffled)) if len(shuffled) else math.nan
        ),
        "shuffle_beta_std": (
            float(np.std(shuffled, ddof=1)) if len(shuffled) > 1 else math.nan
        ),
        "shuffle_beta_ci_low": null_low,
        "shuffle_beta_ci_high": null_high,
        "shuffle_empirical_p_one_sided": p_one_sided,
        "shuffle_successful_iterations": int(len(shuffled)),
        "shuffle_null_contains_zero": null_centered,
        "negative_control_passed": passed,
        "prediction_file": relative_path(prediction_path),
    }
    return row, distribution


def write_report(
    path: Path,
    summary: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    replicated = bool(
        set(summary["split"]) == {"valid", "test"}
        and summary["negative_control_passed"].all()
    )
    lines = [
        "# Target-Prior Shuffle Negative Control",
        "",
        "## Design",
        "",
        "- Unit of shuffling: target.",
        "- Regressor-residual permutation: standardized smoothed training prior is regressed on controls; its target-level residuals are permuted and added back to the fitted component.",
        "- Outcomes and controls are unchanged from the primary PPCE-incidence model.",
        "- Predictions, labels, HCE, PPCE, and model weights are not modified.",
        "",
        "## Results",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Decision",
        "",
        "- Valid/test negative-control replication: `%s`." % replicated,
        "- Passing means the real prior beta exceeds the shuffled 97.5th percentile,",
        "  the shuffled interval contains zero, and empirical one-sided p < 0.05.",
        "- This is a controlled exposure-residual diagnostic, not an outcome-residual Freedman-Lane procedure or proof of causal induction.",
        "",
        "## Reproducibility",
        "",
        "- Confidence threshold: `%s`." % args.confidence_threshold,
        "- Requested shuffles per split: `%d`." % args.shuffle_iterations,
        "- Seed: `%d`." % args.seed,
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

    rows: List[Dict[str, object]] = []
    distributions: List[pd.DataFrame] = []
    for split, prediction, path in (
        ("valid", valid, valid_path),
        ("test", test, test_path),
    ):
        row, distribution = analyze_split(
            split=split,
            prediction=prediction,
            prediction_path=path,
            target_covariates=target_covariates,
            drug_covariates=drug_covariates,
            global_rate=global_rate,
            args=args,
        )
        rows.append(row)
        distributions.append(distribution)

    summary = pd.DataFrame(rows)
    replicated = bool(
        set(summary["split"]) == {"valid", "test"}
        and summary["negative_control_passed"].all()
    )
    summary["negative_control_replicated_valid_test"] = replicated

    output_base = resolve_path("outputs/metrics/%s" % args.output_prefix)
    summary_path = output_base.with_name(output_base.name + "_summary.csv")
    distribution_path = output_base.with_name(
        output_base.name + "_null_distribution.csv"
    )
    report_path = resolve_path("docs/%s_report.md" % args.output_prefix)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    pd.concat(distributions, ignore_index=True).to_csv(
        distribution_path,
        index=False,
    )
    write_report(report_path, summary, args)

    print(summary.to_string(index=False))
    print("Saved:")
    print(summary_path)
    print(distribution_path)
    print(report_path)


if __name__ == "__main__":
    main()

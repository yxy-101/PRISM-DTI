"""Manuscript analysis functions retained from the research implementation."""
from __future__ import annotations
import math
from typing import Dict
import numpy as np
import pandas as pd
def ppce_target_cluster_inference(
    dataset_name: str,
    split: str,
    predictions: pd.DataFrame,
    target_priors: pd.DataFrame,
    threshold: float,
    high_prior_threshold: float,
    bootstrap_draws: int,
    permutations: int,
    seed: int,
) -> Dict[str, object]:
    required = {"target_id", "label", "p_raw", "pcs"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise KeyError(f"Missing prediction columns for {dataset_name}/{split}: {missing}")
    frame = predictions.copy()
    frame["target_id"] = frame["target_id"].astype(str)
    frame = frame.merge(
        target_priors[["target_positive_rate"]],
        left_on="target_id",
        right_index=True,
        how="left",
        validate="many_to_one",
    )
    missing_prior_mask = frame["target_positive_rate"].isna()
    missing_prior_sample_count = int(missing_prior_mask.sum())
    missing_prior_target_count = int(
        frame.loc[missing_prior_mask, "target_id"].nunique()
    )
    frame = frame.loc[~missing_prior_mask].copy()
    if frame.empty:
        raise ValueError(
            f"No samples have train-derived target priors for {dataset_name}/{split}."
        )
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    frame["p_raw"] = pd.to_numeric(frame["p_raw"], errors="raise")
    frame["pcs"] = pd.to_numeric(frame["pcs"], errors="raise")
    frame["high_prior"] = frame["target_positive_rate"] >= high_prior_threshold
    frame["hcfp"] = (frame["label"] == 0) & (frame["p_raw"] >= threshold)
    frame["ppce"] = frame["hcfp"] & (frame["pcs"] > 0.0)

    by_target = frame.groupby("target_id", sort=True).agg(
        high_prior=("high_prior", "first"),
        hcfp_count=("hcfp", "sum"),
        ppce_count=("ppce", "sum"),
    )
    informative = by_target[by_target["hcfp_count"] > 0].copy()
    informative["ppce_fraction"] = (
        informative["ppce_count"] / informative["hcfp_count"]
    )
    common = {
        "dataset": dataset_name,
        "split": split,
        "confidence_threshold": threshold,
        "high_prior_threshold": high_prior_threshold,
        "sample_count": int(len(frame)),
        "target_count": int(frame["target_id"].nunique()),
        "excluded_missing_prior_sample_count": missing_prior_sample_count,
        "excluded_missing_prior_target_count": missing_prior_target_count,
        "hcfp_count": int(frame["hcfp"].sum()),
        "ppce_count": int(frame["ppce"].sum()),
        "informative_hcfp_target_count": int(len(informative)),
        "informative_high_prior_target_count": int(informative["high_prior"].sum()),
        "informative_non_high_prior_target_count": int((~informative["high_prior"]).sum()),
        "bootstrap_draws": bootstrap_draws,
        "permutations": permutations,
        "inference_unit": "target",
    }
    if informative["high_prior"].nunique() < 2:
        return {
            **common,
            "pair_weighted_odds_ratio_haldane": float("nan"),
            "pair_weighted_log_odds_ratio": float("nan"),
            "pair_weighted_log_or_ci_low": float("nan"),
            "pair_weighted_log_or_ci_high": float("nan"),
            "pair_weighted_risk_difference": float("nan"),
            "pair_weighted_risk_difference_ci_low": float("nan"),
            "pair_weighted_risk_difference_ci_high": float("nan"),
            "target_equal_mean_ppce_fraction_high": float("nan"),
            "target_equal_mean_ppce_fraction_non_high": float("nan"),
            "target_equal_mean_difference": float("nan"),
            "target_equal_mean_difference_ci_low": float("nan"),
            "target_equal_mean_difference_ci_high": float("nan"),
            "target_equal_permutation_p_two_sided": float("nan"),
            "estimable": False,
            "not_estimable_reason": (
                "HCFP targets do not include both high-prior and non-high-prior groups"
            ),
        }

    observed_pair = pair_weighted_statistics(informative)
    observed_equal = target_equal_statistics(informative)
    rng = np.random.default_rng(seed + stable_offset(dataset_name, split))
    boot_pair_log_or = np.empty(bootstrap_draws, dtype=float)
    boot_pair_rd = np.empty(bootstrap_draws, dtype=float)
    boot_equal_diff = np.empty(bootstrap_draws, dtype=float)
    n_targets = len(informative)
    for draw in range(bootstrap_draws):
        sampled = informative.iloc[rng.integers(0, n_targets, size=n_targets)]
        pair_stats = pair_weighted_statistics(sampled)
        equal_stats = target_equal_statistics(sampled)
        boot_pair_log_or[draw] = pair_stats["log_odds_ratio"]
        boot_pair_rd[draw] = pair_stats["risk_difference"]
        boot_equal_diff[draw] = equal_stats["mean_difference"]

    high_labels = informative["high_prior"].to_numpy(bool)
    fractions = informative["ppce_fraction"].to_numpy(float)
    permuted_differences = np.empty(permutations, dtype=float)
    for draw in range(permutations):
        permuted = rng.permutation(high_labels)
        permuted_differences[draw] = (
            float(np.mean(fractions[permuted]))
            - float(np.mean(fractions[~permuted]))
        )
    observed_difference = observed_equal["mean_difference"]
    p_two_sided = (
        1.0
        + float(np.sum(np.abs(permuted_differences) >= abs(observed_difference)))
    ) / (permutations + 1.0)

    return {
        **common,
        "pair_weighted_odds_ratio_haldane": observed_pair["odds_ratio"],
        "pair_weighted_log_odds_ratio": observed_pair["log_odds_ratio"],
        "pair_weighted_log_or_ci_low": percentile(boot_pair_log_or, 0.025),
        "pair_weighted_log_or_ci_high": percentile(boot_pair_log_or, 0.975),
        "pair_weighted_risk_difference": observed_pair["risk_difference"],
        "pair_weighted_risk_difference_ci_low": percentile(boot_pair_rd, 0.025),
        "pair_weighted_risk_difference_ci_high": percentile(boot_pair_rd, 0.975),
        "target_equal_mean_ppce_fraction_high": observed_equal["high_mean"],
        "target_equal_mean_ppce_fraction_non_high": observed_equal["non_high_mean"],
        "target_equal_mean_difference": observed_difference,
        "target_equal_mean_difference_ci_low": percentile(boot_equal_diff, 0.025),
        "target_equal_mean_difference_ci_high": percentile(boot_equal_diff, 0.975),
        "target_equal_permutation_p_two_sided": p_two_sided,
        "estimable": True,
        "not_estimable_reason": "",
    }

def pair_weighted_statistics(frame: pd.DataFrame) -> Dict[str, float]:
    high = frame[frame["high_prior"]]
    non_high = frame[~frame["high_prior"]]
    a = float(high["ppce_count"].sum())
    b = float(high["hcfp_count"].sum() - a)
    c = float(non_high["ppce_count"].sum())
    d = float(non_high["hcfp_count"].sum() - c)
    odds_ratio = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
    high_risk = a / max(1.0, a + b)
    non_high_risk = c / max(1.0, c + d)
    return {
        "odds_ratio": float(odds_ratio),
        "log_odds_ratio": float(math.log(odds_ratio)),
        "risk_difference": float(high_risk - non_high_risk),
    }

def target_equal_statistics(frame: pd.DataFrame) -> Dict[str, float]:
    high_mean = float(frame.loc[frame["high_prior"], "ppce_fraction"].mean())
    non_high_mean = float(
        frame.loc[~frame["high_prior"], "ppce_fraction"].mean()
    )
    return {
        "high_mean": high_mean,
        "non_high_mean": non_high_mean,
        "mean_difference": high_mean - non_high_mean,
    }

def percentile(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    return float("nan") if not len(finite) else float(np.quantile(finite, q))

def stable_offset(dataset_name: str, split: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(dataset_name + split))

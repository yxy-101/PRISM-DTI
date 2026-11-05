"""Manuscript Eqs.18--25: HCFP risk, PPCE and training-prior association."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.analyze_target_prior_ppce_independent_risk import (
    build_train_covariates, prepare_target_risk_table, fit_effect,
    variable_control_columns, bootstrap_effects, permutation_effects,
)
from src.utils.target_equal_ppce import ppce_target_cluster_inference

ROOT = Path(__file__).resolve().parents[2]
KEYS = ["drug_id", "target_id", "label"]


def read_pairs(path, prediction=False):
    required = KEYS + (["p_raw", "p_do", "pcs"] if prediction else [])
    columns = pd.read_csv(path, nrows=0).columns
    if not set(required).issubset(columns):
        raise ValueError("Missing input columns: " + str(path))
    frame = pd.read_csv(path, dtype={"drug_id": str, "target_id": str}, usecols=required)
    if not set(required).issubset(frame) or frame[required].isna().any().any():
        raise ValueError("Missing input columns/values: " + str(path))
    if frame.duplicated(KEYS[:2]).any() or not frame.label.isin([0, 1]).all():
        raise ValueError("Invalid pairs or labels: " + str(path))
    if prediction:
        p = frame[["p_raw", "p_do"]].to_numpy(float)
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("Invalid probabilities: " + str(path))
        if not np.allclose(frame.pcs, frame.p_raw - frame.p_do, atol=3e-8, rtol=0):
            raise ValueError("PCS identity mismatch: " + str(path))
    return frame.sort_values(KEYS[:2]).reset_index(drop=True)


def bh_adjust(values):
    p = np.asarray(values, float)
    q = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    if len(order):
        ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
        q[order] = np.minimum(1., np.minimum.accumulate(ranked[::-1])[::-1])
    return q


def load_inputs(manifest):
    specs = json.loads(manifest.read_text(encoding="utf-8"))["datasets"]
    sources = {manifest.relative_to(ROOT).as_posix(): hashlib.sha256(manifest.read_bytes()).hexdigest()}
    result = {}
    for dataset, spec in specs.items():
        frames = {}
        for key in ("train_csv", "valid_csv", "test_csv", "valid_pred", "test_pred"):
            path = ROOT / spec[key]
            frames[key] = read_pairs(path, key.endswith("_pred"))
            sources[spec[key]] = hashlib.sha256(path.read_bytes()).hexdigest()
        for split in ("valid", "test"):
            if not frames[split + "_csv"][KEYS].equals(frames[split + "_pred"][KEYS]):
                raise ValueError("Evaluation pair/label identity mismatch: " + dataset + "/" + split)
        for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
            a = pd.MultiIndex.from_frame(frames[left + "_csv"][KEYS[:2]])
            b = pd.MultiIndex.from_frame(frames[right + "_csv"][KEYS[:2]])
            if len(a.intersection(b)):
                raise ValueError("Split pair leakage: " + dataset + "/" + left + "/" + right)
        result[dataset] = frames
    return result, sources


def analyze(inputs, output, *, bootstrap, permutations, seed):
    associations, contrasts = [], []
    for number, (dataset, frames) in enumerate(inputs.items()):
        train = frames["train_csv"]
        covariates, drug, rate = build_train_covariates(train, prior_strength=10.)
        raw_prior = train.groupby("target_id").label.mean().to_frame("target_positive_rate")
        for split in ("valid", "test"):
            prediction = frames[split + "_pred"]
            for threshold in (0.7, 0.8, 0.9):
                negatives, table = prepare_target_risk_table(prediction, covariates, drug, rate, threshold)
                # Eq.22 is defined for training-seen targets; never assign unseen targets to the other group.
                table = table[table.target_id.isin(raw_prior.index)].reset_index(drop=True)
                active, dropped = variable_control_columns(table)
                table.to_csv(output / f"{dataset}_{split}_{threshold}_target_risk.csv", index=False)
                for outcome in ("hcfp_count", "ppce_count"):
                    row = dict(dataset=dataset, split=split, threshold=threshold, outcome=outcome,
                               controls=",".join(active), dropped_controls=",".join(dropped),
                               targets=len(table), negatives=int(table.eval_negative_count.sum()),
                               events=int(table[outcome].sum()), risk_set="all benchmark-negative pairs")
                    try:
                        effect = fit_effect(table, outcome, "grouped_binomial", control_columns=active)
                    except (ValueError, RuntimeError) as error:
                        associations.append(dict(row, status="not_estimable", reason=str(error)))
                        continue
                    rng = np.random.default_rng(seed + number * 10000 + int(threshold * 100) + (0 if split == "valid" else 1000))
                    boots = bootstrap_effects(table, outcome, "grouped_binomial", bootstrap, rng, active)
                    null = permutation_effects(table, outcome, "grouped_binomial", permutations, rng, active)
                    low, high = np.quantile(boots, [.025, .975]) if len(boots) else (np.nan, np.nan)
                    p = (1 + int((null >= effect).sum())) / (1 + len(null)) if len(null) else np.nan
                    associations.append(dict(row, status="estimated", beta=effect,
                                             odds_ratio_per_sd=float(np.exp(effect)), beta_ci_low=low,
                                             beta_ci_high=high, permutation_p_one_sided=p,
                                             bootstrap_successful=len(boots), permutation_successful=len(null)))
            contrast = ppce_target_cluster_inference(dataset, split, prediction, raw_prior,
                                                     0.8, 0.7, bootstrap, permutations, seed)
            contrasts.append(contrast)
    table = pd.DataFrame(associations)
    table["permutation_bh_q"] = np.nan
    for _, group in table.groupby(["split", "threshold", "outcome"]):
        table.loc[group.index, "permutation_bh_q"] = bh_adjust(group.get("permutation_p_one_sided", pd.Series(np.nan, index=group.index)))
    table.to_csv(output / "risk_associations.csv", index=False)
    contrasts = pd.DataFrame(contrasts)
    contrasts["target_equal_bh_q"] = np.nan
    for _, group in contrasts.groupby("split"):
        contrasts.loc[group.index, "target_equal_bh_q"] = bh_adjust(group.target_equal_permutation_p_two_sided)
    contrasts.to_csv(output / "target_equal_ppce.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run"), default="preflight")
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/submission/final_frozen_predictions.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/manuscript_analysis")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    if args.bootstrap <= 0 or args.permutations <= 0:
        raise ValueError("Positive bootstrap and permutation counts are required.")
    inputs, sources = load_inputs(args.manifest.resolve())
    if args.mode == "preflight":
        print("Verified pair/label identities and PCS for", len(inputs), "datasets.")
        return
    if args.output.exists():
        raise FileExistsError("Preserve existing results: " + str(args.output))
    args.output.mkdir(parents=True)
    protocol = dict(sources=sources, bootstrap=args.bootstrap, permutations=args.permutations, seed=args.seed,
                    source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    estimator="training prior lambda=10; negative-count weighted standardization; grouped binomial; ridge=1e-8",
                    permutation="regress standardized exposure on controls, then permute exposure residuals at target level",
                    multiplicity="BH across datasets separately for each split/threshold/outcome; target-equal contrast separately per split",
                    limitation="Reconstruction settings, not certified historical randomization settings or independent confirmation")
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    analyze(inputs, args.output, bootstrap=args.bootstrap, permutations=args.permutations, seed=args.seed)
    print("Saved manuscript analysis to", args.output)


if __name__ == "__main__":
    main()

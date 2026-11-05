#!/usr/bin/env python
"""Prepare DeepDTA Davis files for the Stage 8B readiness audit.

This script converts existing DeepDTA Davis files into PRISM-DTI processed
CSV splits and writes value-distribution plus threshold-sensitivity audits. It
does not train, predict, create checkpoints, or modify model structure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEEPDTA_DAVIS_DIR = Path("data/raw/deepdta/davis")
DEFAULT_PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_DISTRIBUTION_OUT = ROOT / "outputs" / "metrics" / "stage8b_davis_value_distribution.csv"
DEFAULT_THRESHOLD_OUT = ROOT / "outputs" / "metrics" / "stage8b_davis_threshold_sensitivity.csv"
DEFAULT_ALL_PAIRS_OUT = ROOT / "data" / "processed" / "stage8b_davis_all_pairs.csv"
DEFAULT_REPORT_OUT = ROOT / "docs" / "stage8b_davis_preparation_report.md"

THRESHOLDS = (5.0, 6.0, 7.0)
QUANTILES = (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DeepDTA Davis into Stage 8B processed CSV splits."
    )
    parser.add_argument("--deepdta-davis-dir", default=str(DEFAULT_DEEPDTA_DAVIS_DIR))
    parser.add_argument("--processed-dir", default=str(DEFAULT_PROCESSED_DIR))
    parser.add_argument("--distribution-out", default=str(DEFAULT_DISTRIBUTION_OUT))
    parser.add_argument("--threshold-out", default=str(DEFAULT_THRESHOLD_OUT))
    parser.add_argument("--all-pairs-out", default=str(DEFAULT_ALL_PAIRS_OUT))
    parser.add_argument("--report-out", default=str(DEFAULT_REPORT_OUT))
    parser.add_argument("--valid-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> Dict[str, Any] | List[Any]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_y(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open("rb") as handle:
        try:
            value = pickle.load(handle, encoding="latin1")
        except TypeError:
            handle.seek(0)
            value = pickle.load(handle)
    return np.asarray(value, dtype=float)


def flatten_train_folds(train_folds: Sequence[Any]) -> List[int]:
    if not train_folds:
        return []
    if isinstance(train_folds[0], list):
        flattened: List[int] = []
        for fold in train_folds:
            flattened.extend(int(index) for index in fold)
        return flattened
    return [int(index) for index in train_folds]


def stable_drug_id(source_id: str, smiles: str) -> str:
    clean = "".join(ch if ch.isalnum() else "_" for ch in str(source_id)).strip("_")
    if clean:
        return f"davis_drug_{clean}"
    digest = hashlib.md5(smiles.encode("utf-8")).hexdigest()
    return f"davis_drug_{digest[:12]}"


def stable_target_id(source_id: str) -> str:
    clean = "".join(ch if ch.isalnum() else "_" for ch in str(source_id)).strip("_")
    return f"davis_target_{clean}" if clean else "davis_target_unknown"


def infer_value_scale(values: np.ndarray) -> Tuple[bool, bool, str]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False, False, "no finite values"
    min_value = float(np.min(finite))
    max_value = float(np.max(finite))
    median = float(np.median(finite))
    pkd_like = min_value >= 0.0 and max_value <= 15.0 and median <= 12.0
    kd_nm_like = min_value > 0.0 and max_value >= 1000.0 and median >= 100.0
    if pkd_like:
        return True, False, "pKd-like bounded affinity scale"
    if kd_nm_like:
        return False, True, "raw Kd nM-like scale; pKd conversion should be confirmed before binary labels"
    return False, False, "unrecognized affinity scale"


def pkd_from_kd_nm(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return 9.0 - np.log10(numeric)


def distribution_row(
    y: np.ndarray,
    ligand_count: int,
    protein_count: int,
    train_count: int,
    valid_count: int,
    test_count: int,
    pkd_like: bool,
    kd_nm_like: bool,
    scale_note: str,
    valid_fraction: float,
    seed: int,
) -> Dict[str, Any]:
    finite = y[np.isfinite(y)]
    row: Dict[str, Any] = {
        "dataset": "davis",
        "source": "DeepDTA",
        "y_shape_rows": int(y.shape[0]) if y.ndim >= 1 else 0,
        "y_shape_cols": int(y.shape[1]) if y.ndim >= 2 else 0,
        "ligand_count": int(ligand_count),
        "protein_count": int(protein_count),
        "shape_matches_ligand_protein_count": bool(
            y.ndim == 2 and y.shape == (ligand_count, protein_count)
        ),
        "finite_count": int(np.isfinite(y).sum()),
        "nan_count": int(np.isnan(y).sum()),
        "inf_count": int(np.isinf(y).sum()),
        "missing_count": int((~np.isfinite(y)).sum()),
        "train_sample_count": int(train_count),
        "valid_sample_count": int(valid_count),
        "test_sample_count": int(test_count),
        "valid_fraction_from_deepdta_train": float(valid_fraction),
        "split_seed": int(seed),
        "is_pkd_like": bool(pkd_like),
        "is_kd_nm_like": bool(kd_nm_like),
        "scale_note": scale_note,
        "label_generated": bool(pkd_like),
        "default_label_policy": "affinity_value >= 5.0" if pkd_like else "not_generated_raw_y_not_pkd_like",
    }
    if finite.size:
        row.update(
            {
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite)),
            }
        )
        for quantile in QUANTILES:
            row[f"q{quantile}"] = float(np.percentile(finite, quantile))
    return row


def build_all_pairs(
    y: np.ndarray,
    ligands: Dict[str, str],
    proteins: Dict[str, str],
    pkd_like: bool,
) -> pd.DataFrame:
    ligand_items = list(ligands.items())
    protein_items = list(proteins.items())
    finite_mask = np.isfinite(y)
    row_indexes, col_indexes = np.where(finite_mask)
    rows: List[Dict[str, Any]] = []
    for pair_index, (drug_index, target_index) in enumerate(zip(row_indexes, col_indexes)):
        drug_source_id, smiles = ligand_items[int(drug_index)]
        target_source_id, sequence = protein_items[int(target_index)]
        affinity = float(y[int(drug_index), int(target_index)])
        row: Dict[str, Any] = {
            "pair_index": int(pair_index),
            "drug_index": int(drug_index),
            "target_index": int(target_index),
            "drug_source_id": str(drug_source_id),
            "target_source_id": str(target_source_id),
            "drug_id": stable_drug_id(str(drug_source_id), str(smiles)),
            "target_id": stable_target_id(str(target_source_id)),
            "smiles": str(smiles),
            "protein_sequence": str(sequence),
            "affinity_value": affinity,
            "split_source": "unassigned",
        }
        if affinity > 0:
            row["pKd_from_kd_nM"] = float(9.0 - math.log10(affinity))
        else:
            row["pKd_from_kd_nM"] = math.nan
        if pkd_like:
            row["label"] = int(affinity >= 5.0)
        rows.append(row)
    return pd.DataFrame(rows)


def validate_fold_indexes(train_indexes: Sequence[int], test_indexes: Sequence[int], pair_count: int) -> None:
    all_indexes = list(train_indexes) + list(test_indexes)
    if not all_indexes:
        raise ValueError("DeepDTA Davis fold files contain no pair indexes.")
    bad = [index for index in all_indexes if index < 0 or index >= pair_count]
    if bad:
        raise ValueError(f"Fold index out of range, first bad values: {bad[:10]}")
    overlap = set(train_indexes) & set(test_indexes)
    if overlap:
        raise ValueError(f"Train/test fold overlap detected, first values: {sorted(overlap)[:10]}")
    if len(set(train_indexes)) != len(train_indexes):
        raise ValueError("Duplicate pair indexes found inside train folds.")
    if len(set(test_indexes)) != len(test_indexes):
        raise ValueError("Duplicate pair indexes found inside test fold.")


def assign_splits(
    pairs: pd.DataFrame,
    train_indexes: Sequence[int],
    test_indexes: Sequence[int],
    valid_fraction: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_fold_indexes(train_indexes, test_indexes, len(pairs))
    rng = np.random.default_rng(seed)
    train_array = np.asarray(list(train_indexes), dtype=int)
    shuffled = rng.permutation(train_array)
    valid_count = int(round(len(train_array) * valid_fraction))
    valid_indexes = set(int(index) for index in shuffled[:valid_count])
    final_train_indexes = set(int(index) for index in shuffled[valid_count:])
    final_test_indexes = set(int(index) for index in test_indexes)

    assigned = pairs.copy()
    assigned["split"] = "unassigned"
    assigned.loc[assigned["pair_index"].isin(final_train_indexes), "split"] = "train"
    assigned.loc[assigned["pair_index"].isin(valid_indexes), "split"] = "valid"
    assigned.loc[assigned["pair_index"].isin(final_test_indexes), "split"] = "test"
    assigned.loc[assigned["split"] == "train", "split_source"] = "deepdta_train_setting1_remaining_after_seed7_valid_split"
    assigned.loc[assigned["split"] == "valid", "split_source"] = "deepdta_train_setting1_seed7_valid10pct"
    assigned.loc[assigned["split"] == "test", "split_source"] = "deepdta_test_setting1"
    if (assigned["split"] == "unassigned").any():
        count = int((assigned["split"] == "unassigned").sum())
        raise ValueError(f"{count} finite pairs were not assigned to train/valid/test.")
    train = assigned[assigned["split"] == "train"].copy()
    valid = assigned[assigned["split"] == "valid"].copy()
    test = assigned[assigned["split"] == "test"].copy()
    return assigned, train, valid, test


def threshold_sensitivity(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    pkd_like: bool,
    kd_nm_like: bool,
) -> pd.DataFrame:
    if pkd_like:
        basis = "affinity_value"
        direction = "affinity_value >= threshold"
        splits = {"train": train, "valid": valid, "test": test}
    elif kd_nm_like:
        basis = "pKd_from_kd_nM"
        direction = "pKd_from_kd_nM >= threshold"
        splits = {"train": train, "valid": valid, "test": test}
    else:
        basis = "unrecognized_affinity_scale"
        direction = "not_recommended"
        splits = {"train": train, "valid": valid, "test": test}

    rows: List[Dict[str, Any]] = []
    for threshold in THRESHOLDS:
        row: Dict[str, Any] = {
            "threshold": float(threshold),
            "threshold_basis": basis,
            "label_direction": direction,
        }
        total_positive = 0
        total_negative = 0
        for split, frame in splits.items():
            if basis in frame.columns:
                scores = pd.to_numeric(frame[basis], errors="coerce")
                labels = (scores >= threshold).astype(int)
                positive = int(labels.sum())
                negative = int(len(labels) - positive)
                positive_rate = float(labels.mean()) if len(labels) else math.nan
            else:
                positive = 0
                negative = 0
                positive_rate = math.nan
            row[f"{split}_positive_rate"] = positive_rate
            row[f"{split}_positive_count"] = positive
            row[f"{split}_negative_count"] = negative
            total_positive += positive
            total_negative += negative
        row["positive_count"] = total_positive
        row["negative_count"] = total_negative
        if pkd_like and threshold == 5.0:
            recommendation = "working_default_if_pKd_label_confirmed"
        elif kd_nm_like:
            recommendation = "not_final_label_raw_y_is_kd_nm_like_confirm_pKd_conversion_first"
        else:
            recommendation = "not_recommended_unrecognized_affinity_scale"
        row["recommendation"] = recommendation
        rows.append(row)
    return pd.DataFrame(rows)


def write_processed_splits(processed_dir: Path, train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "drug_id",
        "target_id",
        "smiles",
        "protein_sequence",
        "affinity_value",
        "split_source",
        "split",
        "pair_index",
        "drug_index",
        "target_index",
        "drug_source_id",
        "target_source_id",
        "pKd_from_kd_nM",
    ]
    if "label" in train.columns:
        columns.insert(5, "label")
    for split_name, frame in [("train", train), ("valid", valid), ("test", test)]:
        frame.loc[:, columns].to_csv(processed_dir / f"davis_{split_name}.csv", index=False)


def markdown_table(df: pd.DataFrame, columns: Sequence[str]) -> str:
    if df.empty:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for _, row in df[columns].iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def write_report(
    report_out: Path,
    distribution: pd.DataFrame,
    threshold_df: pd.DataFrame,
    processed_dir: Path,
    all_pairs_out: Path,
) -> None:
    report_out.parent.mkdir(parents=True, exist_ok=True)
    dist = distribution.iloc[0]
    threshold_cols = [
        "threshold",
        "threshold_basis",
        "train_positive_rate",
        "valid_positive_rate",
        "test_positive_rate",
        "positive_count",
        "negative_count",
        "recommendation",
    ]
    report = f"""# Stage 8B Davis Preparation Report

## Scope

This stage converts existing DeepDTA Davis files into processed CSV splits and
audits the affinity-value scale. It does not train, predict, create
checkpoints, or modify model structure.

## Source Check

- Y shape: `{dist['y_shape_rows']} x {dist['y_shape_cols']}`
- Ligands: `{dist['ligand_count']}`
- Proteins: `{dist['protein_count']}`
- Shape matches ligand/protein counts: `{dist['shape_matches_ligand_protein_count']}`
- Finite values: `{dist['finite_count']}`
- NaN values: `{dist['nan_count']}`
- Inf values: `{dist['inf_count']}`
- Scale note: `{dist['scale_note']}`

The observed Y range is `{dist['min']}` to `{dist['max']}` with median
`{dist['q50']}`. This does not look pKd-like when `is_pkd_like` is false.
When the scale looks like raw Kd in nM, this report computes pKd-threshold
sensitivity using `pKd_from_kd_nM = 9 - log10(Kd_nM)` but does not generate a
final binary label.

## Splits

- Train samples: `{dist['train_sample_count']}`
- Valid samples: `{dist['valid_sample_count']}`
- Test samples: `{dist['test_sample_count']}`
- Valid fraction from DeepDTA train fold: `{dist['valid_fraction_from_deepdta_train']}`
- Split seed: `{dist['split_seed']}`

Processed outputs:

- `{relative_path(processed_dir / 'davis_train.csv')}`
- `{relative_path(processed_dir / 'davis_valid.csv')}`
- `{relative_path(processed_dir / 'davis_test.csv')}`
- `{relative_path(all_pairs_out)}`

## Threshold Sensitivity

{markdown_table(threshold_df, threshold_cols)}

## Label Policy

Default final label generated: `{dist['label_generated']}`.

Because the current Davis Y values are not pKd-like, Stage 8B does not generate
a final binary `label` column. The threshold table is a sensitivity audit only.
Before model training, the project should explicitly confirm whether the
DeepDTA Davis Y values should be converted from Kd nM to pKd and then choose a
binary threshold.
"""
    report_out.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    deepdta_dir = resolve_path(args.deepdta_davis_dir)
    processed_dir = resolve_path(args.processed_dir)
    distribution_out = resolve_path(args.distribution_out)
    threshold_out = resolve_path(args.threshold_out)
    all_pairs_out = resolve_path(args.all_pairs_out)
    report_out = resolve_path(args.report_out)

    y = load_y(deepdta_dir / "Y")
    ligands = load_json(deepdta_dir / "ligands_can.txt")
    proteins = load_json(deepdta_dir / "proteins.txt")
    train_folds = load_json(deepdta_dir / "folds" / "train_fold_setting1.txt")
    test_fold = load_json(deepdta_dir / "folds" / "test_fold_setting1.txt")
    if not isinstance(ligands, dict) or not isinstance(proteins, dict):
        raise ValueError("DeepDTA Davis ligands/proteins files must be JSON objects.")
    if y.ndim != 2:
        raise ValueError(f"Expected a 2D Y matrix, found shape {y.shape}.")
    if y.shape != (len(ligands), len(proteins)):
        raise ValueError(
            f"Y shape {y.shape} does not match ligand/protein counts "
            f"({len(ligands)}, {len(proteins)})."
        )

    pkd_like, kd_nm_like, scale_note = infer_value_scale(y)
    all_pairs = build_all_pairs(y, ligands, proteins, pkd_like=pkd_like)
    train_indexes = flatten_train_folds(train_folds)
    test_indexes = [int(index) for index in test_fold]
    assigned, train, valid, test = assign_splits(
        all_pairs,
        train_indexes=train_indexes,
        test_indexes=test_indexes,
        valid_fraction=args.valid_fraction,
        seed=args.seed,
    )

    distribution = pd.DataFrame(
        [
            distribution_row(
                y,
                ligand_count=len(ligands),
                protein_count=len(proteins),
                train_count=len(train),
                valid_count=len(valid),
                test_count=len(test),
                pkd_like=pkd_like,
                kd_nm_like=kd_nm_like,
                scale_note=scale_note,
                valid_fraction=args.valid_fraction,
                seed=args.seed,
            )
        ]
    )
    threshold_df = threshold_sensitivity(train, valid, test, pkd_like=pkd_like, kd_nm_like=kd_nm_like)

    write_processed_splits(processed_dir, train, valid, test)
    all_pairs_out.parent.mkdir(parents=True, exist_ok=True)
    assigned.to_csv(all_pairs_out, index=False)
    distribution_out.parent.mkdir(parents=True, exist_ok=True)
    distribution.to_csv(distribution_out, index=False)
    threshold_df.to_csv(threshold_out, index=False)
    write_report(report_out, distribution, threshold_df, processed_dir, all_pairs_out)

    print("Stage 8B Davis preparation complete.")
    print(distribution.T)
    print("\nThreshold sensitivity:")
    print(threshold_df)
    print("\nSaved:")
    print(processed_dir / "davis_train.csv")
    print(processed_dir / "davis_valid.csv")
    print(processed_dir / "davis_test.csv")
    print(all_pairs_out)
    print(distribution_out)
    print(threshold_out)
    print(report_out)


if __name__ == "__main__":
    main()

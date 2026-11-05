#!/usr/bin/env python
"""Prepare KIBA binary processed CSV splits from DeepDTA raw files.

This is an optional Stage 11B dataset-extension step. It converts DeepDTA KIBA
setting1 folds into the project CSV contract and applies a fixed train-agnostic
binary threshold to KIBA scores. It does not train, predict, build features, or
change model structure.
"""

from __future__ import annotations

import argparse
import ast
import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KIBA_ROOT = Path("data/raw/deepdta/kiba")
DEFAULT_PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_SUMMARY_OUT = ROOT / "outputs" / "metrics" / "stage11b_kiba_binary_summary.csv"
DEFAULT_ALL_PAIRS_OUT = ROOT / "data" / "processed" / "stage11b_kiba_all_pairs.csv"
DEFAULT_REPORT_OUT = ROOT / "docs" / "stage11b_kiba_binary_preparation_report.md"
DEFAULT_THRESHOLD = 12.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare KIBA binary CSV splits.")
    parser.add_argument("--kiba-root", default=str(DEFAULT_KIBA_ROOT))
    parser.add_argument("--processed-dir", default=str(DEFAULT_PROCESSED_DIR))
    parser.add_argument("--summary-out", default=str(DEFAULT_SUMMARY_OUT))
    parser.add_argument("--all-pairs-out", default=str(DEFAULT_ALL_PAIRS_OUT))
    parser.add_argument("--report-out", default=str(DEFAULT_REPORT_OUT))
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--valid-fold-index",
        type=int,
        default=0,
        help="DeepDTA setting1 train fold used as validation.",
    )
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


def load_kiba_matrix(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"KIBA Y file not found: {path}")
    with path.open("rb") as handle:
        value = pickle.load(handle, encoding="latin1")
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D KIBA Y matrix, got shape {matrix.shape}.")
    return matrix


def load_json_object(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required KIBA JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}.")
    return value


def load_fold(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required KIBA fold file not found: {path}")
    return ast.literal_eval(path.read_text(encoding="utf-8"))


def clean_id(prefix: str, source_id: Any) -> str:
    text = str(source_id)
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return f"{prefix}_{cleaned}" if cleaned else f"{prefix}_unknown"


def flatten_kiba_pairs(
    y: np.ndarray,
    ligands: Mapping[str, Any],
    proteins: Mapping[str, Any],
    threshold: float,
) -> pd.DataFrame:
    ligand_items = list(ligands.items())
    protein_items = list(proteins.items())
    if y.shape != (len(ligand_items), len(protein_items)):
        raise ValueError(
            f"KIBA Y shape {y.shape} does not match ligand/protein counts "
            f"({len(ligand_items)}, {len(protein_items)})."
        )

    flat_y = y.ravel(order="C")
    finite_flat_indices = np.flatnonzero(np.isfinite(flat_y))
    drug_indexes, target_indexes = np.unravel_index(
        finite_flat_indices,
        y.shape,
        order="C",
    )
    rows: List[Dict[str, Any]] = []
    for pair_index, (flat_index, drug_index, target_index) in enumerate(
        zip(finite_flat_indices, drug_indexes, target_indexes)
    ):
        drug_source_id, smiles = ligand_items[int(drug_index)]
        target_source_id, sequence = protein_items[int(target_index)]
        score = float(flat_y[int(flat_index)])
        rows.append(
            {
                "drug_id": clean_id("kiba_drug", drug_source_id),
                "target_id": clean_id("kiba_target", target_source_id),
                "smiles": str(smiles),
                "protein_sequence": str(sequence),
                "label": int(score >= float(threshold)),
                "label_policy": f"kiba_score_ge_{float(threshold):g}",
                "kiba_score": score,
                "split_source": "unassigned",
                "split": "unassigned",
                "pair_index": int(pair_index),
                "finite_flat_index": int(flat_index),
                "drug_index": int(drug_index),
                "target_index": int(target_index),
                "drug_source_id": str(drug_source_id),
                "target_source_id": str(target_source_id),
            }
        )
    return pd.DataFrame(rows)


def validate_folds(
    train_folds: Sequence[Sequence[int]],
    test_fold: Sequence[int],
    pair_count: int,
) -> None:
    fold_indexes = [int(index) for fold in train_folds for index in fold]
    test_indexes = [int(index) for index in test_fold]
    all_indexes = fold_indexes + test_indexes
    if not all_indexes:
        raise ValueError("KIBA fold files contain no indexes.")
    bad = [index for index in all_indexes if index < 0 or index >= pair_count]
    if bad:
        raise ValueError(f"KIBA fold index out of range, first bad: {bad[:10]}")
    if len(set(fold_indexes)) != len(fold_indexes):
        raise ValueError("Duplicate indexes found inside KIBA train folds.")
    if len(set(test_indexes)) != len(test_indexes):
        raise ValueError("Duplicate indexes found inside KIBA test fold.")
    overlap = set(fold_indexes) & set(test_indexes)
    if overlap:
        raise ValueError(f"KIBA train/test overlap detected: {sorted(overlap)[:10]}")
    covered = set(all_indexes)
    if len(covered) != pair_count:
        raise ValueError(
            f"KIBA folds cover {len(covered)} finite pairs, expected {pair_count}."
        )


def assign_splits(
    pairs: pd.DataFrame,
    train_folds: Sequence[Sequence[int]],
    test_fold: Sequence[int],
    valid_fold_index: int,
) -> Dict[str, pd.DataFrame]:
    if valid_fold_index < 0 or valid_fold_index >= len(train_folds):
        raise ValueError("--valid-fold-index is outside the available train folds.")
    validate_folds(train_folds, test_fold, pair_count=len(pairs))

    valid_indexes = set(int(index) for index in train_folds[valid_fold_index])
    train_indexes = {
        int(index)
        for fold_index, fold in enumerate(train_folds)
        if fold_index != valid_fold_index
        for index in fold
    }
    test_indexes = set(int(index) for index in test_fold)

    assigned = pairs.copy()
    assigned.loc[assigned["pair_index"].isin(train_indexes), "split"] = "train"
    assigned.loc[assigned["pair_index"].isin(valid_indexes), "split"] = "valid"
    assigned.loc[assigned["pair_index"].isin(test_indexes), "split"] = "test"
    assigned.loc[assigned["split"] == "train", "split_source"] = (
        "deepdta_setting1_train_folds_except_valid_fold"
    )
    assigned.loc[assigned["split"] == "valid", "split_source"] = (
        f"deepdta_setting1_train_fold_{valid_fold_index}_as_valid"
    )
    assigned.loc[assigned["split"] == "test", "split_source"] = "deepdta_setting1_test"
    if (assigned["split"] == "unassigned").any():
        count = int((assigned["split"] == "unassigned").sum())
        raise ValueError(f"{count} finite pairs were not assigned.")
    return {
        "all": assigned,
        "train": assigned[assigned["split"] == "train"].copy(),
        "valid": assigned[assigned["split"] == "valid"].copy(),
        "test": assigned[assigned["split"] == "test"].copy(),
    }


def split_summary(
    split: str,
    frame: pd.DataFrame,
    threshold: float,
    valid_fold_index: int,
) -> Dict[str, Any]:
    target_counts = frame.groupby("target_id")["label"].agg(["count", "mean", "sum"])
    return {
        "dataset": "kiba",
        "split": split,
        "sample_count": int(len(frame)),
        "positive_count": int(frame["label"].sum()),
        "negative_count": int((1 - frame["label"]).sum()),
        "positive_rate": float(frame["label"].mean()),
        "drug_count": int(frame["drug_id"].nunique()),
        "target_count": int(frame["target_id"].nunique()),
        "target_count_mean": float(target_counts["count"].mean()),
        "targets_ge_10": int((target_counts["count"] >= 10).sum()),
        "targets_ge_20": int((target_counts["count"] >= 20).sum()),
        "targets_ge_50": int((target_counts["count"] >= 50).sum()),
        "threshold": float(threshold),
        "valid_fold_index": int(valid_fold_index),
        "label_policy": f"kiba_score_ge_{float(threshold):g}",
    }


def write_splits(processed_dir: Path, split_frames: Mapping[str, pd.DataFrame]) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "drug_id",
        "target_id",
        "smiles",
        "protein_sequence",
        "label",
        "label_policy",
        "kiba_score",
        "split_source",
        "split",
        "pair_index",
        "finite_flat_index",
        "drug_index",
        "target_index",
        "drug_source_id",
        "target_source_id",
    ]
    for split in ("train", "valid", "test"):
        split_frames[split].loc[:, columns].to_csv(
            processed_dir / f"kiba_{split}.csv",
            index=False,
        )


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def write_report(
    report_out: Path,
    summary: pd.DataFrame,
    processed_dir: Path,
    all_pairs_out: Path,
) -> None:
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report = "\n".join(
        [
            "# Stage 11B KIBA Binary Preparation Report",
            "",
            "## Scope",
            "",
            "This report records the optional KIBA binary processed split creation.",
            "It does not train, predict, build features, or tune thresholds against model outputs.",
            "",
            "## Outputs",
            "",
            f"- `{relative_path(processed_dir / 'kiba_train.csv')}`",
            f"- `{relative_path(processed_dir / 'kiba_valid.csv')}`",
            f"- `{relative_path(processed_dir / 'kiba_test.csv')}`",
            f"- `{relative_path(all_pairs_out)}`",
            "",
            "## Split Summary",
            "",
            markdown_table(summary),
            "",
            "## Interpretation Boundary",
            "",
            "- KIBA is an optional fourth-dataset extension.",
            "- The fixed threshold is defined before training and must not be retuned on model results.",
            "- Use KIBA as external validation context, not as a new primary benchmark claim.",
            "",
        ]
    )
    report_out.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    kiba_root = resolve_path(args.kiba_root)
    processed_dir = resolve_path(args.processed_dir)
    summary_out = resolve_path(args.summary_out)
    all_pairs_out = resolve_path(args.all_pairs_out)
    report_out = resolve_path(args.report_out)

    y = load_kiba_matrix(kiba_root / "Y")
    ligands = load_json_object(kiba_root / "ligands_can.txt")
    proteins = load_json_object(kiba_root / "proteins.txt")
    train_folds = load_fold(kiba_root / "folds" / "train_fold_setting1.txt")
    test_fold = load_fold(kiba_root / "folds" / "test_fold_setting1.txt")
    if not isinstance(train_folds, list) or not all(
        isinstance(fold, list) for fold in train_folds
    ):
        raise TypeError("train_fold_setting1.txt must contain a list of lists.")
    if not isinstance(test_fold, list):
        raise TypeError("test_fold_setting1.txt must contain a list.")

    pairs = flatten_kiba_pairs(y, ligands, proteins, threshold=float(args.threshold))
    split_frames = assign_splits(
        pairs,
        train_folds=train_folds,
        test_fold=test_fold,
        valid_fold_index=int(args.valid_fold_index),
    )
    write_splits(processed_dir, split_frames)
    all_pairs_out.parent.mkdir(parents=True, exist_ok=True)
    split_frames["all"].to_csv(all_pairs_out, index=False)

    summary = pd.DataFrame(
        [
            split_summary(split, split_frames[split], args.threshold, args.valid_fold_index)
            for split in ("train", "valid", "test")
        ]
    )
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_out, index=False)
    write_report(report_out, summary, processed_dir, all_pairs_out)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    print("Stage 11B KIBA binary preparation complete.")
    print(summary)
    print("\nSaved:")
    print(relative_path(processed_dir / "kiba_train.csv"))
    print(relative_path(processed_dir / "kiba_valid.csv"))
    print(relative_path(processed_dir / "kiba_test.csv"))
    print(relative_path(summary_out))
    print(relative_path(report_out))


if __name__ == "__main__":
    main()

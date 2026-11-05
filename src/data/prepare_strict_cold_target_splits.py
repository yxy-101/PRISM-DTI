"""Create fixed, strict three-way target-disjoint DTI splits."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[2]
REQUIRED_COLUMNS = {"drug_id", "target_id", "smiles", "protein_sequence", "label"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("biosnap", "davis", "kiba"), required=True)
    parser.add_argument("--split-seed", type=int, default=2024)
    parser.add_argument("--valid-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_fractions(args.valid_fraction, args.test_fraction)
    source, source_paths, duplicate_rows = load_sources(args.dataset)
    splits = group_split(
        source,
        valid_fraction=args.valid_fraction,
        test_fraction=args.test_fraction,
        seed=args.split_seed,
    )
    write_outputs(args.dataset, splits, source_paths, duplicate_rows, args)


def validate_fractions(valid_fraction: float, test_fraction: float) -> None:
    if valid_fraction <= 0 or test_fraction <= 0:
        raise ValueError("Validation and test fractions must be positive.")
    if valid_fraction + test_fraction >= 1:
        raise ValueError("Validation plus test fraction must be below one.")


def load_sources(dataset: str):
    paths = [
        ROOT / "data" / "processed" / ("%s_%s.csv" % (dataset, split))
        for split in ("train", "valid", "test")
    ]
    frames: List[pd.DataFrame] = []
    for split, path in zip(("train", "valid", "test"), paths):
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError("Missing columns in %s: %s" % (path, missing))
        frame = frame.copy()
        frame["source_split"] = split
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["drug_id"] = combined["drug_id"].astype(str)
    combined["target_id"] = combined["target_id"].astype(str)
    combined["label"] = pd.to_numeric(combined["label"], errors="raise").astype(int)
    if sorted(combined["label"].unique().tolist()) != [0, 1]:
        raise ValueError("Strict cold-target splitting requires binary labels 0/1.")

    pair_columns = ["drug_id", "target_id"]
    conflicting = combined.groupby(pair_columns)["label"].nunique()
    conflicting = conflicting[conflicting > 1]
    if len(conflicting):
        raise ValueError("%d drug-target pairs have conflicting labels." % len(conflicting))
    duplicate_rows = int(combined.duplicated(pair_columns, keep="first").sum())
    combined = combined.drop_duplicates(pair_columns, keep="first").copy()
    combined["_source_row"] = np.arange(len(combined))
    return combined, paths, duplicate_rows


def group_split(
    data: pd.DataFrame,
    valid_fraction: float,
    test_fraction: float,
    seed: int,
) -> Dict[str, pd.DataFrame]:
    target_count = int(data["target_id"].nunique())
    fold_count = min(10, target_count)
    if fold_count < 5:
        raise ValueError("At least five unique targets are required.")
    splitter = StratifiedGroupKFold(
        n_splits=fold_count,
        shuffle=True,
        random_state=seed,
    )
    fold_ids = np.full(len(data), -1, dtype=int)
    groups = data["target_id"].astype(str)
    for fold, (_, held_out) in enumerate(splitter.split(data, data["label"], groups)):
        fold_ids[held_out] = fold
    if (fold_ids < 0).any():
        raise RuntimeError("Some rows were not assigned to a group fold.")

    global_rate = float(data["label"].mean())

    def allocation_score(folds: Sequence[int], fraction: float) -> float:
        mask = np.isin(fold_ids, list(folds))
        return abs(float(mask.mean()) - fraction) + 2.0 * abs(
            float(data.loc[mask, "label"].mean()) - global_rate
        )

    valid_fold_count = max(1, round(valid_fraction * fold_count))
    test_fold_count = max(1, round(test_fraction * fold_count))
    candidates = range(fold_count)
    valid_folds = min(
        itertools.combinations(candidates, valid_fold_count),
        key=lambda folds: allocation_score(folds, valid_fraction),
    )
    remaining = [fold for fold in candidates if fold not in valid_folds]
    test_folds = min(
        itertools.combinations(remaining, test_fold_count),
        key=lambda folds: allocation_score(folds, test_fraction),
    )
    train_folds = [fold for fold in remaining if fold not in test_folds]
    assignments = {
        "train": train_folds,
        "valid": valid_folds,
        "test": test_folds,
    }

    splits: Dict[str, pd.DataFrame] = {}
    for split, folds in assignments.items():
        frame = data.iloc[np.flatnonzero(np.isin(fold_ids, folds))].copy()
        if sorted(frame["label"].unique().tolist()) != [0, 1]:
            raise ValueError("%s split does not contain both labels." % split)
        frame["split"] = split
        frame.sort_values("_source_row", inplace=True)
        frame.drop(columns=["_source_row"], inplace=True)
        frame.reset_index(drop=True, inplace=True)
        splits[split] = frame
    assert_disjoint(splits)
    return splits


def assert_disjoint(splits: Dict[str, pd.DataFrame]) -> None:
    names = ("train", "valid", "test")
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            left_targets = set(splits[left]["target_id"].astype(str))
            right_targets = set(splits[right]["target_id"].astype(str))
            if left_targets & right_targets:
                raise RuntimeError("Target overlap between %s and %s." % (left, right))
            left_pairs = set(zip(splits[left]["drug_id"], splits[left]["target_id"]))
            right_pairs = set(zip(splits[right]["drug_id"], splits[right]["target_id"]))
            if left_pairs & right_pairs:
                raise RuntimeError("Pair overlap between %s and %s." % (left, right))


def write_outputs(dataset, splits, source_paths, duplicate_rows, args) -> None:
    prefix = "%s_strict_cold_target" % dataset
    output_paths = {
        split: ROOT / "data" / "processed" / ("%s_%s.csv" % (prefix, split))
        for split in ("train", "valid", "test")
    }
    summary_path = ROOT / "outputs" / "metrics" / (prefix + "_split_summary.csv")
    manifest_path = ROOT / "outputs" / "metrics" / (prefix + "_manifest.json")
    occupied = [
        path
        for path in (*output_paths.values(), summary_path, manifest_path)
        if path.exists()
    ]
    if occupied:
        raise FileExistsError("Refusing to overwrite strict cold-target artifacts: %s" % occupied)

    rows = []
    for split, frame in splits.items():
        path = output_paths[split]
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "sample_count": len(frame),
                "positive_rate": float(frame["label"].mean()),
                "unique_drugs": int(frame["drug_id"].nunique()),
                "unique_targets": int(frame["target_id"].nunique()),
                "duplicate_pairs": int(frame.duplicated(["drug_id", "target_id"]).sum()),
                "csv": relative(path),
                "sha256": sha256(path),
            }
        )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    manifest = {
        "dataset": dataset,
        "cold_kind": "strict_target",
        "held_out_entity": "target_id",
        "split_seed": int(args.split_seed),
        "valid_fraction": float(args.valid_fraction),
        "test_fraction": float(args.test_fraction),
        "assignment_uses_labels_for_stratification": True,
        "assignment_uses_model_outputs": False,
        "duplicate_pair_rows_removed": duplicate_rows,
        "source_files": [relative(path) for path in source_paths],
        "outputs": {split: relative(path) for split, path in output_paths.items()},
        "overlap_audit": {
            "train_valid_target_overlap": 0,
            "train_test_target_overlap": 0,
            "valid_test_target_overlap": 0,
            "all_pair_overlaps": 0,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print("saved:", summary_path, manifest_path)


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Normalize TAPB raw CSV splits into the PRISM-DTI processed schema."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Optional


ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "data" / "raw"
PROCESSED_ROOT = ROOT / "data" / "processed"

DATASETS = ("biosnap", "bindingdb", "davis")
SPLITS = {
    "train_with_id.csv": "train",
    "val_with_id.csv": "valid",
    "test_with_id.csv": "test",
}
REQUIRED_COLUMNS = ("SMILES", "Protein", "Y", "pr_id")
OUTPUT_COLUMNS = (
    "drug_id",
    "target_id",
    "smiles",
    "protein_sequence",
    "label",
    "split",
)


def parse_args() -> argparse.Namespace:
    """Parse dataset preparation arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare TAPB raw splits for PRISM-DTI."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASETS + ("all",),
        help="Dataset to prepare: biosnap, bindingdb, davis, or all.",
    )
    return parser.parse_args()


def main() -> None:
    """Prepare one dataset or all supported TAPB datasets."""
    args = parse_args()
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        prepare_dataset(dataset)


def prepare_dataset(dataset: str) -> None:
    """Prepare train/valid/test CSV files for one dataset."""
    for filename, split in SPLITS.items():
        input_path = RAW_ROOT / dataset / "random" / filename
        output_path = PROCESSED_ROOT / f"{dataset}_{split}.csv"
        stats = write_processed_split(
            input_path=input_path,
            output_path=output_path,
            split=split,
        )
        print_stats(stats)


def write_processed_split(input_path: Path, output_path: Path, split: str) -> Dict:
    """Write one processed split and return a compact stats dictionary."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input split not found: {input_path}")

    label_counts: Counter = Counter()
    unique_drugs = set()
    unique_targets = set()
    row_count = 0

    with input_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        validate_columns(input_path=input_path, columns=reader.fieldnames)

        with output_path.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=list(OUTPUT_COLUMNS))
            writer.writeheader()

            for raw_row in reader:
                processed_row = normalize_row(raw_row=raw_row, split=split)
                writer.writerow(processed_row)

                row_count += 1
                label_counts[processed_row["label"]] += 1
                unique_drugs.add(processed_row["drug_id"])
                unique_targets.add(processed_row["target_id"])

    return {
        "output_path": output_path,
        "shape": (row_count, len(OUTPUT_COLUMNS)),
        "columns": list(OUTPUT_COLUMNS),
        "label_distribution": dict(sorted(label_counts.items())),
        "unique_drug_id": len(unique_drugs),
        "unique_target_id": len(unique_targets),
    }


def validate_columns(input_path: Path, columns: Optional[Iterable[str]]) -> None:
    """Raise a clear error when a raw TAPB split misses required columns."""
    column_set = set(columns or [])
    missing_columns = [name for name in REQUIRED_COLUMNS if name not in column_set]
    if missing_columns:
        raise ValueError(
            "Missing required columns in %s: %s"
            % (input_path, ", ".join(missing_columns))
        )


def normalize_row(raw_row: Dict[str, str], split: str) -> Dict[str, str]:
    """Map one TAPB row into the canonical PRISM-DTI CSV schema."""
    smiles = raw_row["SMILES"].strip()
    protein = raw_row["Protein"].strip()
    label = normalize_label(raw_row["Y"])
    target_id = raw_row["pr_id"].strip()

    return {
        "drug_id": stable_drug_id(smiles),
        "target_id": target_id,
        "smiles": smiles,
        "protein_sequence": protein,
        "label": label,
        "split": split,
    }


def stable_drug_id(smiles: str) -> str:
    """Create a stable compact drug ID from the canonical input SMILES string."""
    digest = hashlib.md5(smiles.encode("utf-8")).hexdigest()
    return "drug_%s" % digest[:12]


def normalize_label(value: str) -> str:
    """Normalize binary labels like 1.0/0.0 to 1/0 while preserving other values."""
    label = value.strip()
    try:
        numeric = float(label)
    except ValueError:
        return label
    if numeric.is_integer():
        return str(int(numeric))
    return label


def print_stats(stats: Dict) -> None:
    """Print the requested per-file preparation summary."""
    output_path = stats["output_path"]
    try:
        display_path = output_path.relative_to(ROOT)
    except ValueError:
        display_path = output_path

    print(f"output path: {display_path}")
    print(f"shape: {stats['shape']}")
    print(f"columns: {stats['columns']}")
    print(f"label distribution: {stats['label_distribution']}")
    print(f"number of unique drug_id: {stats['unique_drug_id']}")
    print(f"number of unique target_id: {stats['unique_target_id']}")
    print()


if __name__ == "__main__":
    main()

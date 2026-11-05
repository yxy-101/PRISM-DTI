#!/usr/bin/env python
"""Prepare Stage 8C Davis binary labels from converted DeepDTA Kd values.

This script updates the Stage 8B Davis processed CSVs with explicit binary
label columns. It does not train, predict, create checkpoints, or modify model
structure.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_SUMMARY_OUT = (
    ROOT / "outputs" / "metrics" / "stage8c_davis_binary_label_summary.csv"
)
LABEL_POLICY = "pKd_from_Kd_nM_ge_6.0"
SPLITS = ("train", "valid", "test")
QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add Stage 8C Davis pKd-based binary labels."
    )
    parser.add_argument("--processed-dir", default=str(DEFAULT_PROCESSED_DIR))
    parser.add_argument("--summary-out", default=str(DEFAULT_SUMMARY_OUT))
    parser.add_argument("--primary-threshold", type=float, default=6.0)
    parser.add_argument("--sensitivity-threshold", type=float, default=7.0)
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


def split_path(processed_dir: Path, split: str) -> Path:
    return processed_dir / ("davis_%s.csv" % split)


def backup_path(processed_dir: Path, split: str) -> Path:
    return processed_dir / ("stage8b_backup_davis_%s.csv" % split)


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError("Davis split not found: %s" % path)
    df = pd.read_csv(path)
    required = ["drug_id", "target_id", "smiles", "protein_sequence", "affinity_value"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError("Missing required columns in %s: %s" % (path, missing))
    return df


def ensure_backup(source: Path, backup: Path) -> bool:
    """Create a Stage 8B backup once; return True when created now."""
    if backup.exists():
        return False
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, backup)
    return True


def prepare_labels(
    df: pd.DataFrame,
    primary_threshold: float,
    sensitivity_threshold: float,
) -> pd.DataFrame:
    frame = df.copy()
    affinity = pd.to_numeric(frame["affinity_value"], errors="coerce")
    finite = np.isfinite(affinity.to_numpy(dtype=float))
    if not bool(finite.all()):
        bad_count = int((~finite).sum())
        raise ValueError("affinity_value contains non-finite values: %d" % bad_count)
    if not bool((affinity > 0).all()):
        bad_count = int((affinity <= 0).sum())
        raise ValueError("affinity_value must be positive for Kd nM conversion: %d" % bad_count)

    if "pKd_from_kd_nM" not in frame.columns:
        frame["pKd_from_kd_nM"] = 9.0 - np.log10(affinity.astype(float))
    else:
        pkd = pd.to_numeric(frame["pKd_from_kd_nM"], errors="coerce")
        missing = ~np.isfinite(pkd.to_numpy(dtype=float))
        if bool(missing.any()):
            pkd = pkd.astype(float)
            pkd[missing] = 9.0 - np.log10(affinity.astype(float)[missing])
            frame["pKd_from_kd_nM"] = pkd

    pkd_values = pd.to_numeric(frame["pKd_from_kd_nM"], errors="coerce")
    if not bool(np.isfinite(pkd_values.to_numpy(dtype=float)).all()):
        raise ValueError("pKd_from_kd_nM contains non-finite values after conversion.")

    frame["label_pkd6"] = (pkd_values >= float(primary_threshold)).astype(int)
    frame["label_pkd7"] = (pkd_values >= float(sensitivity_threshold)).astype(int)
    frame["label"] = frame["label_pkd6"].astype(int)
    frame["label_policy"] = LABEL_POLICY
    return frame


def split_summary(
    split: str,
    df: pd.DataFrame,
    backup_created: bool,
    backup: Path,
    primary_threshold: float,
    sensitivity_threshold: float,
) -> Dict[str, Any]:
    pkd = pd.to_numeric(df["pKd_from_kd_nM"], errors="coerce").astype(float)
    label6 = pd.to_numeric(df["label_pkd6"], errors="coerce").astype(float)
    label7 = pd.to_numeric(df["label_pkd7"], errors="coerce").astype(float)
    quantile_values = pkd.quantile(list(QUANTILES))
    row: Dict[str, Any] = {
        "dataset": "davis",
        "split": split,
        "sample_count": int(len(df)),
        "primary_threshold": float(primary_threshold),
        "sensitivity_threshold": float(sensitivity_threshold),
        "label_policy": LABEL_POLICY,
        "positive_count_label_pkd6": int(label6.sum()),
        "positive_rate_label_pkd6": float(label6.mean()),
        "positive_count_label_pkd7": int(label7.sum()),
        "positive_rate_label_pkd7": float(label7.mean()),
        "pKd_min": float(pkd.min()),
        "pKd_max": float(pkd.max()),
        "pKd_mean": float(pkd.mean()),
        "pKd_std": float(pkd.std(ddof=1)),
        "backup_path": relative_path(backup),
        "backup_created_now": bool(backup_created),
        "note": (
            "pKd>=6.0 is a working primary threshold; pKd>=7.0 is retained as "
            "a sensitivity label and is not used for training in Stage 8C."
        ),
    }
    for quantile, value in quantile_values.items():
        key = "pKd_q%s" % str(quantile).replace(".", "_")
        row[key] = float(value)
    return row


def write_summary(rows: Sequence[Dict[str, Any]], summary_out: Path) -> pd.DataFrame:
    summary = pd.DataFrame(list(rows))
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_out, index=False)
    return summary


def main() -> None:
    args = parse_args()
    processed_dir = resolve_path(args.processed_dir)
    summary_out = resolve_path(args.summary_out)

    rows: List[Dict[str, Any]] = []
    for split in SPLITS:
        path = split_path(processed_dir, split)
        backup = backup_path(processed_dir, split)
        df = load_split(path)
        backup_created = ensure_backup(path, backup)
        labeled = prepare_labels(
            df,
            primary_threshold=float(args.primary_threshold),
            sensitivity_threshold=float(args.sensitivity_threshold),
        )
        labeled.to_csv(path, index=False)
        rows.append(
            split_summary(
                split=split,
                df=labeled,
                backup_created=backup_created,
                backup=backup,
                primary_threshold=float(args.primary_threshold),
                sensitivity_threshold=float(args.sensitivity_threshold),
            )
        )

    summary = write_summary(rows, summary_out)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    print("Stage 8C Davis binary label summary:")
    print(summary)
    print("\nSaved:")
    print(relative_path(summary_out))


if __name__ == "__main__":
    main()

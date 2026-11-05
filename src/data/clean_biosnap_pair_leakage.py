#!/usr/bin/env python
"""Audit and remove exact BioSNAP pair leakage from the training split.

Validation and test rows are preserved. Exact (drug_id, target_id) pairs that
also occur in validation or test are removed from train, and duplicate train
pairs are collapsed after label-consistency checks. Applying the cleanup first
backs up all three original processed CSV files.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS = {
    "train": ROOT / "data" / "processed" / "biosnap_train.csv",
    "valid": ROOT / "data" / "processed" / "biosnap_valid.csv",
    "test": ROOT / "data" / "processed" / "biosnap_test.csv",
}
DEFAULT_BACKUP_DIR = (
    ROOT / "data" / "processed" / "backup_before_biosnap_pair_cleanup"
)
DEFAULT_REPORT = (
    ROOT / "outputs" / "metrics" / "biosnap_pair_leakage_cleanup_audit.csv"
)
PAIR_COLUMNS = ("drug_id", "target_id")
REQUIRED_COLUMNS = ("drug_id", "target_id", "label")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit BioSNAP exact-pair leakage; use --apply to back up and clean "
            "the processed training CSV while preserving valid/test."
        )
    )
    parser.add_argument("--train-csv", default=str(DEFAULT_SPLITS["train"]))
    parser.add_argument("--valid-csv", default=str(DEFAULT_SPLITS["valid"]))
    parser.add_argument("--test-csv", default=str(DEFAULT_SPLITS["test"]))
    parser.add_argument("--backup-dir", default=str(DEFAULT_BACKUP_DIR))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_split(path: Path, split: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError("BioSNAP %s CSV not found: %s" % (split, path))
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError("%s is missing required columns: %s" % (path, missing))
    frame["drug_id"] = frame["drug_id"].astype(str)
    frame["target_id"] = frame["target_id"].astype(str)
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    return frame


def pair_index(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(frame[list(PAIR_COLUMNS)])


def pair_set(frame: pd.DataFrame) -> set:
    return set(map(tuple, frame[list(PAIR_COLUMNS)].itertuples(index=False, name=None)))


def assert_label_consistency(frame: pd.DataFrame, source: str) -> None:
    conflicts = (
        frame.groupby(list(PAIR_COLUMNS), dropna=False)["label"].nunique().gt(1)
    )
    if bool(conflicts.any()):
        examples = conflicts[conflicts].head(10).index.tolist()
        raise ValueError(
            "%s contains pairs with conflicting labels: %s" % (source, examples)
        )


def overlap_detail(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> pd.DataFrame:
    return left.merge(
        right,
        on=list(PAIR_COLUMNS),
        suffixes=("_left", "_right"),
    )


def build_clean_train(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    assert_label_consistency(train, "train")
    assert_label_consistency(valid, "valid")
    assert_label_consistency(test, "test")

    valid_test_overlap = pair_set(valid).intersection(pair_set(test))
    if valid_test_overlap:
        raise ValueError(
            "valid/test exact-pair overlap must be resolved manually: %s"
            % sorted(valid_test_overlap)[:10]
        )
    valid_duplicates = int(valid.duplicated(list(PAIR_COLUMNS)).sum())
    test_duplicates = int(test.duplicated(list(PAIR_COLUMNS)).sum())
    if valid_duplicates or test_duplicates:
        raise ValueError(
            "valid/test duplicates are preserved by policy and must be resolved "
            "manually; valid=%d test=%d" % (valid_duplicates, test_duplicates)
        )

    evaluation_pairs = pair_set(valid).union(pair_set(test))
    train_pairs = pair_index(train)
    leakage_mask = train_pairs.isin(
        pd.MultiIndex.from_tuples(
            sorted(evaluation_pairs),
            names=list(PAIR_COLUMNS),
        )
    )
    without_leakage = train.loc[~leakage_mask].copy()
    cleaned = without_leakage.drop_duplicates(
        subset=list(PAIR_COLUMNS),
        keep="first",
    ).reset_index(drop=True)
    if "split" in cleaned.columns:
        cleaned["split"] = "train"

    counts = {
        "train_rows_before": int(len(train)),
        "train_rows_after": int(len(cleaned)),
        "cross_split_train_rows_removed": int(leakage_mask.sum()),
        "cross_split_unique_pairs_removed": int(
            train.loc[leakage_mask, list(PAIR_COLUMNS)].drop_duplicates().shape[0]
        ),
        "within_train_duplicate_rows_removed": int(
            len(without_leakage) - len(cleaned)
        ),
        "valid_rows_preserved": int(len(valid)),
        "test_rows_preserved": int(len(test)),
    }
    return cleaned, counts


def audit_rows(
    splits: Dict[str, pd.DataFrame],
    stage: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for split, frame in splits.items():
        rows.append(
            {
                "stage": stage,
                "check": "split_statistics",
                "left_split": split,
                "right_split": "",
                "row_count": int(len(frame)),
                "unique_pair_count": int(
                    frame[list(PAIR_COLUMNS)].drop_duplicates().shape[0]
                ),
                "overlap_count": "",
                "duplicate_extra_row_count": int(
                    frame.duplicated(list(PAIR_COLUMNS)).sum()
                ),
                "label_conflict_count": int(
                    frame.groupby(list(PAIR_COLUMNS))["label"]
                    .nunique()
                    .gt(1)
                    .sum()
                ),
                "status": "observed",
            }
        )
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        detail = overlap_detail(splits[left], splits[right])
        rows.append(
            {
                "stage": stage,
                "check": "cross_split_exact_pair_overlap",
                "left_split": left,
                "right_split": right,
                "row_count": "",
                "unique_pair_count": "",
                "overlap_count": int(
                    detail[list(PAIR_COLUMNS)].drop_duplicates().shape[0]
                ),
                "duplicate_extra_row_count": "",
                "label_conflict_count": int(
                    (detail["label_left"] != detail["label_right"]).sum()
                ),
                "status": "pass" if detail.empty else "fail",
            }
        )
    return rows


def write_frame_atomically(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".pair_cleanup_tmp")
    frame.to_csv(temporary, index=False)
    try:
        temporary.replace(path)
    except PermissionError:
        # Some managed Windows workspaces allow file writes but reject replace().
        shutil.copyfile(temporary, path)
        try:
            temporary.unlink()
        except PermissionError:
            print("warning: could not remove temporary file:", temporary)


def ensure_backup(
    paths: Dict[str, Path],
    backup_dir: Path,
) -> None:
    if backup_dir.exists() and any(backup_dir.iterdir()):
        for path in paths.values():
            backup_path = backup_dir / path.name
            if not backup_path.exists():
                raise FileExistsError(
                    "Backup directory is incomplete: %s" % backup_dir
                )
            if sha256_file(backup_path) != sha256_file(path):
                raise FileExistsError(
                    "Existing backup differs from the current source: %s"
                    % backup_path
                )
        return
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in paths.values():
        shutil.copy2(path, backup_dir / path.name)


def print_overlap_examples(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    for right_name, right in (("valid", valid), ("test", test)):
        detail = overlap_detail(train, right)
        print(
            "train-%s overlap unique pairs: %d"
            % (
                right_name,
                detail[list(PAIR_COLUMNS)].drop_duplicates().shape[0],
            )
        )
        if not detail.empty:
            columns = list(PAIR_COLUMNS) + ["label_left", "label_right"]
            print(detail[columns].drop_duplicates().to_string(index=False))


def main() -> None:
    args = parse_args()
    paths = {
        "train": resolve_path(args.train_csv),
        "valid": resolve_path(args.valid_csv),
        "test": resolve_path(args.test_csv),
    }
    backup_dir = resolve_path(args.backup_dir)
    report_path = resolve_path(args.report)
    splits = {
        split: load_split(path, split)
        for split, path in paths.items()
    }
    print_overlap_examples(splits["train"], splits["valid"], splits["test"])
    clean_train, counts = build_clean_train(
        splits["train"],
        splits["valid"],
        splits["test"],
    )
    clean_splits = {
        "train": clean_train,
        "valid": splits["valid"],
        "test": splits["test"],
    }
    report_rows = audit_rows(splits, "before") + audit_rows(clean_splits, "after")

    changes_required = bool(
        counts["cross_split_train_rows_removed"]
        or counts["within_train_duplicate_rows_removed"]
    )
    if args.apply and changes_required:
        ensure_backup(paths, backup_dir)
        before_hashes = {split: sha256_file(path) for split, path in paths.items()}
        write_frame_atomically(clean_train, paths["train"])
        after_hashes = {split: sha256_file(path) for split, path in paths.items()}
        if before_hashes["valid"] != after_hashes["valid"]:
            raise RuntimeError("Validation CSV changed unexpectedly")
        if before_hashes["test"] != after_hashes["test"]:
            raise RuntimeError("Test CSV changed unexpectedly")
        mode = "applied"
    elif args.apply:
        before_hashes = {split: sha256_file(path) for split, path in paths.items()}
        after_hashes = dict(before_hashes)
        mode = "already_clean"
    else:
        before_hashes = {split: sha256_file(path) for split, path in paths.items()}
        after_hashes = dict(before_hashes)
        mode = "audit_only"

    for split in ("train", "valid", "test"):
        report_rows.append(
            {
                "stage": "file_integrity",
                "check": "sha256",
                "left_split": split,
                "right_split": "",
                "row_count": "",
                "unique_pair_count": "",
                "overlap_count": "",
                "duplicate_extra_row_count": "",
                "label_conflict_count": "",
                "status": (
                    "changed_as_expected"
                    if split == "train"
                    and args.apply
                    and before_hashes[split] != after_hashes[split]
                    else "unchanged"
                ),
                "sha256_before": before_hashes[split],
                "sha256_after": after_hashes[split],
            }
        )
    report = pd.DataFrame(report_rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)

    print("mode:", mode)
    for key, value in counts.items():
        print("%s: %s" % (key, value))
    print("report:", report_path)
    if args.apply:
        print("backup:", backup_dir)
        print(
            "BioSNAP checkpoints and predictions created from the previous "
            "training CSV must be treated as pre-cleanup artifacts."
        )


if __name__ == "__main__":
    main()

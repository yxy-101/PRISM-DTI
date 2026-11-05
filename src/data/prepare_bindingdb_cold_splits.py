"""Create deterministic entity- or scaffold-disjoint BindingDB splits."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:  # Optional until --kind scaffold is requested.
    Chem = None
    MurckoScaffold = None


ROOT = Path(__file__).resolve().parents[2]
SOURCE_SPLITS = (
    ROOT / "data" / "processed" / "bindingdb_train.csv",
    ROOT / "data" / "processed" / "bindingdb_valid.csv",
    ROOT / "data" / "processed" / "bindingdb_test.csv",
)
REQUIRED_COLUMNS = (
    "drug_id",
    "target_id",
    "smiles",
    "protein_sequence",
    "label",
    "split",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build fixed BindingDB cold-drug/cold-target/scaffold-cold splits. "
            "Assignment uses group-stratified labels and a prespecified seed, "
            "never model scores."
        )
    )
    parser.add_argument(
        "--kind",
        choices=("all", "drug", "target", "scaffold"),
        default="all",
        help=(
            "all preserves the established drug+target behavior; request "
            "scaffold explicitly to create the harder Murcko split."
        ),
    )
    parser.add_argument("--split-seed", type=int, default=2024)
    parser.add_argument("--valid-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_fractions(args.valid_fraction, args.test_fraction)
    source = load_sources()
    kinds = ("drug", "target") if args.kind == "all" else (args.kind,)
    for kind in kinds:
        current = attach_scaffold_groups(source) if kind == "scaffold" else source
        entity_column = {
            "drug": "drug_id",
            "target": "target_id",
            "scaffold": "scaffold_id",
        }[kind]
        splits = group_split(
            current,
            entity_column=entity_column,
            valid_fraction=args.valid_fraction,
            test_fraction=args.test_fraction,
            seed=args.split_seed,
        )
        write_outputs(kind, splits, args.split_seed)


def validate_fractions(valid_fraction: float, test_fraction: float) -> None:
    if valid_fraction <= 0 or test_fraction <= 0:
        raise ValueError("Validation and test fractions must be positive.")
    if valid_fraction + test_fraction >= 1:
        raise ValueError("Validation plus test fraction must be below one.")


def load_sources() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in SOURCE_SPLITS:
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError("Missing columns in %s: %s" % (path, missing))
        frames.append(frame.loc[:, REQUIRED_COLUMNS])
    combined = pd.concat(frames, ignore_index=True)
    combined["drug_id"] = combined["drug_id"].astype(str)
    combined["target_id"] = combined["target_id"].astype(str)
    combined["label"] = pd.to_numeric(combined["label"], errors="raise").astype(int)
    if sorted(combined["label"].unique().tolist()) != [0, 1]:
        raise ValueError("BindingDB cold-start requires binary labels 0/1.")
    duplicate_pairs = int(combined.duplicated(["drug_id", "target_id"]).sum())
    if duplicate_pairs:
        raise ValueError(
            "Source splits contain %d duplicate drug-target pairs." % duplicate_pairs
        )
    combined["_source_row"] = range(len(combined))
    return combined


def attach_scaffold_groups(data: pd.DataFrame) -> pd.DataFrame:
    """Attach one canonical Bemis-Murcko group to every unique drug."""
    if Chem is None or MurckoScaffold is None:
        raise ImportError(
            "--kind scaffold requires RDKit. Install rdkit or run this command "
            "inside the project's chemistry environment."
        )
    drug_smiles = data.loc[:, ["drug_id", "smiles"]].drop_duplicates()
    ambiguous = drug_smiles.groupby("drug_id")["smiles"].nunique()
    ambiguous = ambiguous[ambiguous > 1]
    if len(ambiguous):
        raise ValueError(
            "%d drug IDs map to multiple SMILES; scaffold assignment is unsafe."
            % len(ambiguous)
        )

    records = []
    for row in drug_smiles.itertuples(index=False):
        mol = Chem.MolFromSmiles(str(row.smiles))
        if mol is None:
            raise ValueError("Invalid SMILES for drug_id=%s" % row.drug_id)
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol,
            includeChirality=False,
        )
        if scaffold:
            group_key = "murcko:%s" % scaffold
            scaffold_type = "murcko"
        else:
            # Acyclic molecules have no Murcko framework. Keep each canonical
            # molecule as its own group instead of merging all acyclic drugs.
            group_key = "acyclic:%s" % canonical
            scaffold_type = "acyclic_singleton"
        records.append(
            {
                "drug_id": str(row.drug_id),
                "scaffold_id": "scaffold_"
                + hashlib.sha1(group_key.encode("utf-8")).hexdigest()[:16],
                "scaffold_smiles": scaffold,
                "scaffold_type": scaffold_type,
            }
        )

    scaffold_map = pd.DataFrame(records)
    result = data.merge(scaffold_map, on="drug_id", how="left", validate="many_to_one")
    if result["scaffold_id"].isna().any():
        raise RuntimeError("Scaffold assignment did not cover every BindingDB row.")
    return result


def group_split(
    data: pd.DataFrame,
    entity_column: str,
    valid_fraction: float,
    test_fraction: float,
    seed: int,
) -> Dict[str, pd.DataFrame]:
    fold_count = 10
    splitter = StratifiedGroupKFold(
        n_splits=fold_count,
        shuffle=True,
        random_state=seed,
    )
    fold_ids = np.full(len(data), -1, dtype=int)
    groups = data[entity_column].astype(str)
    for fold, (_, held_out_index) in enumerate(
        splitter.split(data, data["label"], groups=groups)
    ):
        fold_ids[held_out_index] = fold
    if int((fold_ids < 0).sum()):
        raise RuntimeError("Some rows were not assigned to a stratified group fold.")

    global_rate = float(data["label"].mean())

    def allocation_score(folds: Sequence[int], desired_fraction: float) -> float:
        mask = np.isin(fold_ids, list(folds))
        sample_fraction = float(mask.mean())
        positive_rate = float(data.loc[mask, "label"].mean())
        return abs(sample_fraction - desired_fraction) + 2.0 * abs(
            positive_rate - global_rate
        )

    valid_fold = min(
        range(fold_count),
        key=lambda fold: allocation_score((fold,), valid_fraction),
    )
    remaining = [fold for fold in range(fold_count) if fold != valid_fold]
    test_fold_count = max(1, round(test_fraction * fold_count))
    test_folds = min(
        itertools.combinations(remaining, test_fold_count),
        key=lambda folds: allocation_score(folds, test_fraction),
    )
    train_folds = [fold for fold in remaining if fold not in test_folds]

    train = data.iloc[np.flatnonzero(np.isin(fold_ids, train_folds))].copy()
    valid = data.iloc[np.flatnonzero(fold_ids == valid_fold)].copy()
    test = data.iloc[np.flatnonzero(np.isin(fold_ids, test_folds))].copy()
    splits = {"train": train, "valid": valid, "test": test}

    for split, frame in splits.items():
        if sorted(frame["label"].unique().tolist()) != [0, 1]:
            raise ValueError("%s split does not contain both labels." % split)
        frame["split"] = split
        frame.sort_values("_source_row", inplace=True)
        frame.drop(columns=["_source_row"], inplace=True)
        frame.reset_index(drop=True, inplace=True)
    assert_disjoint(splits, entity_column)
    return splits


def assert_disjoint(splits: Dict[str, pd.DataFrame], entity_column: str) -> None:
    names = ("train", "valid", "test")
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            left_entities = set(splits[left][entity_column].astype(str))
            right_entities = set(splits[right][entity_column].astype(str))
            overlap = left_entities & right_entities
            if overlap:
                raise RuntimeError(
                    "%s overlap between %s and %s: %d"
                    % (entity_column, left, right, len(overlap))
                )
            left_pairs = set(
                zip(splits[left]["drug_id"], splits[left]["target_id"])
            )
            right_pairs = set(
                zip(splits[right]["drug_id"], splits[right]["target_id"])
            )
            if left_pairs & right_pairs:
                raise RuntimeError("Pair overlap between %s and %s." % (left, right))
            if entity_column == "scaffold_id":
                left_drugs = set(splits[left]["drug_id"].astype(str))
                right_drugs = set(splits[right]["drug_id"].astype(str))
                if left_drugs & right_drugs:
                    raise RuntimeError(
                        "Drug overlap between scaffold splits %s and %s." % (left, right)
                    )


def write_outputs(
    kind: str, splits: Dict[str, pd.DataFrame], split_seed: int
) -> None:
    prefix = "bindingdb_cold_%s" % kind
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
        raise FileExistsError("Refusing to overwrite cold split artifacts: %s" % occupied)

    rows = []
    for split, frame in splits.items():
        path = output_paths[split]
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        rows.append(
            {
                "kind": kind,
                "split": split,
                "sample_count": len(frame),
                "positive_rate": float(frame["label"].mean()),
                "unique_drugs": int(frame["drug_id"].nunique()),
                "unique_targets": int(frame["target_id"].nunique()),
                "unique_scaffolds": (
                    int(frame["scaffold_id"].nunique())
                    if "scaffold_id" in frame
                    else None
                ),
                "acyclic_drugs": (
                    int(
                        frame.loc[
                            frame["scaffold_type"] == "acyclic_singleton", "drug_id"
                        ].nunique()
                    )
                    if "scaffold_type" in frame
                    else None
                ),
                "duplicate_pairs": int(
                    frame.duplicated(["drug_id", "target_id"]).sum()
                ),
                "csv": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(path),
            }
        )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    manifest = {
        "dataset": "bindingdb",
        "cold_kind": kind,
        "held_out_entity": {
            "drug": "drug_id",
            "target": "target_id",
            "scaffold": "scaffold_id",
        }[kind],
        "scaffold_definition": (
            "RDKit Bemis-Murcko; acyclic molecules grouped by canonical SMILES"
            if kind == "scaffold"
            else None
        ),
        "split_seed": int(split_seed),
        "assignment_uses_labels_for_stratification": True,
        "assignment_uses_model_outputs": False,
        "stratification": "StratifiedGroupKFold with deterministic fold allocation",
        "source_files": [str(path.relative_to(ROOT)).replace("\\", "/") for path in SOURCE_SPLITS],
        "outputs": {
            split: str(path.relative_to(ROOT)).replace("\\", "/")
            for split, path in output_paths.items()
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print("saved:", summary_path, manifest_path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()

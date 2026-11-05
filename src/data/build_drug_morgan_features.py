#!/usr/bin/env python
"""Build drug-id keyed Morgan fingerprint feature dictionaries."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    """Parse Morgan feature build options."""
    parser = argparse.ArgumentParser(
        description="Build RDKit Morgan fingerprint features for processed DTI CSVs"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("bindingdb", "biosnap", "davis", "kiba"),
        help="Dataset name used in data/processed/{dataset}_{split}.csv.",
    )
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--out_dir", default="data/features/drug")
    parser.add_argument("--n_bits", type=int, default=2048)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument(
        "--as_sequence",
        nargs="?",
        const=True,
        default=True,
        type=parse_bool,
        help="Save each fingerprint as [1, n_bits]. Use false for [n_bits].",
    )
    parser.add_argument(
        "--no-as_sequence",
        "--no-as-sequence",
        dest="as_sequence",
        action="store_false",
        help="Save each fingerprint as a flat [n_bits] vector.",
    )
    return parser.parse_args()


def main() -> None:
    """Build and save Morgan fingerprints."""
    args = parse_args()
    Chem, AllChem = load_rdkit()
    csv_paths = dataset_csv_paths(args.dataset, args.processed_dir)
    drug_smiles = load_drug_smiles(csv_paths)
    output_dir = resolve_path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / ("%s_drug_morgan%d.pt" % (args.dataset, args.n_bits))
    failed_path = output_dir / (
        "%s_drug_morgan%d_failed.csv" % (args.dataset, args.n_bits)
    )

    features: Dict[str, torch.Tensor] = {}
    failures: List[Tuple[str, str, str]] = []
    for drug_id, smiles in sorted(drug_smiles.items()):
        try:
            features[drug_id] = morgan_feature(
                smiles=smiles,
                Chem=Chem,
                AllChem=AllChem,
                n_bits=int(args.n_bits),
                radius=int(args.radius),
                as_sequence=bool(args.as_sequence),
            )
        except ValueError as exc:
            failures.append((drug_id, smiles, str(exc)))

    torch.save(features, output_path)
    write_failed_smiles(failed_path, failures)

    print("unique_drugs: %d" % len(drug_smiles))
    print("saved: %d" % len(features))
    print("failed: %d" % len(failures))
    print("output_path: %s" % relative_path(output_path))
    print("failed_path: %s" % relative_path(failed_path))


def load_rdkit():
    """Import RDKit lazily and fail with a clear setup message."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise SystemExit(
            "RDKit is required to build Morgan fingerprints. "
            "Install rdkit in the active environment before running this script."
        ) from exc
    return Chem, AllChem


def dataset_csv_paths(dataset: str, processed_dir: str) -> List[Path]:
    """Return train/valid/test processed CSV paths."""
    base_dir = resolve_path(processed_dir)
    paths = [
        base_dir / ("%s_train.csv" % dataset),
        base_dir / ("%s_valid.csv" % dataset),
        base_dir / ("%s_test.csv" % dataset),
    ]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError("Processed CSV not found: %s" % path)
    return paths


def load_drug_smiles(csv_paths: Iterable[Path]) -> Dict[str, str]:
    """Collect one SMILES string per drug ID across processed splits."""
    required = {"drug_id", "smiles"}
    drug_smiles: Dict[str, str] = {}
    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            missing = [column for column in sorted(required) if column not in columns]
            if missing:
                raise ValueError(
                    "Missing required columns in %s: %s"
                    % (csv_path, ", ".join(missing))
                )
            for row in reader:
                drug_id = str(row["drug_id"]).strip()
                smiles = str(row["smiles"]).strip()
                if not drug_id:
                    raise ValueError("Empty drug_id found in %s." % csv_path)
                if drug_id not in drug_smiles or (
                    not drug_smiles[drug_id] and smiles
                ):
                    drug_smiles[drug_id] = smiles
    if not drug_smiles:
        raise ValueError("No drug_id values found in processed CSV files.")
    return drug_smiles


def morgan_feature(
    smiles: str,
    Chem,
    AllChem,
    n_bits: int,
    radius: int,
    as_sequence: bool,
) -> torch.Tensor:
    """Return one Morgan fingerprint tensor for a SMILES string."""
    if n_bits <= 0:
        raise ValueError("--n_bits must be positive.")
    if radius < 0:
        raise ValueError("--radius must be non-negative.")
    if not smiles:
        raise ValueError("empty SMILES")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("RDKit failed to parse SMILES")
    bit_vector = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        int(radius),
        nBits=int(n_bits),
    )
    tensor = torch.tensor(
        [1.0 if bit == "1" else 0.0 for bit in bit_vector.ToBitString()],
        dtype=torch.float32,
    )
    if as_sequence:
        return tensor.unsqueeze(0)
    return tensor


def write_failed_smiles(path: Path, failures: List[Tuple[str, str, str]]) -> None:
    """Write failed SMILES rows, even when there are no failures."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["drug_id", "smiles", "reason"],
        )
        writer.writeheader()
        for drug_id, smiles, reason in failures:
            writer.writerow(
                {
                    "drug_id": drug_id,
                    "smiles": smiles,
                    "reason": reason,
                }
            )


def parse_bool(value: object) -> bool:
    """Parse a command-line bool value."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value, got %r." % value)


def resolve_path(path: str) -> Path:
    """Resolve a path relative to the project root."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def relative_path(path: Path) -> str:
    """Return a compact repo-relative path string when possible."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()

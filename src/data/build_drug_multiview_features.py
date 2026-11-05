#!/usr/bin/env python
"""Build complementary fixed-length drug fingerprint views by drug ID."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.build_drug_morgan_features import (
    dataset_csv_paths,
    load_drug_smiles,
    relative_path,
    resolve_path,
    write_failed_smiles,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build five-view RDKit drug fingerprints."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("bindingdb", "biosnap", "davis", "kiba"),
    )
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--output", required=True)
    parser.add_argument("--failed-output", required=True)
    parser.add_argument("--n-bits", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.n_bits) <= 0:
        raise ValueError("--n-bits must be positive.")
    Chem, AllChem, rdMolDescriptors = load_rdkit()
    csv_paths = dataset_csv_paths(args.dataset, args.processed_dir)
    drug_smiles = load_drug_smiles(csv_paths)
    output = resolve_path(args.output)
    failed_output = resolve_path(args.failed_output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite feature bank: %s" % output)
    output.parent.mkdir(parents=True, exist_ok=True)
    failed_output.parent.mkdir(parents=True, exist_ok=True)

    features: Dict[str, torch.Tensor] = {}
    failures: List[Tuple[str, str, str]] = []
    for drug_id, smiles in sorted(drug_smiles.items()):
        try:
            features[drug_id] = multiview_feature(
                smiles=smiles,
                Chem=Chem,
                AllChem=AllChem,
                rdMolDescriptors=rdMolDescriptors,
                n_bits=int(args.n_bits),
            )
        except ValueError as exc:
            failures.append((drug_id, smiles, str(exc)))

    payload = {
        "features": features,
        "metadata": {
            "dataset": args.dataset,
            "representation": "rdkit_multiview_fingerprint",
            "views": [
                "morgan_radius_1",
                "morgan_radius_2",
                "morgan_radius_3",
                "hashed_atom_pair",
                "hashed_topological_torsion",
            ],
            "shape": [5, int(args.n_bits)],
        },
    }
    torch.save(payload, output)
    write_failed_smiles(failed_output, failures)
    print(
        {
            "unique_drugs": len(drug_smiles),
            "saved": len(features),
            "failed": len(failures),
            "shape": [5, int(args.n_bits)],
            "output": relative_path(output),
            "failed_output": relative_path(failed_output),
        }
    )


def load_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdMolDescriptors
    except ImportError as exc:
        raise SystemExit(
            "RDKit is required to build multiview drug features."
        ) from exc
    return Chem, AllChem, rdMolDescriptors


def multiview_feature(
    smiles: str,
    Chem,
    AllChem,
    rdMolDescriptors,
    n_bits: int,
) -> torch.Tensor:
    """Return five complementary binary fingerprint views as [5, n_bits]."""
    if not smiles:
        raise ValueError("empty SMILES")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("RDKit failed to parse SMILES")
    fingerprints = [
        AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        for radius in (1, 2, 3)
    ]
    fingerprints.extend(
        [
            rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
                mol,
                nBits=n_bits,
            ),
            rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(
                mol,
                nBits=n_bits,
            ),
        ]
    )
    return torch.stack([bit_vector_tensor(fp) for fp in fingerprints])


def bit_vector_tensor(bit_vector) -> torch.Tensor:
    return torch.tensor(
        [1.0 if bit == "1" else 0.0 for bit in bit_vector.ToBitString()],
        dtype=torch.float32,
    )


if __name__ == "__main__":
    main()

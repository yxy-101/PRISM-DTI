#!/usr/bin/env python
"""Build real RDKit 3D drug features for PRISM-DTI.

The EviDTI reference stores drug 3D information as GeoGNN graph dictionaries.
This script borrows the same RDKit/MMFF conformer idea, then adapts each
SMILES into the current PRISM-DTI FeatureStore contract:
`{drug_id: FloatTensor[num_nodes, feature_dim]}`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    """Parse real drug 3D feature build options."""
    parser = argparse.ArgumentParser(
        description="Build drug-id keyed real 3D features from processed CSV SMILES."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("biosnap", "bindingdb", "davis", "kiba"),
        help="Dataset name used in data/processed/{dataset}_{split}.csv.",
    )
    parser.add_argument(
        "--evidti-root",
        default="references/EviDTI-main",
        help="Reference EviDTI root used for provenance checks.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .pt path. Default: data/features/drug/{dataset}_drug3d_evidti.pt",
    )
    parser.add_argument(
        "--failed-output",
        default=None,
        help="Failed SMILES CSV path. Default: data/features/drug/{dataset}_drug3d_evidti_failed.csv",
    )
    parser.add_argument(
        "--report-output",
        default=None,
        help="Report JSON path. Default: data/features/drug/{dataset}_drug3d_evidti_report.json",
    )
    parser.add_argument("--num-nodes", type=int, default=16)
    parser.add_argument("--feature-dim", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Directory containing {dataset}_{train,valid,test}.csv.",
    )
    return parser.parse_args()


def main() -> None:
    """Build and save real 3D drug feature tensors."""
    args = parse_args()
    validate_shape_args(args.num_nodes, args.feature_dim)
    Chem, AllChem = load_rdkit()
    paths = output_paths(args)
    for key in ("output", "failed_output", "report_output"):
        warn_if_overwriting(paths[key])
        paths[key].parent.mkdir(parents=True, exist_ok=True)

    csv_paths = dataset_csv_paths(args.dataset, args.processed_dir)
    drug_smiles, conflicts = load_drug_smiles(csv_paths)
    reference_status = inspect_evidti_reference(args.evidti_root)

    features: Dict[str, torch.Tensor] = {}
    failures: List[Tuple[str, str, str]] = []
    success_count = 0
    for index, (drug_id, smiles) in enumerate(sorted(drug_smiles.items()), start=1):
        try:
            seed = molecule_seed(args.seed, drug_id)
            features[drug_id] = real_drug3d_feature(
                smiles=smiles,
                Chem=Chem,
                AllChem=AllChem,
                num_nodes=args.num_nodes,
                feature_dim=args.feature_dim,
                seed=seed,
            )
            success_count += 1
        except Exception as exc:
            features[drug_id] = zero_feature(args.num_nodes, args.feature_dim)
            failures.append((drug_id, smiles, str(exc)))

        if index % 1000 == 0:
            print("processed_drugs: %d/%d" % (index, len(drug_smiles)))

    torch.save(features, paths["output"])
    write_failed_csv(paths["failed_output"], failures)
    report = build_report(
        args=args,
        paths=paths,
        num_drugs=len(drug_smiles),
        success_count=success_count,
        failures=failures,
        conflicts=conflicts,
        reference_status=reference_status,
    )
    paths["report_output"].write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("dataset: %s" % args.dataset)
    print("unique_drugs: %d" % len(drug_smiles))
    print("success_count: %d" % success_count)
    print("failed_count: %d" % len(failures))
    print("feature_shape: [%d, %d]" % (args.num_nodes, args.feature_dim))
    print("output: %s" % relative_path(paths["output"]))
    print("failed_output: %s" % relative_path(paths["failed_output"]))
    print("report_output: %s" % relative_path(paths["report_output"]))


def validate_shape_args(num_nodes: int, feature_dim: int) -> None:
    """Validate target tensor shape options."""
    if int(num_nodes) <= 0:
        raise ValueError("--num-nodes must be positive.")
    if int(feature_dim) <= 0:
        raise ValueError("--feature-dim must be positive.")


def load_rdkit():
    """Import RDKit lazily with a clear setup message."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise SystemExit(
            "RDKit is required for real drug3D feature building. "
            "Install rdkit in the active environment before running this script."
        ) from exc
    return Chem, AllChem


def output_paths(args: argparse.Namespace) -> Dict[str, Path]:
    """Resolve dynamic output paths for a dataset."""
    defaults = {
        "output": "data/features/drug/{dataset}_drug3d_evidti.pt",
        "failed_output": "data/features/drug/{dataset}_drug3d_evidti_failed.csv",
        "report_output": "data/features/drug/{dataset}_drug3d_evidti_report.json",
    }
    return {
        key: resolve_path(
            str(getattr(args, key) or default).format(dataset=args.dataset)
        )
        for key, default in defaults.items()
    }


def warn_if_overwriting(path: Path) -> None:
    """Print a warning when a repeat run will overwrite an output file."""
    if path.exists():
        print("WARNING: overwriting existing file: %s" % relative_path(path))


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


def load_drug_smiles(
    csv_paths: Iterable[Path],
) -> Tuple[Dict[str, str], List[Tuple[str, str, str]]]:
    """Collect one SMILES string per drug ID across processed splits."""
    required = {"drug_id", "smiles"}
    drug_smiles: Dict[str, str] = {}
    conflicts: List[Tuple[str, str, str]] = []
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
                previous = drug_smiles.get(drug_id)
                if previous is None or (not previous and smiles):
                    drug_smiles[drug_id] = smiles
                elif smiles and previous and smiles != previous:
                    conflicts.append((drug_id, previous, smiles))
    if not drug_smiles:
        raise ValueError("No drug_id values found in processed CSV files.")
    if conflicts:
        print("WARNING: found %d drug_id SMILES conflicts; kept first." % len(conflicts))
    return drug_smiles, conflicts


def real_drug3d_feature(
    smiles: str,
    Chem,
    AllChem,
    num_nodes: int,
    feature_dim: int,
    seed: int,
) -> torch.Tensor:
    """Generate one real 3D atom-level feature tensor from a SMILES string."""
    if not str(smiles or "").strip():
        raise ValueError("empty SMILES")
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError("RDKit failed to parse SMILES")
    mol = largest_fragment(mol, Chem)
    if mol.GetNumAtoms() == 0:
        raise ValueError("SMILES produced an empty molecule")
    mol_3d = embed_3d_molecule(mol, Chem, AllChem, seed=seed)
    return atom_level_feature_tensor(
        mol=mol_3d,
        Chem=Chem,
        num_nodes=num_nodes,
        feature_dim=feature_dim,
    )


def largest_fragment(mol, Chem):
    """Use the largest fragment for salts and multi-species SMILES."""
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not fragments:
        return mol
    return max(fragments, key=lambda item: (item.GetNumHeavyAtoms(), item.GetNumAtoms()))


def embed_3d_molecule(mol, Chem, AllChem, seed: int):
    """Generate an ETKDG conformer and optimize it with MMFF or UFF."""
    mol_h = Chem.AddHs(mol)
    params = make_etkdg_params(AllChem, seed)
    status = AllChem.EmbedMolecule(mol_h, params)
    if status != 0:
        retry_params = make_etkdg_params(AllChem, seed)
        retry_params.useRandomCoords = True
        status = AllChem.EmbedMolecule(mol_h, retry_params)
    if status != 0:
        raise ValueError("RDKit ETKDG EmbedMolecule failed with code %s" % status)

    if not optimize_molecule(mol_h, AllChem):
        raise ValueError("RDKit MMFF and UFF optimization failed")

    mol_3d = Chem.RemoveHs(mol_h)
    if mol_3d.GetNumConformers() == 0:
        raise ValueError("3D conformer missing after hydrogen removal")
    return mol_3d


def make_etkdg_params(AllChem, seed: int):
    """Create deterministic ETKDG parameters."""
    try:
        params = AllChem.ETKDGv3()
    except AttributeError:
        params = AllChem.ETKDG()
    params.randomSeed = int(seed)
    return params


def optimize_molecule(mol_h, AllChem) -> bool:
    """Run MMFF optimization, falling back to UFF when MMFF is unavailable."""
    try:
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94")
    except Exception:
        mmff_props = None
    if mmff_props is not None:
        try:
            status = AllChem.MMFFOptimizeMolecule(
                mol_h,
                mmffVariant="MMFF94",
                maxIters=200,
            )
            if status >= 0:
                return True
        except Exception:
            pass

    try:
        status = AllChem.UFFOptimizeMolecule(mol_h, maxIters=200)
    except Exception:
        return False
    return status >= 0


def atom_level_feature_tensor(mol, Chem, num_nodes: int, feature_dim: int) -> torch.Tensor:
    """Convert an optimized conformer into `[num_nodes, feature_dim]`."""
    atom_count = mol.GetNumAtoms()
    if atom_count == 0:
        raise ValueError("3D molecule has no atoms")
    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        raise ValueError("3D molecule contains dummy atoms")
    conf = mol.GetConformer()
    positions = [
        (
            float(conf.GetAtomPosition(index).x),
            float(conf.GetAtomPosition(index).y),
            float(conf.GetAtomPosition(index).z),
        )
        for index in range(atom_count)
    ]
    selected_count = min(num_nodes, atom_count)
    selected_positions = positions[:selected_count]
    centered_positions, max_radius = center_positions(selected_positions)
    pairwise_distances = pairwise_distance_matrix(selected_positions, num_nodes)
    bond_order_matrix = selected_bond_order_matrix(mol, selected_count, num_nodes)
    topological_distances = topological_distance_matrix(mol, selected_count, num_nodes)
    charges = partial_charges(mol)

    feature = zero_feature(num_nodes, feature_dim)
    for atom_index in range(selected_count):
        atom = mol.GetAtomWithIdx(atom_index)
        vector = atom_feature_vector(
            atom=atom,
            atom_index=atom_index,
            atom_count=atom_count,
            Chem=Chem,
            centered_position=centered_positions[atom_index],
            raw_position=selected_positions[atom_index],
            max_radius=max_radius,
            distances=pairwise_distances[atom_index],
            bond_orders=bond_order_matrix[atom_index],
            topo_distances=topological_distances[atom_index],
            charge=charges[atom_index],
            feature_dim=feature_dim,
            positions=positions,
        )
        feature[atom_index] = torch.tensor(vector, dtype=torch.float32)
    return feature


def atom_feature_vector(
    atom,
    atom_index: int,
    atom_count: int,
    Chem,
    centered_position: Tuple[float, float, float],
    raw_position: Tuple[float, float, float],
    max_radius: float,
    distances: Sequence[float],
    bond_orders: Sequence[float],
    topo_distances: Sequence[float],
    charge: float,
    feature_dim: int,
    positions: Sequence[Tuple[float, float, float]],
) -> List[float]:
    """Build one deterministic atom feature vector, then pad/truncate."""
    atomic_num = int(atom.GetAtomicNum())
    periodic_table = Chem.GetPeriodicTable()
    x, y, z = centered_position
    raw_x, raw_y, raw_z = raw_position
    radius = math.sqrt(x * x + y * y + z * z)
    local_lengths = local_bond_lengths(atom, positions)
    local_angles = local_bond_angles(atom, positions)

    vector: List[float] = [
        atomic_num / 118.0,
        float(atom.GetMass()) / 250.0,
        float(atom.GetDegree()) / 8.0,
        float(atom.GetTotalDegree()) / 8.0,
        float(atom.GetExplicitValence()) / 8.0,
        float(atom.GetImplicitValence()) / 8.0,
        float(atom.GetFormalCharge()) / 8.0,
        float(atom.GetTotalNumHs()) / 8.0,
        1.0 if atom.GetIsAromatic() else 0.0,
        1.0 if atom.IsInRing() else 0.0,
        enum_to_float(atom.GetChiralTag()) / 8.0,
        enum_to_float(atom.GetHybridization()) / 8.0,
        clip(charge, -4.0, 4.0) / 4.0,
        safe_periodic_value(periodic_table.GetRvdw, atomic_num) / 4.0,
        safe_periodic_value(periodic_table.GetRcovalent, atomic_num) / 3.0,
        safe_periodic_value(periodic_table.GetNOuterElecs, atomic_num) / 8.0,
        float(atom_index) / max(1.0, float(atom_count - 1)),
        x / max_radius,
        y / max_radius,
        z / max_radius,
        raw_x / 20.0,
        raw_y / 20.0,
        raw_z / 20.0,
        radius / max_radius,
    ]
    vector.extend(one_hot(atomic_num, [1, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 19, 20, 35, 53]))
    vector.extend(
        one_hot(
            atom.GetHybridization(),
            [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
            ],
        )
    )
    vector.extend(one_hot(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3]))
    vector.extend(ring_size_features(atom))
    vector.extend(local_bond_summary(local_lengths))
    vector.extend(local_angle_summary(local_angles))
    vector.extend([float(value) / 20.0 for value in distances])
    vector.extend([float(value) / 3.0 for value in bond_orders])
    vector.extend([float(value) / 20.0 for value in topo_distances])

    if len(vector) < feature_dim:
        vector.extend([0.0] * (feature_dim - len(vector)))
    return vector[:feature_dim]


def center_positions(
    positions: Sequence[Tuple[float, float, float]]
) -> Tuple[List[Tuple[float, float, float]], float]:
    """Center selected coordinates and return a radius scale."""
    if not positions:
        return [], 1.0
    count = float(len(positions))
    cx = sum(pos[0] for pos in positions) / count
    cy = sum(pos[1] for pos in positions) / count
    cz = sum(pos[2] for pos in positions) / count
    centered = [(x - cx, y - cy, z - cz) for x, y, z in positions]
    max_radius = max(math.sqrt(x * x + y * y + z * z) for x, y, z in centered)
    return centered, max(max_radius, 1.0)


def pairwise_distance_matrix(
    positions: Sequence[Tuple[float, float, float]],
    num_nodes: int,
) -> List[List[float]]:
    """Return a padded pairwise Euclidean distance matrix."""
    matrix = [[0.0 for _ in range(num_nodes)] for _ in range(num_nodes)]
    for i, left in enumerate(positions[:num_nodes]):
        for j, right in enumerate(positions[:num_nodes]):
            matrix[i][j] = euclidean_distance(left, right)
    return matrix


def selected_bond_order_matrix(mol, selected_count: int, num_nodes: int) -> List[List[float]]:
    """Return a padded selected-atom bond order matrix."""
    matrix = [[0.0 for _ in range(num_nodes)] for _ in range(num_nodes)]
    for bond in mol.GetBonds():
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        if begin >= selected_count or end >= selected_count:
            continue
        order = float(bond.GetBondTypeAsDouble())
        matrix[begin][end] = order
        matrix[end][begin] = order
    return matrix


def topological_distance_matrix(mol, selected_count: int, num_nodes: int) -> List[List[float]]:
    """Return a padded graph-distance matrix."""
    matrix = [[0.0 for _ in range(num_nodes)] for _ in range(num_nodes)]
    distances = ChemDistanceMatrix(mol)
    for i in range(selected_count):
        for j in range(selected_count):
            matrix[i][j] = float(distances[i][j])
    return matrix


def ChemDistanceMatrix(mol):
    """Wrap RDKit's topological distance matrix for testability."""
    from rdkit import Chem

    return Chem.GetDistanceMatrix(mol)


def partial_charges(mol) -> List[float]:
    """Return Gasteiger partial charges, using zero if computation fails."""
    try:
        from rdkit import Chem

        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
    except Exception:
        return [0.0 for _ in mol.GetAtoms()]
    charges = []
    for atom in mol.GetAtoms():
        try:
            charge = float(atom.GetProp("_GasteigerCharge"))
        except Exception:
            charge = 0.0
        if math.isnan(charge) or math.isinf(charge):
            charge = 0.0
        charges.append(charge)
    return charges


def ring_size_features(atom) -> List[float]:
    """Return ring-size indicators for sizes 3 through 8."""
    return [1.0 if atom.IsInRingSize(size) else 0.0 for size in range(3, 9)]


def local_bond_lengths(
    atom,
    positions: Sequence[Tuple[float, float, float]],
) -> List[float]:
    """Return 3D lengths for bonds incident to an atom."""
    atom_index = atom.GetIdx()
    lengths = []
    for bond in atom.GetBonds():
        other_index = bond.GetOtherAtomIdx(atom_index)
        lengths.append(euclidean_distance(positions[atom_index], positions[other_index]))
    return lengths


def local_bond_angles(
    atom,
    positions: Sequence[Tuple[float, float, float]],
) -> List[float]:
    """Return angles between incident bonds around one atom."""
    atom_index = atom.GetIdx()
    center = positions[atom_index]
    neighbor_indices = [
        bond.GetOtherAtomIdx(atom_index)
        for bond in atom.GetBonds()
    ]
    angles = []
    for left_pos, left_index in enumerate(neighbor_indices):
        for right_index in neighbor_indices[left_pos + 1 :]:
            angles.append(
                angle_between(
                    vector_between(center, positions[left_index]),
                    vector_between(center, positions[right_index]),
                )
            )
    return angles


def local_bond_summary(lengths: Sequence[float]) -> List[float]:
    """Return count and length statistics for incident bonds."""
    if not lengths:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        len(lengths) / 8.0,
        min(lengths) / 4.0,
        max(lengths) / 4.0,
        (sum(lengths) / len(lengths)) / 4.0,
    ]


def local_angle_summary(angles: Sequence[float]) -> List[float]:
    """Return count and angle statistics for incident bond angles."""
    if not angles:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        len(angles) / 16.0,
        min(angles) / math.pi,
        max(angles) / math.pi,
        (sum(angles) / len(angles)) / math.pi,
    ]


def vector_between(
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Return the vector from start to end."""
    return (end[0] - start[0], end[1] - start[1], end[2] - start[2])


def angle_between(
    left: Tuple[float, float, float],
    right: Tuple[float, float, float],
) -> float:
    """Return the angle between two vectors in radians."""
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    dot = sum(left[i] * right[i] for i in range(3)) / (left_norm * right_norm)
    return math.acos(clip(dot, -1.0, 1.0))


def euclidean_distance(
    left: Tuple[float, float, float],
    right: Tuple[float, float, float],
) -> float:
    """Return Euclidean distance between two 3D points."""
    return math.sqrt(
        (left[0] - right[0]) ** 2
        + (left[1] - right[1]) ** 2
        + (left[2] - right[2]) ** 2
    )


def one_hot(value: Any, choices: Sequence[Any]) -> List[float]:
    """Return a one-hot vector with a final miscellaneous bucket."""
    matched = [1.0 if value == choice else 0.0 for choice in choices]
    matched.append(0.0 if any(value == choice for choice in choices) else 1.0)
    return matched


def safe_periodic_value(function, atomic_num: int) -> float:
    """Call an RDKit periodic-table getter safely."""
    try:
        value = float(function(int(atomic_num)))
    except Exception:
        return 0.0
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return value


def enum_to_float(value: Any) -> float:
    """Convert an RDKit enum-like value to a float."""
    try:
        return float(int(value))
    except Exception:
        return 0.0


def clip(value: float, low: float, high: float) -> float:
    """Clamp one scalar value."""
    return max(low, min(high, float(value)))


def zero_feature(num_nodes: int, feature_dim: int) -> torch.Tensor:
    """Return the deterministic zero fallback tensor."""
    return torch.zeros(int(num_nodes), int(feature_dim), dtype=torch.float32)


def molecule_seed(base_seed: int, drug_id: str) -> int:
    """Derive a deterministic conformer seed per drug ID."""
    return (int(base_seed) + stable_int(str(drug_id))) % 2_000_000_000


def stable_int(text: str) -> int:
    """Return a stable integer hash."""
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def write_failed_csv(path: Path, failures: List[Tuple[str, str, str]]) -> None:
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


def inspect_evidti_reference(evidti_root: str) -> Dict[str, Any]:
    """Record whether key EviDTI reference files are present."""
    root = Path(evidti_root).expanduser()
    files = [
        "extract_drug_3d_emb.py",
        "compound_encode.py",
        "compound_tools.py",
        "extract_drug_2d_emb.py",
    ]
    directories = [
        "configs",
        "dataset",
    ]
    return {
        "root": str(root),
        "files_present": {name: (root / name).exists() for name in files},
        "directories_present": {
            name: (root / name).is_dir() for name in directories
        },
        "direct_reuse": False,
        "direct_reuse_reason": (
            "EviDTI builds Paddle/PyG GeoGNN graph dictionaries stored as "
            "sample-order .npy objects, not a drug_id-keyed PRISM-DTI "
            "FeatureStore tensor bank with shape [num_nodes, feature_dim]."
        ),
    }


def build_report(
    args: argparse.Namespace,
    paths: Dict[str, Path],
    num_drugs: int,
    success_count: int,
    failures: Sequence[Tuple[str, str, str]],
    conflicts: Sequence[Tuple[str, str, str]],
    reference_status: Dict[str, Any],
) -> Dict[str, Any]:
    """Create the JSON report payload."""
    return {
        "dataset": args.dataset,
        "num_drugs": int(num_drugs),
        "success_count": int(success_count),
        "failed_count": int(len(failures)),
        "output": relative_path(paths["output"]),
        "failed_output": relative_path(paths["failed_output"]),
        "feature_shape": [int(args.num_nodes), int(args.feature_dim)],
        "note": (
            "v0.7 real drug3D features use deterministic RDKit ETKDG 3D "
            "conformers with MMFF/UFF optimization, atom-order truncation, "
            "zero padding to the target shape, and zero fallback only for "
            "failed conformer generation recorded in failed.csv."
        ),
        "report_output": relative_path(paths["report_output"]),
        "seed": int(args.seed),
        "smiles_conflict_count": int(len(conflicts)),
        "evidti_reference": reference_status,
    }


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

#!/usr/bin/env python
"""Run a paired experimental-coordinate robustness check for one DTI case.

Only the drug 3D coordinates are changed. Experimental ligand atoms are mapped
to the processed-SMILES atom order and rigidly aligned to the ETKDG coordinate
frame before the existing 16 x 300 feature builder is called. This prevents
atom order, translation, and rotation from contaminating the comparison.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import gc
import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.build_drug3d_features_from_evidti import atom_level_feature_tensor


RAW_COORDINATE_SLICE = slice(20, 23)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace one BioSNAP drug's ETKDG coordinates with aligned experimental SDF coordinates."
    )
    parser.add_argument("--test-csv", default="data/processed/biosnap_test.csv")
    parser.add_argument("--drug-id", default="drug_6869a315f27b")
    parser.add_argument("--target-id", default="204")
    parser.add_argument(
        "--baseline-drug3d-bank",
        default="data/features/drug/biosnap_drug3d_evidti.pt",
    )
    parser.add_argument(
        "--experimental-sdf",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Repeat for each experimental ligand instance.",
    )
    parser.add_argument(
        "--allow-canonical-tautomer-map",
        action="store_true",
        help=(
            "If exact bond-order graph matching fails, canonicalize both "
            "molecules' tautomers and require a complete heavy-atom mapping. "
            "Stereochemistry is not used for atom mapping and is reported as "
            "an identity limitation when unspecified in the benchmark SMILES."
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 2024])
    parser.add_argument(
        "--config-template",
        default=(
            "configs/final/biosnap/"
            "biosnap_do_causal_ft_rho025_seed{seed}.json"
        ),
    )
    parser.add_argument(
        "--checkpoint-template",
        default="outputs/checkpoints/biosnap_do_causal_ft_rho025_seed{seed}.pt",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--pcs-margin", type=float, default=0.0)
    parser.add_argument(
        "--prediction-repeats",
        type=int,
        default=1,
        help=(
            "Repeat each fixed checkpoint/input prediction to estimate numerical "
            "inference variation. Repeats do not alter model or audit thresholds."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        default="biosnap_rho025_ketamine_experimental3d_coordinate_robustness",
    )
    parser.add_argument(
        "--skip-prediction",
        action="store_true",
        help="Build aligned feature banks without invoking scripts/predict.py.",
    )
    return parser.parse_args()


def resolve(path: str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def parse_named_paths(values: Sequence[str]) -> List[Tuple[str, Path]]:
    if not values:
        values = [
            "7EU7=data/verified3d/rcsb/ligands/7EU7_JC9_D_905.sdf",
            "7EU8=data/verified3d/rcsb/ligands/7EU8_JC9_B_903.sdf",
            "7SAC=data/verified3d/rcsb/ligands/7SAC_JC9_B_903.sdf",
        ]
    parsed: List[Tuple[str, Path]] = []
    seen = set()
    for value in values:
        if "=" not in value:
            raise ValueError("--experimental-sdf must use NAME=PATH: %s" % value)
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name or name in seen:
            raise ValueError("Experimental SDF names must be nonempty and unique.")
        path = resolve(raw_path.strip())
        if not path.exists():
            raise FileNotFoundError("Experimental SDF not found: %s" % path)
        seen.add(name)
        parsed.append((name, path))
    return parsed


def load_case(csv_path: Path, drug_id: str, target_id: str) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    required = {"drug_id", "target_id", "label", "smiles"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Missing case columns: %s" % ", ".join(missing))
    selected = frame[
        frame["drug_id"].astype(str).eq(str(drug_id))
        & frame["target_id"].astype(str).eq(str(target_id))
    ].copy()
    if len(selected) != 1:
        raise ValueError(
            "Expected exactly one case for %s/%s, found %d."
            % (drug_id, target_id, len(selected))
        )
    return selected


def unwrap_bank(payload, drug_id: str) -> Dict[str, torch.Tensor]:
    candidates = [payload]
    if isinstance(payload, dict):
        candidates.extend(
            payload.get(key) for key in ("features", "feature_bank", "data")
        )
    for candidate in candidates:
        if isinstance(candidate, dict) and drug_id in candidate:
            return candidate
    raise KeyError("Drug %s is absent from the baseline 3D feature bank." % drug_id)


def config_feature_path(config: Dict[str, object], key: str) -> Path:
    """Resolve one top-level or nested feature-bank path from a config."""
    features = config.get("features", {})
    nested = features if isinstance(features, dict) else {}
    value = config.get(key, nested.get(key))
    if not value:
        raise KeyError("Config is missing %s." % key)
    return resolve(str(value))


def release_host_memory() -> None:
    """Return large temporary torch-load allocations before child inference."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (AttributeError, OSError):
        pass


def extract_case_feature(
    source_path: Path,
    feature_id: str,
    output_path: Path,
) -> Path:
    """Write one ID from a full feature bank to a small prediction-only bank."""
    payload = torch.load(source_path, map_location="cpu")
    bank = unwrap_bank(payload, feature_id)
    feature = torch.as_tensor(bank[feature_id]).detach().cpu().clone()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({feature_id: feature}, output_path)
    del feature
    del bank
    del payload
    release_host_memory()
    return output_path


def build_case_input_banks(
    args: argparse.Namespace,
    case: pd.DataFrame,
) -> Dict[str, Path]:
    """Create one-case protein and 2D banks to bound inference memory."""
    first_seed = int(args.seeds[0])
    config_path = resolve(args.config_template.format(seed=first_seed))
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    feature_dir = resolve("data/verified3d/features")
    prefix = str(args.output_prefix)
    target_id = str(case.iloc[0]["target_id"])
    drug_id = str(case.iloc[0]["drug_id"])
    protein = extract_case_feature(
        source_path=config_feature_path(config, "protein_feature_path"),
        feature_id=target_id,
        output_path=feature_dir / (prefix + "_protein_case.pt"),
    )
    drug2d = extract_case_feature(
        source_path=config_feature_path(config, "drug2d_feature_path"),
        feature_id=drug_id,
        output_path=feature_dir / (prefix + "_drug2d_case.pt"),
    )
    return {"protein": protein, "drug2d": drug2d}


def largest_fragment(mol, Chem):
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    return max(fragments, key=lambda item: item.GetNumHeavyAtoms()) if fragments else mol


def read_sdf(path: Path, Chem):
    supplier = Chem.SDMolSupplier(str(path), removeHs=True)
    mol = next((item for item in supplier if item is not None), None)
    if mol is None or mol.GetNumConformers() != 1:
        raise ValueError("Expected one readable SDF conformer: %s" % path)
    if not mol.GetConformer().Is3D():
        raise ValueError("SDF does not contain 3D coordinates: %s" % path)
    return mol


def coordinates(mol, atom_indices: Iterable[int]) -> np.ndarray:
    conformer = mol.GetConformer()
    return np.asarray(
        [
            [
                conformer.GetAtomPosition(int(index)).x,
                conformer.GetAtomPosition(int(index)).y,
                conformer.GetAtomPosition(int(index)).z,
            ]
            for index in atom_indices
        ],
        dtype=np.float64,
    )


def rigid_transform(
    moving: np.ndarray, reference: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if moving.shape != reference.shape or moving.ndim != 2 or moving.shape[1] != 3:
        raise ValueError("Coordinate arrays must have equal [atoms, 3] shapes.")
    moving_center = moving.mean(axis=0)
    reference_center = reference.mean(axis=0)
    left = moving - moving_center
    right = reference - reference_center
    u, _, vt = np.linalg.svd(left.T @ right)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    aligned = left @ rotation + reference_center
    rmsd = float(np.sqrt(np.mean(np.sum((aligned - reference) ** 2, axis=1))))
    return rotation, moving_center, reference_center, rmsd


def apply_rigid_transform(
    coordinates_to_align: np.ndarray,
    rotation: np.ndarray,
    moving_center: np.ndarray,
    reference_center: np.ndarray,
) -> np.ndarray:
    return (coordinates_to_align - moving_center) @ rotation + reference_center


def molecule_with_coordinates(template, aligned: np.ndarray, Chem):
    from rdkit.Geometry import Point3D

    mol = Chem.Mol(template)
    mol.RemoveAllConformers()
    conformer = Chem.Conformer(mol.GetNumAtoms())
    conformer.Set3D(True)
    for index, (x, y, z) in enumerate(aligned):
        conformer.SetAtomPosition(
            index,
            Point3D(float(x), float(y), float(z)),
        )
    mol.AddConformer(conformer, assignId=True)
    return mol


def canonical_tautomer_atom_mapping(template, experimental, Chem) -> Tuple[int, ...]:
    """Map template atom order to an experimental SDF through canonical tautomers."""
    from rdkit.Chem.MolStandardize import rdMolStandardize

    template_tagged = Chem.Mol(template)
    experimental_tagged = Chem.Mol(experimental)
    for molecule in (template_tagged, experimental_tagged):
        for index, atom in enumerate(molecule.GetAtoms()):
            atom.SetAtomMapNum(index + 1)

    enumerator = rdMolStandardize.TautomerEnumerator()
    template_canonical = enumerator.Canonicalize(template_tagged)
    experimental_canonical = enumerator.Canonicalize(experimental_tagged)
    match = experimental_canonical.GetSubstructMatch(
        template_canonical, useChirality=False
    )
    if len(match) != template.GetNumAtoms() or (
        experimental.GetNumAtoms() != template.GetNumAtoms()
    ):
        raise ValueError(
            "Canonical tautomer mapping did not cover every heavy atom "
            "(%d/%d mapped; experimental has %d)."
            % (len(match), template.GetNumAtoms(), experimental.GetNumAtoms())
        )

    mapping = [-1] * template.GetNumAtoms()
    for canonical_template_index, canonical_experimental_index in enumerate(match):
        template_original_index = (
            template_canonical.GetAtomWithIdx(canonical_template_index).GetAtomMapNum()
            - 1
        )
        experimental_original_index = (
            experimental_canonical.GetAtomWithIdx(
                canonical_experimental_index
            ).GetAtomMapNum()
            - 1
        )
        if (
            template.GetAtomWithIdx(template_original_index).GetSymbol()
            != experimental.GetAtomWithIdx(experimental_original_index).GetSymbol()
        ):
            raise ValueError("Canonical tautomer mapping changed an atom identity.")
        mapping[template_original_index] = experimental_original_index
    if any(index < 0 for index in mapping) or len(set(mapping)) != len(mapping):
        raise ValueError("Canonical tautomer mapping is not a complete bijection.")
    return tuple(mapping)


def specified_stereochemistry_counts(molecule) -> Dict[str, int]:
    """Count explicit tetrahedral and double-bond stereo in a molecule."""
    from rdkit import Chem

    return {
        "specified_tetrahedral_centers": sum(
            atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
            for atom in molecule.GetAtoms()
        ),
        "specified_stereogenic_bonds": sum(
            bond.GetStereo() != Chem.BondStereo.STEREONONE
            for bond in molecule.GetBonds()
        ),
    }


def build_feature_banks(
    args: argparse.Namespace,
    case: pd.DataFrame,
    named_sdfs: Sequence[Tuple[str, Path]],
) -> Tuple[Dict[str, Path], List[Dict[str, object]]]:
    from rdkit import Chem

    drug_id = str(args.drug_id)
    payload = torch.load(resolve(args.baseline_drug3d_bank), map_location="cpu")
    bank = unwrap_bank(payload, drug_id)
    baseline = torch.as_tensor(bank[drug_id]).detach().cpu().float()
    if baseline.ndim != 2 or baseline.shape[1] < RAW_COORDINATE_SLICE.stop:
        raise ValueError("Unexpected baseline feature shape: %s" % (tuple(baseline.shape),))

    template = largest_fragment(Chem.MolFromSmiles(str(case.iloc[0]["smiles"])), Chem)
    if template is None:
        raise ValueError("Processed SMILES could not be parsed.")
    atom_count = template.GetNumAtoms()
    represented_atom_count = min(atom_count, int(baseline.shape[0]))
    reference = (
        baseline[:represented_atom_count, RAW_COORDINATE_SLICE].numpy() * 20.0
    )

    feature_dir = resolve("data/verified3d/features")
    feature_dir.mkdir(parents=True, exist_ok=True)
    paths = {"ETKDG": feature_dir / (args.output_prefix + "_ETKDG.pt")}
    torch.save({drug_id: baseline}, paths["ETKDG"])
    records: List[Dict[str, object]] = []

    for name, sdf_path in named_sdfs:
        experimental = read_sdf(sdf_path, Chem)
        match = experimental.GetSubstructMatch(template, useChirality=False)
        mapping_method = "exact_heavy_atom_graph"
        if len(match) != atom_count and args.allow_canonical_tautomer_map:
            match = canonical_tautomer_atom_mapping(template, experimental, Chem)
            mapping_method = "canonical_tautomer_heavy_atom_graph"
        if len(match) != atom_count:
            raise ValueError(
                "%s does not map exactly to the processed SMILES (%d/%d atoms)."
                % (name, len(match), atom_count)
            )
        experimental_coordinates = coordinates(experimental, match)
        rotation, moving_center, reference_center, rmsd = rigid_transform(
            experimental_coordinates[:represented_atom_count], reference
        )
        aligned_all = apply_rigid_transform(
            experimental_coordinates,
            rotation,
            moving_center,
            reference_center,
        )
        coordinate_only_mol = molecule_with_coordinates(template, aligned_all, Chem)
        feature = atom_level_feature_tensor(
            mol=coordinate_only_mol,
            Chem=Chem,
            num_nodes=int(baseline.shape[0]),
            feature_dim=int(baseline.shape[1]),
        )
        path = feature_dir / (args.output_prefix + "_%s.pt" % name)
        torch.save({drug_id: feature}, path)
        paths[name] = path
        records.append(
            {
                "structure": name,
                "sdf_path": str(sdf_path.relative_to(ROOT)),
                "atom_mapping_method": mapping_method,
                "atom_count": atom_count,
                "mapped_atom_count": len(match),
                "model_stereochemistry": specified_stereochemistry_counts(template),
                "experimental_stereochemistry": specified_stereochemistry_counts(
                    experimental
                ),
                "represented_atom_count": represented_atom_count,
                "truncated_atom_count": atom_count - represented_atom_count,
                "aligned_coordinate_rmsd": rmsd,
                "feature_max_abs_difference_from_etkdg": float(
                    torch.max(torch.abs(feature - baseline)).item()
                ),
            }
        )
    del baseline
    del bank
    del payload
    release_host_memory()
    return paths, records


def predict_one(
    config_path: Path,
    checkpoint_path: Path,
    case_csv: Path,
    feature_bank: Path,
    case_feature_banks: Dict[str, Path],
    output_path: Path,
    temporary_dir: Path,
) -> pd.Series:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    config["valid_csv"] = str(case_csv.relative_to(ROOT))
    config["test_csv"] = str(case_csv.relative_to(ROOT))
    config["drug3d_feature_path"] = str(feature_bank.relative_to(ROOT))
    config["protein_feature_path"] = str(
        case_feature_banks["protein"].relative_to(ROOT)
    )
    config["drug2d_feature_path"] = str(
        case_feature_banks["drug2d"].relative_to(ROOT)
    )
    config.setdefault("features", {})["protein_feature_path"] = str(
        case_feature_banks["protein"].relative_to(ROOT)
    )
    config["features"]["drug2d_feature_path"] = str(
        case_feature_banks["drug2d"].relative_to(ROOT)
    )
    config.setdefault("features", {})["drug3d_feature_path"] = str(
        feature_bank.relative_to(ROOT)
    )
    config["output_prediction_path"] = str(output_path.relative_to(ROOT))
    config["max_valid_samples"] = None
    config["batch_size"] = 1
    temporary_config = temporary_dir / "case.yaml"
    serialized_config = yaml.safe_dump(config, sort_keys=False)
    temporary_config.write_text(serialized_config, encoding="utf-8")
    command = [
        sys.executable,
        "-B",
        "scripts/predict.py",
        "--config",
        str(temporary_config.relative_to(ROOT)),
        "--checkpoint",
        str(checkpoint_path.relative_to(ROOT)),
    ]
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "4"
    environment["MKL_NUM_THREADS"] = "4"
    environment["PYTHONUNBUFFERED"] = "1"
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        check=False,
        env=environment,
    )
    if completed.returncode != 0:
        debug_config = resolve(
            "outputs/resolved_configs/verified3d_coordinate_failed.yaml"
        )
        debug_config.write_text(serialized_config, encoding="utf-8")
        raise RuntimeError(
            "Prediction failed for %s with return code %d. "
            "Repro config: %s. Command: %s"
            % (
                checkpoint_path.name,
                completed.returncode,
                debug_config.relative_to(ROOT),
                " ".join(command),
            )
        )
    prediction = pd.read_csv(output_path)
    if len(prediction) != 1:
        raise ValueError("Expected one prediction row, found %d." % len(prediction))
    return prediction.iloc[0]


def run_predictions(
    args: argparse.Namespace,
    case_csv: Path,
    feature_paths: Dict[str, Path],
    case_feature_banks: Dict[str, Path],
) -> pd.DataFrame:
    prediction_dir = resolve("outputs/predictions/verified3d_coordinate")
    prediction_dir.mkdir(parents=True, exist_ok=True)
    resolve("outputs/resolved_configs/verified3d_coordinate_failed.yaml").unlink(
        missing_ok=True
    )
    rows = []
    resolved_root = resolve("outputs/resolved_configs")
    resolved_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="verified3d_", dir=str(resolved_root)) as raw_tmp:
        temporary_dir = Path(raw_tmp)
        for seed in args.seeds:
            config_path = resolve(args.config_template.format(seed=seed))
            checkpoint_path = resolve(args.checkpoint_template.format(seed=seed))
            if not config_path.exists() or not checkpoint_path.exists():
                raise FileNotFoundError(
                    "Missing seed %d config/checkpoint: %s | %s"
                    % (seed, config_path, checkpoint_path)
                )
            for structure, feature_path in feature_paths.items():
                for repeat in range(1, args.prediction_repeats + 1):
                    output_path = prediction_dir / (
                        "%s_seed%d_%s_repeat%d.csv"
                        % (args.output_prefix, seed, structure, repeat)
                    )
                    prediction = predict_one(
                        config_path=config_path,
                        checkpoint_path=checkpoint_path,
                        case_csv=case_csv,
                        feature_bank=feature_path,
                        case_feature_banks=case_feature_banks,
                        output_path=output_path,
                        temporary_dir=temporary_dir,
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "structure": structure,
                            "repeat": repeat,
                            "label": int(prediction["label"]),
                            "p_raw": float(prediction["p_raw"]),
                            "u_raw": float(prediction["u_raw"]),
                            "p_do": float(prediction["p_do"]),
                            "u_do": float(prediction["u_do"]),
                            "pcs": float(prediction["pcs"]),
                        }
                    )
                    output_path.unlink(missing_ok=True)
    frame = pd.DataFrame(rows)
    baseline = frame[frame["structure"].eq("ETKDG")][
        ["seed", "repeat", "p_raw", "p_do", "pcs"]
    ].rename(
        columns={
            "p_raw": "baseline_p_raw",
            "p_do": "baseline_p_do",
            "pcs": "baseline_pcs",
        }
    )
    frame = frame.merge(
        baseline,
        on=["seed", "repeat"],
        how="left",
        validate="many_to_one",
    )
    frame["delta_p_raw_vs_etkdg"] = frame["p_raw"] - frame["baseline_p_raw"]
    frame["delta_p_do_vs_etkdg"] = frame["p_do"] - frame["baseline_p_do"]
    frame["delta_pcs_vs_etkdg"] = frame["pcs"] - frame["baseline_pcs"]
    frame["raw_hce"] = (
        frame["label"].eq(0) & frame["p_raw"].ge(args.confidence_threshold)
    )
    frame["ppce"] = frame["raw_hce"] & frame["pcs"].gt(args.pcs_margin)
    return frame


def repeatability_summary(per_repeat: pd.DataFrame) -> pd.DataFrame:
    """Summarize fixed-input inference variation without pooling seeds."""
    rows = []
    for (seed, structure), group in per_repeat.groupby(
        ["seed", "structure"], sort=False
    ):
        row = {
            "seed": int(seed),
            "structure": structure,
            "repeat_count": int(len(group)),
            "raw_hce_agreement": bool(group["raw_hce"].nunique() == 1),
            "ppce_agreement": bool(group["ppce"].nunique() == 1),
        }
        for column in ("p_raw", "u_raw", "p_do", "u_do", "pcs"):
            values = group[column].astype(float)
            row[column + "_mean"] = float(values.mean())
            row[column + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[column + "_range"] = float(values.max() - values.min())
        rows.append(row)
    return pd.DataFrame(rows)


def ensemble_summary(args: argparse.Namespace, per_seed: pd.DataFrame) -> pd.DataFrame:
    seed_level = (
        per_seed.groupby(["seed", "structure"], sort=False, as_index=False)
        .agg(
            label=("label", "first"),
            p_raw=("p_raw", "mean"),
            p_do=("p_do", "mean"),
            pcs=("pcs", "mean"),
        )
    )
    seed_level["ppce"] = (
        seed_level["label"].eq(0)
        & seed_level["p_raw"].ge(args.confidence_threshold)
        & seed_level["pcs"].gt(args.pcs_margin)
    )
    rows = []
    for structure, group in seed_level.groupby("structure", sort=False):
        p_raw = float(group["p_raw"].mean())
        p_do = float(group["p_do"].mean())
        pcs = p_raw - p_do
        raw_hce = int(group.iloc[0]["label"]) == 0 and p_raw >= args.confidence_threshold
        rows.append(
            {
                "structure": structure,
                "seed_count": int(group["seed"].nunique()),
                "prediction_repeat_count": int(args.prediction_repeats),
                "p_raw_mean": p_raw,
                "p_raw_std": float(group["p_raw"].std(ddof=1)),
                "p_do_mean": p_do,
                "p_do_std": float(group["p_do"].std(ddof=1)),
                "pcs_from_mean_probabilities": pcs,
                "raw_hce": bool(raw_hce),
                "ensemble_ppce": bool(raw_hce and pcs > args.pcs_margin),
                "all_seed_ppce": bool(group["ppce"].all()),
            }
        )
    return pd.DataFrame(rows)


def build_claim_summary(
    per_seed: pd.DataFrame,
    ensemble: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize existence and robustness claims for experimental coordinates."""
    experimental_seed = per_seed[~per_seed["structure"].eq("ETKDG")].copy()
    experimental_ensemble = ensemble[
        ~ensemble["structure"].eq("ETKDG")
    ].copy()
    if not len(experimental_ensemble):
        raise ValueError("No experimental-coordinate predictions were produced.")
    return pd.DataFrame(
        [
            {
                "experimental_structure_count": int(len(experimental_ensemble)),
                "seed_count": int(experimental_seed["seed"].nunique()),
                "ensemble_ppce_structure_count": int(
                    experimental_ensemble["ensemble_ppce"].sum()
                ),
                "all_seed_ppce_structure_count": int(
                    experimental_ensemble["all_seed_ppce"].sum()
                ),
                "experimental_coordinate_ppce_exists": bool(
                    experimental_ensemble["ensemble_ppce"].any()
                ),
                "ppce_retained_for_all_experimental_structures": bool(
                    experimental_ensemble["ensemble_ppce"].all()
                ),
                "ppce_retained_in_every_seed_and_structure": bool(
                    experimental_ensemble["all_seed_ppce"].all()
                ),
                "max_abs_delta_p_raw_vs_etkdg": float(
                    experimental_seed["delta_p_raw_vs_etkdg"].abs().max()
                ),
                "max_abs_delta_p_do_vs_etkdg": float(
                    experimental_seed["delta_p_do_vs_etkdg"].abs().max()
                ),
                "max_abs_delta_pcs_vs_etkdg": float(
                    experimental_seed["delta_pcs_vs_etkdg"].abs().max()
                ),
                "claim_scope": (
                    "experimentally observed ligand-coordinate case evidence; "
                    "not a population prevalence estimate"
                ),
            }
        ]
    )


def write_report(
    args: argparse.Namespace,
    feature_records: Sequence[Dict[str, object]],
    per_seed: pd.DataFrame,
    repeatability: pd.DataFrame,
    ensemble: pd.DataFrame,
    claims: pd.DataFrame,
    report_path: Path,
) -> None:
    experimental = ensemble[~ensemble["structure"].eq("ETKDG")]
    ensemble_robust = bool(experimental["ensemble_ppce"].all())
    strict_robust = bool(experimental["all_seed_ppce"].all())
    lines = [
        "# BioSNAP experimental-3D coordinate robustness",
        "",
        "Only aligned ligand coordinates were replaced; checkpoint, processed pair, protein input, 2D input, and causal settings were fixed.",
        "",
        "## Result",
        "",
        "- Experimental structures remain PPCE after three-seed ensembling: `%s`." % ensemble_robust,
        "- Every experimental structure remains PPCE in every seed: `%s`." % strict_robust,
        "- This is a single-case coordinate robustness test, not population-level proof that experimental 3D never matters.",
        "- Because the PDB entries support ligand-receptor binding while BioSNAP labels the pair negative, the case may represent a dataset false negative.",
        "",
        "## Claim summary",
        "",
        claims.to_csv(index=False).strip(),
        "",
        "## Ensemble summary",
        "",
        ensemble.to_csv(index=False).strip(),
        "",
        "## Fixed-input repeatability",
        "",
        "Each row keeps checkpoint and feature tensors fixed; variation is numerical inference variation rather than a coordinate intervention.",
        "",
        repeatability.to_csv(index=False).strip(),
        "",
        "## Coordinate mapping",
        "",
        pd.DataFrame(feature_records).to_csv(index=False).strip(),
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.prediction_repeats < 1:
        raise ValueError("--prediction-repeats must be at least 1.")
    test_csv = resolve(args.test_csv)
    case = load_case(test_csv, args.drug_id, args.target_id)
    named_sdfs = parse_named_paths(args.experimental_sdf)

    case_dir = resolve("data/verified3d/cases")
    case_dir.mkdir(parents=True, exist_ok=True)
    case_csv = case_dir / (args.output_prefix + ".csv")
    case.to_csv(case_csv, index=False)

    feature_paths, feature_records = build_feature_banks(args, case, named_sdfs)
    feature_summary_path = resolve(
        "outputs/metrics/%s_coordinate_mapping.csv" % args.output_prefix
    )
    feature_summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(feature_records).to_csv(feature_summary_path, index=False)
    print("Built aligned coordinate-only feature banks:")
    print(pd.DataFrame(feature_records).to_string(index=False))

    if args.skip_prediction:
        print("Prediction skipped.")
        return

    case_feature_banks = build_case_input_banks(args, case)
    print(
        "Case-local inference banks:",
        {key: str(path.relative_to(ROOT)) for key, path in case_feature_banks.items()},
    )
    per_seed = run_predictions(
        args,
        case_csv,
        feature_paths,
        case_feature_banks,
    )
    repeatability = repeatability_summary(per_seed)
    ensemble = ensemble_summary(args, per_seed)
    claims = build_claim_summary(per_seed, ensemble)
    per_seed_path = resolve("outputs/metrics/%s_per_seed.csv" % args.output_prefix)
    repeatability_path = resolve(
        "outputs/metrics/%s_repeatability.csv" % args.output_prefix
    )
    ensemble_path = resolve("outputs/metrics/%s_ensemble.csv" % args.output_prefix)
    claim_path = resolve("outputs/metrics/%s_claim_summary.csv" % args.output_prefix)
    report_path = resolve("docs/%s_report.md" % args.output_prefix)
    per_seed.to_csv(per_seed_path, index=False)
    repeatability.to_csv(repeatability_path, index=False)
    ensemble.to_csv(ensemble_path, index=False)
    claims.to_csv(claim_path, index=False)
    write_report(
        args,
        feature_records,
        per_seed,
        repeatability,
        ensemble,
        claims,
        report_path,
    )

    print("\nPer-seed paired predictions:")
    print(per_seed.to_string(index=False))
    print("\nFixed-input repeatability:")
    print(repeatability.to_string(index=False))
    print("\nThree-seed ensemble:")
    print(ensemble.to_string(index=False))
    print("\nExperimental-coordinate claim summary:")
    print(claims.to_string(index=False))
    print("\nSaved:")
    for path in (
        feature_summary_path,
        per_seed_path,
        repeatability_path,
        ensemble_path,
        claim_path,
        report_path,
    ):
        print(path)


if __name__ == "__main__":
    main()

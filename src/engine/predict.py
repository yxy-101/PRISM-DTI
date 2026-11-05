#!/usr/bin/env python
"""Checkpoint prediction and raw-versus-adjusted CSV export for PRISM-DTI."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.collate import collate_dti_batch
from src.data.dataset import DTIDataset
from src.data.feature_io import IDFeatureStore
from src.models.prism_dti import PRISMDTI
from src.utils.evidential_audit import derive_evidential_contrasts
from src.utils.input_validation import validate_real_inputs
from src.data.label_prior import training_prior_groups


def parse_args() -> argparse.Namespace:
    """Parse checkpoint prediction options."""
    parser = argparse.ArgumentParser(description="PRISM-DTI checkpoint prediction")
    parser.add_argument(
        "--config",
        required=True,
        help="Final member config in configs/final/.",
    )
    parser.add_argument("--checkpoint", required=True, help="Trained checkpoint path.")
    parser.add_argument(
        "--output-prediction",
        default=None,
        help=(
            "Optional prediction CSV path override. This preserves the config "
            "and is useful for non-destructive frozen-checkpoint audits."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Load a configured checkpoint and export paired branch predictions."""
    args = parse_args()
    if args.config:
        config = load_config(args.config)
        validate_real_inputs(config, prediction=True)
        model = PRISMDTI.from_config(config)
        if args.checkpoint:
            load_checkpoint(model, args.checkpoint)
        print(
            json.dumps(
                run_smoke_prediction_export(
                    model=model,
                    config=config,
                    config_path=args.config,
                    checkpoint_path=args.checkpoint,
                    output_prediction_override=args.output_prediction,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return

    raise ValueError("Prediction requires a final member config and checkpoint.")


def run_smoke_prediction_export(
    model: torch.nn.Module,
    config: Dict[str, Any],
    config_path: str,
    checkpoint_path: Optional[str],
    output_prediction_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Export raw-vs-causal prediction CSV for a configured smoke dataset."""
    device = _resolve_device(str(config.get("trainer", {}).get("device", "cpu")))
    valid_csv = config.get("valid_csv") or config.get("train_csv")
    if not valid_csv:
        raise ValueError("Smoke prediction config requires valid_csv or train_csv.")
    dataset = DTIDataset(
        csv_path=_resolve_repo_path(valid_csv),
        max_samples=config.get("max_valid_samples"),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 32)),
        shuffle=False,
        collate_fn=collate_dti_batch,
    )
    feature_store = build_feature_store(config)
    prior_groups = load_target_prior_groups(config)
    output_prediction_path = resolve_output_prediction_path(
        config=config,
        input_csv=valid_csv,
        override=output_prediction_override,
    )
    output_path = _resolve_repo_path(output_prediction_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.to(device)
    model.eval()
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for records in loader:
            batch = feature_store.build_model_batch(records, device=device)
            outputs = model(batch)
            contrasts = derive_evidential_contrasts(outputs)
            p_raw = outputs["p_raw"][:, -1].detach().cpu().tolist()
            u_raw = outputs["u_raw"].reshape(-1).detach().cpu().tolist()
            p_do = (
                outputs.get("p_do", outputs["p_raw"])[:, -1]
                .detach()
                .cpu()
                .tolist()
            )
            u_do_tensor = outputs.get("u_do", outputs["u_raw"])
            u_do = u_do_tensor.reshape(-1).detach().cpu().tolist()
            pcs_tensor = outputs.get("pcs")
            if pcs_tensor is None:
                pcs = [raw - do for raw, do in zip(p_raw, p_do)]
            else:
                pcs = pcs_tensor.reshape(-1).detach().cpu().tolist()
            evidence_raw = contrasts["evidence_raw"].detach().cpu().tolist()
            evidence_do = contrasts["evidence_do"].detach().cpu().tolist()
            delta_u = contrasts["delta_uncertainty"].detach().cpu().tolist()
            delta_log_evidence = (
                contrasts["delta_log_evidence"].detach().cpu().tolist()
            )

            for index, drug_id in enumerate(records["drug_id"]):
                target_id = str(records["target_id"][index])
                row = {
                    "drug_id": drug_id,
                    "target_id": target_id,
                    "label": int(records["label"][index]),
                    "p_raw": "%.8f" % p_raw[index],
                    "u_raw": "%.8f" % u_raw[index],
                    "p_do": "%.8f" % p_do[index],
                    "u_do": "%.8f" % u_do[index],
                    "pcs": "%.8f" % pcs[index],
                    "evidence_raw": "%.8f" % evidence_raw[index],
                    "evidence_do": "%.8f" % evidence_do[index],
                    "delta_u": "%.8f" % delta_u[index],
                    "delta_log_evidence": "%.8f" % delta_log_evidence[index],
                    "target_prior_group": prior_groups.get(target_id, ""),
                }
                if "scaffold_id" in records:
                    row["scaffold_id"] = str(records["scaffold_id"][index])
                rows.append(row)

    columns = list(prediction_columns(include_scaffold=_has_scaffold(rows)))
    write_prediction_csv(output_path=output_path, rows=rows, columns=columns)
    return {
        "status": "ok",
        "mode": "csv_smoke_prediction",
        "config": config_path,
        "checkpoint": checkpoint_path,
        "input_csv": valid_csv,
        "num_rows": len(rows),
        "device": str(device),
        "output_csv": str(output_path),
        "columns": columns,
    }


def resolve_output_prediction_path(
    config: Dict[str, Any],
    input_csv: str,
    override: Optional[str] = None,
) -> str:
    """Resolve scalar or {valid,test} prediction output config."""
    if override:
        return str(override)
    configured = config.get("output_prediction_path")
    if not configured:
        return "outputs/predictions/bindingdb_smoke_predictions.csv"
    if isinstance(configured, dict):
        split_name = _split_name_from_csv(input_csv)
        if split_name in configured:
            return str(configured[split_name])
        if "valid" in configured:
            return str(configured["valid"])
        raise ValueError(
            "output_prediction_path mapping must contain the split %r or a valid key."
            % split_name
        )
    return str(configured)


def _split_name_from_csv(path: str) -> str:
    """Infer valid/test from the CSV path used by prediction."""
    stem = Path(path).stem.lower()
    if stem.endswith("_test") or "test" in stem:
        return "test"
    return "valid"


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML or JSON config."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError("Config file not found: %s" % config_path)
    if config_path.suffix.lower() == ".json":
        return json.loads(config_path.read_text(encoding="utf-8"))
    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required for smoke prediction configs.") from exc
    return yaml.safe_load(text) or {}


def build_feature_store(config: Dict[str, Any]) -> IDFeatureStore:
    """Create an ID-addressable feature loader from config."""
    model_config = config.get("model", {})
    dummy_config = config.get("dummy_features", {})
    feature_config = config.get("features", {})
    simple_drug_features = config.get(
        "allow_simple_drug_features",
        feature_config.get("allow_simple_drug_features", False),
    )
    simple_drug3d_features = config.get(
        "allow_simple_drug3d_features",
        feature_config.get("allow_simple_drug3d_features"),
    )
    use_drug3d = model_config.get(
        "use_drug3d",
        feature_config.get("use_drug3d", config.get("use_drug3d", True)),
    )
    return IDFeatureStore(
        protein_feature_path=config.get(
            "protein_feature_path",
            feature_config.get("protein_feature_path"),
        ),
        drug2d_feature_path=config.get(
            "drug2d_feature_path",
            feature_config.get("drug2d_feature_path"),
        ),
        drug3d_feature_path=config.get(
            "drug3d_feature_path",
            feature_config.get("drug3d_feature_path"),
        ),
        protein_dir=feature_config.get("protein_dir"),
        drug2d_dir=feature_config.get("drug2d_dir"),
        drug3d_dir=feature_config.get("drug3d_dir"),
        allow_dummy_features=bool(config.get("allow_dummy_features", False)),
        allow_simple_drug_features=_as_bool(simple_drug_features),
        allow_simple_drug3d_features=None
        if simple_drug3d_features is None
        else _as_bool(simple_drug3d_features),
        strict_feature_coverage=_as_bool(
            config.get(
                "strict_feature_coverage",
                feature_config.get("strict_feature_coverage", False),
            )
        ),
        use_drug3d=_as_bool(use_drug3d),
        protein_shape=(
            int(dummy_config.get("protein_seq_len", 32)),
            int(model_config.get("protein_input_dim", 1024)),
        ),
        drug2d_shape=(
            int(dummy_config.get("drug2d_seq_len", 24)),
            int(model_config.get("drug2d_input_dim", 768)),
        ),
        drug3d_shape=(
            int(dummy_config.get("drug3d_num_nodes", 16)),
            int(model_config.get("drug3d_input_dim", 300)),
        ),
    )


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> None:
    """Load a checkpoint, allowing only absent parameters of disabled mixers."""
    loaded = torch.load(str(Path(checkpoint_path).expanduser()), map_location="cpu")
    if isinstance(loaded, dict):
        state_dict = loaded.get("model_state_dict", loaded.get("state_dict", loaded))
    else:
        state_dict = loaded
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    allowed_missing = _disabled_feature_mixer_state_keys(model)
    disallowed_missing = missing - allowed_missing
    if disallowed_missing or unexpected:
        details = []
        if disallowed_missing:
            details.append("missing=%s" % sorted(disallowed_missing))
        if unexpected:
            details.append("unexpected=%s" % sorted(unexpected))
        raise RuntimeError(
            "Checkpoint is incompatible with the active model: %s"
            % "; ".join(details)
        )
    if missing:
        print(
            "checkpoint_compatibility: ignored %d absent parameter(s) from "
            "disabled feature mixers; these parameters are not executed"
            % len(missing)
        )


def _disabled_feature_mixer_state_keys(model: torch.nn.Module) -> set:
    """Return state keys that belong exclusively to disabled optional mixers."""
    prefixes = []
    for encoder_name in ("protein_encoder", "drug2d_encoder"):
        encoder = getattr(model, encoder_name, None)
        mixer = getattr(encoder, "feature_mixer", None)
        if mixer is not None and bool(getattr(mixer, "disabled", False)):
            prefixes.append("%s.feature_mixer." % encoder_name)
    return {
        key
        for key in model.state_dict()
        if any(key.startswith(prefix) for prefix in prefixes)
    }


def load_target_prior_groups(config: Dict[str, Any]) -> Dict[str, str]:
    """Use unsmoothed training target rate >=0.7, as in manuscript Eq.21."""
    if not config.get("train_csv"):
        raise ValueError("Prediction requires train_csv to identify training-only target priors.")
    return training_prior_groups(_resolve_repo_path(config["train_csv"]))


def write_prediction_csv(
    output_path: Path,
    rows: List[Dict[str, Any]],
    columns: Optional[List[str]] = None,
) -> None:
    """Write prediction rows to CSV."""
    fieldnames = columns or list(prediction_columns())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prediction_columns(include_scaffold: bool = False) -> List[str]:
    """Return the BindingDB smoke prediction CSV columns."""
    columns = [
        "drug_id",
        "target_id",
        "label",
        "p_raw",
        "u_raw",
        "p_do",
        "u_do",
        "pcs",
        "evidence_raw",
        "evidence_do",
        "delta_u",
        "delta_log_evidence",
        "target_prior_group",
    ]
    if include_scaffold:
        columns.append("scaffold_id")
    return columns


def _has_scaffold(rows: List[Dict[str, Any]]) -> bool:
    return bool(rows and "scaffold_id" in rows[0])


def dataset_name_from_path(path: str) -> str:
    """Infer dataset name from processed CSV path."""
    stem = Path(path).stem
    for suffix in ("_train", "_valid", "_test"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _resolve_device(requested_device: str) -> torch.device:
    """Resolve CUDA requests with CPU fallback."""
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    return torch.device(requested_device)


def _resolve_repo_path(path: str) -> Path:
    """Resolve config paths relative to the project root."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def _as_bool(value: Any) -> bool:
    """Parse simple bool-like config values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


if __name__ == "__main__":
    main()

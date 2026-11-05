#!/usr/bin/env python
"""Train or dry-run PRISM-DTI raw-vs-causal ablations."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MAX_LOSS_COMPONENT_HISTORY = 100

from src.data.collate import collate_dti_batch
from src.data.dataset import DTIDataset
from src.data.feature_io import IDFeatureStore
from src.data.grouped_batch_sampler import TargetGroupedBatchSampler
from src.data.target_prior_stress import attach_target_prior_stress_weights
from src.losses.evidential_loss import evidential_loss
from src.losses.raw_anchor import raw_anchor_loss
from src.losses.adjusted_protection import adjusted_protection_loss
from src.losses.poer import (
    attach_target_prior_groups,
    poer_total_loss,
    target_prior_groups_to_tensor,
)
from src.models.prism_dti import PRISMDTI
from src.models.poer_adversary import POERPriorAdversary
from src.utils.metrics import auprc, auroc
from src.utils.seed import set_seed
from src.utils.input_validation import validate_real_inputs


def parse_args() -> argparse.Namespace:
    """Parse training arguments."""
    parser = argparse.ArgumentParser(description="PRISM-DTI ablation trainer")
    parser.add_argument(
        "--config",
        "--cfg",
        dest="config",
        required=True,
        help="Path to the model or ablation config file.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Load config, instantiate the model, run a dummy forward, and exit.",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML or JSON config with a small YAML fallback.

    TODO: Move config parsing into `src/utils` when more scripts need it.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError("Config file not found: %s" % config_path)

    if config_path.suffix.lower() == ".json":
        return json.loads(config_path.read_text(encoding="utf-8"))

    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError:
        return _parse_simple_yaml(text)
    loaded = yaml.safe_load(text)
    return loaded or {}


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    """Parse the tiny YAML subset used by the default raw-backbone config."""
    root = {}
    stack = [(-1, root)]

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))
        key, separator, value = line.lstrip().partition(":")
        if separator != ":":
            raise ValueError("Invalid config line: %s" % raw_line)

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()

        container = stack[-1][1]
        key = key.strip()
        value = value.strip()
        if not value:
            child = {}
            container[key] = child
            stack.append((indent, child))
        else:
            container[key] = _parse_scalar(value)

    return root


def _parse_scalar(value: str) -> Any:
    """Convert a simple YAML scalar into a Python value."""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def main() -> None:
    """Parse config and run dry-run or dummy ablation training."""
    args = parse_args()
    config = load_config(args.config)
    if not args.dry_run:
        validate_real_inputs(config)
    set_seed(int(config.get("seed", 7)))
    model = PRISMDTI.from_config(config)
    initialization_summary = _initialize_model_from_checkpoint_if_configured(
        model=model,
        config=config,
    )

    if args.dry_run:
        model.eval()
        with torch.no_grad():
            outputs = model()
        result = {
            "status": "dry_run_ok",
            "config": args.config,
            "output_names": sorted(outputs),
            "output_shapes": _tensor_shapes(outputs),
            "module_switches": model.module_switches()
            if hasattr(model, "module_switches")
            else {},
            "required_external_features": model.required_external_features(),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if config.get("train_csv"):
        print(
            json.dumps(
                run_csv_smoke_training(
                    model=model,
                    config=config,
                    config_path=args.config,
                    initialization_summary=initialization_summary,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return

    raise ValueError("Training requires train_csv; use --dry_run only for synthetic model checks.")


def _tensor_shapes(outputs: Dict[str, Any]) -> Dict[str, list]:
    """Return tensor shapes as plain lists for JSON output."""
    return {
        key: list(value.shape)
        for key, value in outputs.items()
        if torch.is_tensor(value)
    }


def run_csv_smoke_training(
    model: torch.nn.Module,
    config: Dict[str, Any],
    config_path: str,
    initialization_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a tiny real-CSV smoke training loop with loaded or dummy features."""
    trainer_config = config.get("trainer", {})
    device = _resolve_device(
        str(trainer_config.get("device", config.get("device", "cpu")))
    )
    batch_size = int(config.get("batch_size", trainer_config.get("batch_size", 32)))
    epochs = int(config.get("epochs", trainer_config.get("epochs", 1)))
    max_steps = int(config.get("max_steps", trainer_config.get("max_steps", 5)))
    learning_rate = float(
        trainer_config.get("learning_rate", config.get("learning_rate", 1e-3))
    )
    optimizer_config = _resolve_optimizer_config(config)
    scheduler_config = _resolve_scheduler_config(config, epochs=epochs)
    monitor_config = _resolve_performance_monitor_config(config)
    loss_config = _resolve_loss_config(config)
    raw_anchor_config = _resolve_raw_anchor_config(config)
    grouped_batching_config = _resolve_target_grouped_batching_config(config)
    target_ranking_auxiliary_config = (
        _resolve_target_ranking_auxiliary_config(config)
    )
    if (
        grouped_batching_config["enabled"]
        and loss_config["ranking_scope"] != "within_target"
    ):
        raise ValueError(
            "target_grouped_batching requires loss.ranking_scope=within_target."
        )
    protein_consistency_config = _resolve_protein_consistency_config(config)
    branch_specific_randomization = _as_bool(
        config.get("branch_specific_randomization", {}).get("enabled", False)
    )
    if protein_consistency_config["enabled"] and not _use_target_randomization(config):
        raise ValueError(
            "protein_consistency requires target randomization to be enabled."
        )
    if branch_specific_randomization and protein_consistency_config["enabled"]:
        raise ValueError(
            "branch_specific_randomization and protein_consistency are mutually exclusive."
        )
    if branch_specific_randomization and not _use_target_randomization(config):
        raise ValueError(
            "branch_specific_randomization requires target randomization."
        )

    train_csv_path = _resolve_repo_path(config["train_csv"])
    train_dataset = DTIDataset(
        csv_path=train_csv_path,
        max_samples=config.get("max_train_samples"),
    )
    prior_balancing_summary = apply_prior_balancing_if_enabled(
        train_dataset=train_dataset,
        config=config,
        train_csv_path=train_csv_path,
    )
    target_prior_stress_summary = apply_target_prior_stress_if_enabled(
        train_dataset=train_dataset,
        config=config,
        train_csv_path=train_csv_path,
    )
    poer_state = prepare_poer_training_if_enabled(
        train_dataset=train_dataset,
        config=config,
        train_csv_path=train_csv_path,
    )
    valid_dataset = DTIDataset(
        csv_path=_resolve_repo_path(config["valid_csv"]),
        max_samples=config.get("max_valid_samples"),
    )
    feature_store = build_feature_store(config)
    if grouped_batching_config["enabled"]:
        grouped_batch_sampler = TargetGroupedBatchSampler(
            records=train_dataset.records,
            batch_size=batch_size,
            samples_per_target=grouped_batching_config[
                "samples_per_target"
            ],
            seed=grouped_batching_config["seed"],
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=grouped_batch_sampler,
            collate_fn=collate_dti_batch,
        )
        grouped_batching_summary = grouped_batch_sampler.summary()
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_dti_batch,
        )
        grouped_batching_summary = {"enabled": False}
    if target_ranking_auxiliary_config["enabled"]:
        auxiliary_batch_sampler = TargetGroupedBatchSampler(
            records=train_dataset.records,
            batch_size=batch_size,
            samples_per_target=target_ranking_auxiliary_config[
                "samples_per_target"
            ],
            seed=target_ranking_auxiliary_config["seed"],
        )
        auxiliary_loader = DataLoader(
            train_dataset,
            batch_sampler=auxiliary_batch_sampler,
            collate_fn=collate_dti_batch,
        )
        target_ranking_auxiliary_summary = {
            **target_ranking_auxiliary_config,
            "sampler": auxiliary_batch_sampler.summary(),
            "input_view": "clean",
            "main_batch_replaced": False,
        }
    else:
        auxiliary_loader = None
        target_ranking_auxiliary_summary = {"enabled": False}

    model.to(device)
    raw_anchor_reference = None
    if raw_anchor_config["enabled"]:
        if (
            raw_anchor_config["require_init_checkpoint"]
            and not (initialization_summary or {}).get("enabled", False)
        ):
            raise ValueError(
                "raw_anchor requires trainer.init_checkpoint so its reference "
                "is a trained pre-intervention model."
            )
        raw_anchor_reference = copy.deepcopy(model).to(device)
        raw_anchor_reference.eval()
        for parameter in raw_anchor_reference.parameters():
            parameter.requires_grad = False
    training_scope_summary = _apply_training_scope(
        model=model,
        trainer_config=trainer_config,
    )
    poer_adversary = None
    if poer_state["enabled"]:
        poer_adversary = POERPriorAdversary.from_config(config).to(device)
    model.train()
    if poer_adversary is not None:
        poer_adversary.train()
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("No trainable model parameters remain after applying scope.")
    if poer_adversary is not None:
        trainable_parameters.extend(
            parameter for parameter in poer_adversary.parameters() if parameter.requires_grad
        )
    optimizer = _build_optimizer(
        parameters=trainable_parameters,
        learning_rate=learning_rate,
        optimizer_config=optimizer_config,
    )
    scheduler = _build_scheduler(
        optimizer=optimizer,
        scheduler_config=scheduler_config,
    )
    gradient_clip_norm = float(trainer_config.get("gradient_clip_norm", 0.0))
    loss_history = []
    loss_component_history = []
    poer_loss_history = []
    validation_history = []
    learning_rate_history = []
    best_metric_value = None
    best_epoch = None
    best_step = None
    best_model_state = None
    output_keys = []
    output_shapes = {}
    step = 0

    if monitor_config["enabled"] and monitor_config["include_initial_state"]:
        metrics = _evaluate_validation_model(
            model=model,
            dataset=valid_dataset,
            feature_store=feature_store,
            batch_size=batch_size,
            device=device,
        )
        metrics.update({"epoch": -1, "step": 0, "initial_state": True})
        validation_history.append(metrics)
        metric_value = metrics.get(monitor_config["metric"])
        if metric_value is None:
            raise KeyError(
                "Initial validation did not produce monitored metric: %s"
                % monitor_config["metric"]
            )
        best_metric_value = float(metric_value)
        best_epoch = -1
        best_step = 0
        if monitor_config["save_best"]:
            best_model_state = _copy_state_dict_to_cpu(model)
        model.train()

    for epoch in range(epochs):
        auxiliary_iterator = (
            iter(auxiliary_loader) if auxiliary_loader is not None else None
        )
        for records in train_loader:
            batch = feature_store.build_model_batch(records, device=device)
            if branch_specific_randomization:
                if step == 0:
                    print(
                        "[debug] branch_specific_randomization=True "
                        "raw_view=clean causal_view=randomized"
                    )
            elif (
                _use_target_randomization(config)
                and not protein_consistency_config["enabled"]
            ):
                before_randomization = batch["protein_features"].detach().clone()
                batch["protein_features"] = _randomize_protein_features(
                    protein_features=batch["protein_features"],
                    randomization_config=config.get("randomization", {}),
                    step=step,
                )
                if step == 0:
                    diff = (before_randomization.float() - batch["protein_features"].float()).abs().sum().detach().cpu().item()
                    print(f"[debug] target_randomization_enabled=True protein_feature_abs_diff={diff}")
            elif not protein_consistency_config["enabled"]:
                if step == 0:
                    print("[debug] target_randomization_enabled=False")
            elif step == 0:
                print(
                    "[debug] protein_consistency_enabled=True "
                    "clean_supervision=True augmented_view=True"
                )
            labels = torch.as_tensor(records["label"], dtype=torch.long, device=device)
            sample_weights = _sample_weights_from_records(records, device=device)

            optimizer.zero_grad()
            if branch_specific_randomization:
                clean_outputs = model(batch)
                randomized_batch = dict(batch)
                randomized_batch["protein_features"] = _randomize_protein_features(
                    protein_features=batch["protein_features"],
                    randomization_config=config.get("randomization", {}),
                    step=step,
                )
                randomized_outputs = model(randomized_batch)
                outputs = _merge_branch_specific_outputs(
                    clean=clean_outputs,
                    randomized=randomized_outputs,
                )
            else:
                outputs = model(batch)
            original_loss, branch_components = _combined_supervised_loss(
                outputs=outputs,
                labels=labels,
                sample_weights=sample_weights,
                loss_config=loss_config,
                epoch=epoch,
                ranking_group_ids=records.get(
                    loss_config["ranking_group_column"]
                ),
            )
            loss = original_loss
            protection = config.get("adjusted_protection", {})
            if protection.get("enabled", False):
                weight = float(protection.get("weight", 1.0))
                if not (0 < weight < float("inf")):
                    raise ValueError("adjusted_protection weight must be finite and positive")
                loss = loss + weight * adjusted_protection_loss(outputs, labels)
            raw_anchor_components: Dict[str, Any] = {"enabled": False}
            if poer_adversary is not None:
                target_prior_group = target_prior_groups_to_tensor(
                    records[poer_state["prior_target"]],
                    device=device,
                )
                poer_losses = poer_total_loss(
                    original_loss=original_loss,
                    outputs=outputs,
                    labels=labels,
                    target_prior_group=target_prior_group,
                    prior_adversary=poer_adversary,
                    lambda_adv=poer_state["lambda_adv"],
                    lambda_ppce=poer_state["lambda_ppce"],
                    tau=poer_state["tau"],
                    confidence_features=poer_state["confidence_features"],
                    grl_scale=_poer_grl_scale(
                        epoch=epoch,
                        warmup_epochs=poer_state["adv_warmup_epochs"],
                    ),
                )
                loss = poer_losses["loss_total"]
                poer_loss_history.append(
                    {
                        "step": step,
                        "loss_prior_adv": round(
                            float(poer_losses["loss_prior_adv"].detach().cpu().item()),
                            6,
                        ),
                        "loss_ppce_proxy": round(
                            float(poer_losses["loss_ppce_proxy"].detach().cpu().item()),
                            6,
                        ),
                        "z_conf_shape": list(poer_losses["z_conf"].shape),
                    }
                )
            if raw_anchor_reference is not None:
                model.train(False)
                anchored_outputs = model(batch)
                model.train(True)
                with torch.no_grad():
                    reference_outputs = raw_anchor_reference(batch)
                anchor_losses = raw_anchor_loss(
                    outputs=anchored_outputs,
                    reference_outputs=reference_outputs,
                    probability_weight=raw_anchor_config["probability_weight"],
                    uncertainty_weight=raw_anchor_config["uncertainty_weight"],
                    evidence_weight=raw_anchor_config["evidence_weight"],
                )
                weighted_anchor = (
                    raw_anchor_config["weight"] * anchor_losses["loss_total"]
                )
                loss = loss + weighted_anchor
                raw_anchor_components = {
                    "enabled": True,
                    "weight": raw_anchor_config["weight"],
                    "weighted_loss": float(
                        weighted_anchor.detach().cpu().item()
                    ),
                    "probability": float(
                        anchor_losses["probability"].detach().cpu().item()
                    ),
                    "uncertainty": float(
                        anchor_losses["uncertainty"].detach().cpu().item()
                    ),
                    "log_evidence": float(
                        anchor_losses["log_evidence"].detach().cpu().item()
                    ),
                }
            consistency_components: Dict[str, Any] = {"enabled": False}
            if protein_consistency_config["enabled"]:
                augmented_batch = dict(batch)
                augmented_batch["protein_features"] = _randomize_protein_features(
                    protein_features=batch["protein_features"],
                    randomization_config=config.get("randomization", {}),
                    step=step,
                )
                augmented_outputs = model(augmented_batch)
                augmented_loss, augmented_components = _combined_supervised_loss(
                    outputs=augmented_outputs,
                    labels=labels,
                    sample_weights=sample_weights,
                    loss_config=loss_config,
                    epoch=epoch,
                    ranking_group_ids=records.get(
                        loss_config["ranking_group_column"]
                    ),
                )
                consistency_penalty, consistency_by_branch = (
                    _prediction_consistency_loss(outputs, augmented_outputs)
                )
                augmented_weight = float(
                    protein_consistency_config["augmented_supervision_weight"]
                )
                consistency_weight = float(protein_consistency_config["weight"])
                loss = (
                    loss
                    + augmented_weight * augmented_loss
                    + consistency_weight * consistency_penalty
                )
                consistency_components = {
                    "enabled": True,
                    "weight": consistency_weight,
                    "augmented_supervision_weight": augmented_weight,
                    "penalty": float(consistency_penalty.detach().cpu().item()),
                    "branch_penalties": consistency_by_branch,
                    "augmented_branches": augmented_components,
                }
            auxiliary_components: Dict[str, Any] = {"enabled": False}
            if auxiliary_iterator is not None:
                auxiliary_records = next(auxiliary_iterator)
                auxiliary_batch = feature_store.build_model_batch(
                    auxiliary_records,
                    device=device,
                )
                auxiliary_labels = torch.as_tensor(
                    auxiliary_records["label"],
                    dtype=torch.long,
                    device=device,
                )
                auxiliary_outputs = model(auxiliary_batch)
                auxiliary_loss, auxiliary_branch_losses = (
                    _target_ranking_auxiliary_loss(
                        outputs=auxiliary_outputs,
                        labels=auxiliary_labels,
                        target_ids=auxiliary_records["target_id"],
                        temperature=target_ranking_auxiliary_config[
                            "temperature"
                        ],
                        causal_weight=target_ranking_auxiliary_config[
                            "causal_weight"
                        ],
                    )
                )
                auxiliary_weight = target_ranking_auxiliary_config["weight"]
                loss = loss + auxiliary_weight * auxiliary_loss
                auxiliary_components = {
                    "enabled": True,
                    "weight": auxiliary_weight,
                    "unweighted_loss": float(
                        auxiliary_loss.detach().cpu().item()
                    ),
                    "weighted_loss": float(
                        (auxiliary_weight * auxiliary_loss)
                        .detach()
                        .cpu()
                        .item()
                    ),
                    "branch_losses": auxiliary_branch_losses,
                    "input_view": "clean",
                }
            loss.backward()
            if gradient_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    max_norm=gradient_clip_norm,
                )
            optimizer.step()

            loss_history.append(round(float(loss.detach().cpu().item()), 6))
            if len(loss_component_history) >= MAX_LOSS_COMPONENT_HISTORY:
                loss_component_history.pop(0)
            loss_component_history.append(
                {
                    "step": step,
                    "epoch": epoch,
                    "total_loss": round(float(loss.detach().cpu().item()), 6),
                    "base_loss": round(float(original_loss.detach().cpu().item()), 6),
                    "branches": branch_components,
                    "raw_anchor": raw_anchor_components,
                    "protein_consistency": consistency_components,
                    "target_ranking_auxiliary": auxiliary_components,
                }
            )
            output_keys = sorted(outputs)
            output_shapes = _tensor_shapes(outputs)
            step += 1
            if max_steps and step >= max_steps:
                break
        if monitor_config["enabled"] and _should_validate(epoch, monitor_config):
            metrics = _evaluate_validation_model(
                model=model,
                dataset=valid_dataset,
                feature_store=feature_store,
                batch_size=batch_size,
                device=device,
            )
            metrics.update({"epoch": epoch, "step": step})
            validation_history.append(metrics)
            metric_value = metrics.get(monitor_config["metric"])
            if _is_better_metric(
                metric_value=metric_value,
                best_metric_value=best_metric_value,
                mode=monitor_config["mode"],
            ):
                best_metric_value = float(metric_value)
                best_epoch = epoch
                best_step = step
                if monitor_config["save_best"]:
                    best_model_state = _copy_state_dict_to_cpu(model)
            model.train()
            if poer_adversary is not None:
                poer_adversary.train()
        learning_rate_history.append(
            {
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if scheduler is not None:
            scheduler.step()
        if max_steps and step >= max_steps:
            break

    experiment_name = str(config.get("experiment_name", "evi_tapb_dti"))
    configured_checkpoint_path = config.get("checkpoint_path")
    if configured_checkpoint_path:
        checkpoint_path = _resolve_repo_path(str(configured_checkpoint_path))
    else:
        checkpoint_path = ROOT / "outputs" / "checkpoints" / f"{experiment_name}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    selected_checkpoint = "last"
    model_state_dict = model.state_dict()
    if best_model_state is not None and monitor_config["save_best"]:
        selected_checkpoint = "best_%s" % monitor_config["metric"]
        model_state_dict = best_model_state
    checkpoint_state = {
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "config": config,
        "config_path": config_path,
        "step": step,
        "loss_history": loss_history,
        "loss_component_history": loss_component_history,
        "loss_config": loss_config,
        "raw_anchor": raw_anchor_config,
        "prior_balancing": prior_balancing_summary,
        "target_prior_stress": target_prior_stress_summary,
        "protein_consistency": protein_consistency_config,
        "target_grouped_batching": grouped_batching_summary,
        "target_ranking_auxiliary": target_ranking_auxiliary_summary,
        "optimizer_config": optimizer_config,
        "scheduler_config": scheduler_config,
        "learning_rate_history": learning_rate_history,
        "gradient_clip_norm": gradient_clip_norm,
        "performance_monitor": monitor_config,
        "initialization": initialization_summary or {"enabled": False},
        "training_scope": training_scope_summary,
        "validation_history": validation_history,
        "selected_checkpoint": selected_checkpoint,
        "best_metric_value": best_metric_value,
        "best_epoch": best_epoch,
        "best_step": best_step,
        "poer_loss_history": poer_loss_history,
        "poer": poer_state,
    }
    if poer_adversary is not None:
        checkpoint_state["poer_adversary_state_dict"] = poer_adversary.state_dict()
    torch.save(checkpoint_state, checkpoint_path)

    return {
        "status": "ok",
        "mode": "csv_smoke_training",
        "config": config_path,
        "checkpoint_path": str(checkpoint_path),
        "train_csv": config["train_csv"],
        "valid_csv": config.get("valid_csv"),
        "train_samples": len(train_dataset),
        "valid_samples": len(valid_dataset),
        "batch_size": batch_size,
        "epochs": epochs,
        "max_steps": max_steps,
        "device": str(device),
        "steps": step,
        "allow_dummy_features": bool(config.get("allow_dummy_features", False)),
        "loss_history": loss_history,
        "loss_component_history_tail": loss_component_history[-5:],
        "loss_config": loss_config,
        "raw_anchor": raw_anchor_config,
        "protein_consistency": protein_consistency_config,
        "target_grouped_batching": grouped_batching_summary,
        "target_ranking_auxiliary": target_ranking_auxiliary_summary,
        "optimizer_config": optimizer_config,
        "performance_monitor": monitor_config,
        "initialization": initialization_summary or {"enabled": False},
        "training_scope": training_scope_summary,
        "validation_history": validation_history,
        "selected_checkpoint": selected_checkpoint,
        "best_metric_value": best_metric_value,
        "best_epoch": best_epoch,
        "best_step": best_step,
        "output_keys": output_keys,
        "output_shapes": output_shapes,
        "module_switches": model.module_switches()
        if hasattr(model, "module_switches")
        else {},
        "prior_balancing": prior_balancing_summary,
        "target_prior_stress": target_prior_stress_summary,
        "poer": {
            **poer_state,
            "loss_history_tail": poer_loss_history[-5:],
        },
    }


def _initialize_model_from_checkpoint_if_configured(
    model: torch.nn.Module,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Initialize model weights from a configured checkpoint before training."""
    trainer_config = config.get("trainer", {})
    checkpoint_value = trainer_config.get(
        "init_checkpoint",
        config.get("init_checkpoint"),
    )
    if not checkpoint_value:
        return {"enabled": False}

    checkpoint_path = _resolve_repo_path(str(checkpoint_value))
    if not checkpoint_path.exists():
        raise FileNotFoundError("Initial checkpoint not found: %s" % checkpoint_path)
    loaded = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(loaded, dict):
        state_dict = loaded.get("model_state_dict", loaded.get("state_dict", loaded))
    else:
        state_dict = loaded
    if not isinstance(state_dict, dict):
        raise TypeError("Initial checkpoint does not contain a model state dictionary.")
    allowed_new_prefixes = tuple(
        str(prefix)
        for prefix in trainer_config.get("init_new_module_prefixes", [])
    )
    if allowed_new_prefixes:
        current_state = model.state_dict()
        source_keys = set(state_dict)
        current_keys = set(current_state)
        unexpected = sorted(source_keys - current_keys)
        missing = sorted(current_keys - source_keys)
        disallowed_missing = [
            key
            for key in missing
            if not key.startswith(allowed_new_prefixes)
        ]
        if unexpected or disallowed_missing:
            raise RuntimeError(
                "Controlled checkpoint initialization mismatch: "
                "unexpected=%s disallowed_missing=%s"
                % (unexpected, disallowed_missing)
            )
        current_state.update(state_dict)
        model.load_state_dict(current_state, strict=True)
    else:
        missing = []
        model.load_state_dict(state_dict, strict=True)
    return {
        "enabled": True,
        "checkpoint": str(checkpoint_path),
        "strict": True,
        "new_parameter_keys": missing,
        "allowed_new_prefixes": list(allowed_new_prefixes),
    }


def _apply_training_scope(
    model: torch.nn.Module,
    trainer_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Freeze model modules according to the configured fine-tuning scope."""
    scope = str(trainer_config.get("trainable_scope", "all")).strip().lower()
    for parameter in model.parameters():
        parameter.requires_grad = True

    if scope == "all":
        trainable_modules = ["all"]
    elif scope == "causal_only":
        for parameter in model.parameters():
            parameter.requires_grad = False
        trainable_modules = [
            "cam",
            "causal_fusion",
            "causal_evidential_head",
        ]
        for module_name in trainable_modules:
            module = getattr(model, module_name, None)
            if module is None:
                raise AttributeError(
                    "Model has no module required by causal_only scope: %s"
                    % module_name
                )
            for parameter in module.parameters():
                parameter.requires_grad = True
    elif scope == "conditioned_attention_only":
        for parameter in model.parameters():
            parameter.requires_grad = False
        trainable_modules = ["drug_conditioned_protein_attention"]
        module = getattr(model, trainable_modules[0], None)
        if module is None:
            raise AttributeError(
                "Model has no drug-conditioned protein attention module."
            )
        for parameter in module.parameters():
            parameter.requires_grad = True
    else:
        raise ValueError("Unsupported trainer.trainable_scope: %s" % scope)

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    frozen_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if not parameter.requires_grad
    )
    return {
        "scope": scope,
        "trainable_modules": trainable_modules,
        "trainable_parameter_count": trainable_parameter_count,
        "frozen_parameter_count": frozen_parameter_count,
    }


def prepare_poer_training_if_enabled(
    train_dataset: DTIDataset,
    config: Dict[str, Any],
    train_csv_path: Path,
) -> Dict[str, Any]:
    """Attach POER target-prior groups and return runtime settings."""
    poer_config = config.get("poer", {})
    enabled = _as_bool(poer_config.get("enabled", False))
    if not enabled:
        return {"enabled": False}

    prior_target = str(poer_config.get("prior_target", "target_prior_group"))
    prior_source_dataset = DTIDataset(csv_path=train_csv_path)
    prior_summary = attach_target_prior_groups(
        records=train_dataset.records,
        prior_target=prior_target,
        prior_records=prior_source_dataset.records,
    )
    confidence_features = list(
        poer_config.get(
            "confidence_features",
            ["logit_p_raw_positive", "u_raw", "pcs", "evidence_strength"],
        )
    )
    return {
        "enabled": True,
        "prior_target": prior_target,
        "lambda_adv": float(poer_config.get("lambda_adv", 0.01)),
        "lambda_ppce": float(poer_config.get("lambda_ppce", 0.05)),
        "adv_warmup_epochs": int(poer_config.get("adv_warmup_epochs", 2)),
        "tau": float(poer_config.get("tau", 0.8)),
        "confidence_features": confidence_features,
        "target_prior_group_summary": prior_summary,
    }


def _poer_grl_scale(epoch: int, warmup_epochs: int) -> float:
    """Linearly warm up the backbone-side adversarial reversal scale."""
    warmup_epochs = int(warmup_epochs)
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, float(epoch + 1) / float(warmup_epochs))


def build_feature_store(config: Dict[str, Any]) -> IDFeatureStore:
    """Create the feature store used by CSV smoke train/predict paths."""
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


def _resolve_optimizer_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize optimizer settings for CSV training."""
    trainer_config = config.get("trainer", {})
    optimizer_config = config.get("optimizer", {})
    if not isinstance(optimizer_config, dict):
        optimizer_config = {}
    name = str(optimizer_config.get("name", trainer_config.get("optimizer", "adam")))
    return {
        "name": name.strip().lower(),
        "weight_decay": float(
            optimizer_config.get(
                "weight_decay",
                trainer_config.get("weight_decay", 0.0),
            )
        ),
        "betas": tuple(optimizer_config.get("betas", (0.9, 0.999))),
    }


def _build_optimizer(
    parameters: List[torch.nn.Parameter],
    learning_rate: float,
    optimizer_config: Dict[str, Any],
) -> torch.optim.Optimizer:
    """Build Adam or AdamW from config."""
    optimizer_name = str(optimizer_config.get("name", "adam")).lower()
    weight_decay = float(optimizer_config.get("weight_decay", 0.0))
    betas = tuple(float(value) for value in optimizer_config.get("betas", (0.9, 0.999)))
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
        )
    if optimizer_name == "adam":
        return torch.optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
        )
    raise ValueError("Unsupported optimizer.name: %s" % optimizer_name)


def _resolve_scheduler_config(
    config: Dict[str, Any],
    epochs: int,
) -> Dict[str, Any]:
    """Normalize optional epoch-level learning-rate scheduling."""
    scheduler_config = config.get("scheduler", {})
    if not isinstance(scheduler_config, dict):
        scheduler_config = {}
    name = str(scheduler_config.get("name", "none")).strip().lower()
    return {
        "name": name,
        "epochs": max(1, int(epochs)),
        "min_learning_rate": float(
            scheduler_config.get("min_learning_rate", 1e-5)
        ),
    }


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_config: Dict[str, Any],
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """Build an optional scheduler without changing the legacy default."""
    name = str(scheduler_config.get("name", "none")).lower()
    if name in {"", "none", "off", "disabled"}:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(scheduler_config["epochs"])),
            eta_min=float(scheduler_config["min_learning_rate"]),
        )
    raise ValueError("Unsupported scheduler.name: %s" % name)


def _resolve_performance_monitor_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize validation-best checkpoint selection settings."""
    monitor_config = config.get("performance_monitor", {})
    if not isinstance(monitor_config, dict):
        monitor_config = {}
    metric = str(monitor_config.get("metric", "valid_raw_aupr")).strip()
    return {
        "enabled": _as_bool(monitor_config.get("enabled", False)),
        "metric": metric,
        "mode": str(monitor_config.get("mode", "max")).strip().lower(),
        "save_best": _as_bool(monitor_config.get("save_best", True)),
        "include_initial_state": _as_bool(
            monitor_config.get("include_initial_state", False)
        ),
        "interval_epochs": max(1, int(monitor_config.get("interval_epochs", 1))),
        "reason": (
            "select checkpoint by validation ranking metric instead of the last "
            "training epoch"
        ),
    }


def _should_validate(epoch: int, monitor_config: Dict[str, Any]) -> bool:
    """Return whether this epoch should run validation."""
    interval = max(1, int(monitor_config.get("interval_epochs", 1)))
    return (int(epoch) + 1) % interval == 0


def _evaluate_validation_model(
    model: torch.nn.Module,
    dataset: DTIDataset,
    feature_store: IDFeatureStore,
    batch_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate raw and causal branch ranking metrics on the validation CSV."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_dti_batch,
    )
    labels: List[int] = []
    raw_scores: List[float] = []
    do_scores: List[float] = []

    model.eval()
    with torch.no_grad():
        for records in loader:
            batch = feature_store.build_model_batch(records, device=device)
            outputs = model(batch)
            labels.extend(int(label) for label in records["label"])
            raw_scores.extend(outputs["p_raw"][:, -1].detach().cpu().tolist())
            if "p_do" in outputs:
                do_scores.extend(outputs["p_do"][:, -1].detach().cpu().tolist())

    raw_predictions = [1 if score >= 0.5 else 0 for score in raw_scores]
    raw_accuracy = _accuracy(labels, raw_predictions)
    metrics = {
        "valid_sample_count": len(labels),
        "valid_raw_auc": _none_to_nan(auroc(labels, raw_scores)),
        "valid_raw_aupr": _none_to_nan(auprc(labels, raw_scores)),
        "valid_raw_accuracy": raw_accuracy,
    }
    if do_scores:
        do_predictions = [1 if score >= 0.5 else 0 for score in do_scores]
        metrics.update(
            {
                "valid_do_auc": _none_to_nan(auroc(labels, do_scores)),
                "valid_do_aupr": _none_to_nan(auprc(labels, do_scores)),
                "valid_do_accuracy": _accuracy(labels, do_predictions),
            }
        )
    return metrics


def _is_better_metric(
    metric_value: Any,
    best_metric_value: Optional[float],
    mode: str,
) -> bool:
    """Return whether a validation metric improves the current best value."""
    if metric_value is None:
        return False
    try:
        current = float(metric_value)
    except (TypeError, ValueError):
        return False
    if current != current:
        return False
    if best_metric_value is None:
        return True
    if str(mode).lower() == "min":
        return current < best_metric_value
    return current > best_metric_value


def _copy_state_dict_to_cpu(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Copy model weights for best-checkpoint retention without GPU storage."""
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _accuracy(labels: List[int], predictions: List[int]) -> Optional[float]:
    """Compute binary accuracy for aligned label/prediction lists."""
    if not labels:
        return None
    correct = sum(int(label == pred) for label, pred in zip(labels, predictions))
    return float(correct) / float(len(labels))


def _none_to_nan(value: Optional[float]) -> Optional[float]:
    """Round-trip missing metrics as JSON null."""
    if value is None:
        return None
    return float(value)


def apply_prior_balancing_if_enabled(
    train_dataset: DTIDataset,
    config: Dict[str, Any],
    train_csv_path: Path,
) -> Dict[str, Any]:
    """Attach optional train-derived group weights to training records only."""
    balancing_config = config.get("prior_balancing", {})
    if not _as_bool(balancing_config.get("enabled", False)):
        return {"enabled": False}

    method = str(
        balancing_config.get("method", "inverse_target_label_frequency")
    ).lower()
    supported_methods = {
        "smoothed_target_label_propensity",
        "inverse_target_label_frequency",
        "inverse_target_frequency",
        "inverse_scaffold_frequency",
    }
    if method not in supported_methods:
        raise ValueError("Unsupported prior_balancing.method: %s" % method)

    prior_source_dataset = DTIDataset(csv_path=train_csv_path)
    frequency_power = float(balancing_config.get("frequency_power", 1.0))
    if frequency_power <= 0.0:
        raise ValueError("prior_balancing.frequency_power must be positive.")
    if method == "smoothed_target_label_propensity":
        from src.data.smoothed_prior_balance import compute_weights
        if _as_bool(balancing_config.get("normalize_mean", True)) or float(balancing_config.get("min_weight", .5)) != .5 or float(balancing_config.get("max_weight", 2.)) != 2.:
            raise ValueError("Smoothed propensity protocol requires normalize_mean=false, bounds=.5/2")
        raw_weights = compute_weights(train_dataset.records, prior_source_dataset.records,
            shuffled=_as_bool(balancing_config.get("shuffle_within_class", False)),
            seed=int(config.get("seed", 7)))
    elif method == "inverse_scaffold_frequency":
        raw_weights = _compute_inverse_group_frequency_weights(
            records=train_dataset.records,
            prior_records=prior_source_dataset.records,
            group_column="scaffold_id",
            frequency_power=frequency_power,
        )
    elif method == "inverse_target_frequency":
        raw_weights = _compute_inverse_target_frequency_weights(
            records=train_dataset.records,
            prior_records=prior_source_dataset.records,
            frequency_power=frequency_power,
        )
    else:
        raw_weights = _compute_inverse_target_label_frequency_weights(
            records=train_dataset.records,
            prior_records=prior_source_dataset.records,
        )
    if not raw_weights:
        return {"enabled": True, "method": method, "sample_count": 0}

    weights = list(raw_weights)
    if _as_bool(balancing_config.get("normalize_mean", True)):
        mean_weight = sum(weights) / len(weights)
        if mean_weight > 0:
            weights = [weight / mean_weight for weight in weights]

    min_weight = float(balancing_config.get("min_weight", 0.2))
    max_weight = float(balancing_config.get("max_weight", 5.0))
    if min_weight <= 0:
        raise ValueError("prior_balancing.min_weight must be positive.")
    if max_weight < min_weight:
        raise ValueError("prior_balancing.max_weight must be >= min_weight.")
    weights = [min(max(weight, min_weight), max_weight) for weight in weights]

    for record, weight in zip(train_dataset.records, weights):
        record["sample_weight"] = float(weight)

    return {
        "enabled": True,
        "method": method,
        "sample_count": len(weights),
        "prior_source": str(train_csv_path),
        "prior_source_sample_count": len(prior_source_dataset),
        "normalize_mean": _as_bool(balancing_config.get("normalize_mean", True)),
        "frequency_power": frequency_power,
        "min_weight": min_weight,
        "max_weight": max_weight,
        "weight_min": round(min(weights), 6),
        "weight_max": round(max(weights), 6),
        "weight_mean": round(sum(weights) / len(weights), 6),
    }


def apply_target_prior_stress_if_enabled(
    train_dataset: DTIDataset,
    config: Dict[str, Any],
    train_csv_path: Path,
) -> Dict[str, Any]:
    """Attach a controlled train-only target-prior stress intervention."""
    configured = config.get("target_prior_stress", {})
    if not isinstance(configured, dict):
        configured = {}
    if not _as_bool(configured.get("enabled", False)):
        return {"enabled": False}
    if _as_bool(config.get("prior_balancing", {}).get("enabled", False)):
        raise ValueError(
            "target_prior_stress and prior_balancing must be run as separate "
            "experiments so their effects remain identifiable."
        )

    prior_source_dataset = DTIDataset(csv_path=train_csv_path)
    summary = attach_target_prior_stress_weights(
        records=train_dataset.records,
        prior_records=prior_source_dataset.records,
        strength=float(configured.get("strength", 0.0)),
        smoothing=float(configured.get("smoothing", 10.0)),
        min_weight=float(configured.get("min_weight", 0.2)),
        max_weight=float(configured.get("max_weight", 5.0)),
        normalize_mean=_as_bool(configured.get("normalize_mean", True)),
        assignment_mode=str(configured.get("assignment_mode", "target_prior")),
        shuffle_seed=int(configured.get("shuffle_seed", config.get("seed", 7))),
    )
    summary["prior_source"] = str(train_csv_path)
    summary["prior_source_sample_count"] = len(prior_source_dataset)
    summary["validation_or_test_labels_used"] = False
    return summary


def _compute_inverse_target_label_frequency_weights(
    records: List[Dict[str, Any]],
    prior_records: List[Dict[str, Any]],
) -> List[float]:
    """Compute inverse P_train(label | target) weights from train records."""
    target_counts: Dict[str, int] = defaultdict(int)
    target_label_counts: Dict[tuple, int] = defaultdict(int)
    for record in prior_records:
        target_id = str(record["target_id"])
        label = int(record["label"])
        target_counts[target_id] += 1
        target_label_counts[(target_id, label)] += 1

    weights = []
    for record in records:
        target_id = str(record["target_id"])
        label = int(record["label"])
        total = target_counts[target_id]
        label_count = target_label_counts[(target_id, label)]
        if total <= 0 or label_count <= 0:
            weights.append(1.0)
            continue
        weights.append(float(total) / float(label_count))
    return weights


def _compute_inverse_target_frequency_weights(
    records: List[Dict[str, Any]],
    prior_records: List[Dict[str, Any]],
    frequency_power: float = 1.0,
) -> List[float]:
    """Downweight repeated targets without using validation or test labels."""
    return _compute_inverse_group_frequency_weights(
        records=records,
        prior_records=prior_records,
        group_column="target_id",
        frequency_power=frequency_power,
    )


def _compute_inverse_group_frequency_weights(
    records: List[Dict[str, Any]],
    prior_records: List[Dict[str, Any]],
    group_column: str,
    frequency_power: float = 1.0,
) -> List[float]:
    """Downweight repeated train-only groups such as targets or scaffolds."""
    target_counts: Dict[str, int] = defaultdict(int)
    for record in prior_records:
        if group_column not in record or not str(record[group_column]).strip():
            raise ValueError(
                "Training record is missing required group column: %s"
                % group_column
            )
        target_counts[str(record[group_column])] += 1

    power = float(frequency_power)
    weights = []
    for record in records:
        if group_column not in record or not str(record[group_column]).strip():
            raise ValueError(
                "Weighted record is missing required group column: %s"
                % group_column
            )
        count = target_counts[str(record[group_column])]
        if count <= 0:
            weights.append(1.0)
            continue
        weights.append(1.0 / (float(count) ** power))
    return weights


def _sample_weights_from_records(
    records: Dict[str, Any],
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Return a sample-weight tensor when the collated batch contains weights."""
    if "sample_weight" not in records:
        return None
    return torch.as_tensor(records["sample_weight"], dtype=torch.float32, device=device)


def _resolve_loss_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize loss settings while preserving the legacy NLL default."""
    trainer_config = config.get("trainer", {})
    configured = config.get("loss", {})
    if not isinstance(configured, dict):
        configured = {}
    loss_type = str(
        configured.get("type", configured.get("loss_type", config.get("loss_type", "nll")))
    ).strip().lower()
    if loss_type in {"edl", "evidential", "dirichlet"}:
        loss_type = "evidential"
    raw_loss_weight = float(
        configured.get("raw_loss_weight", trainer_config.get("raw_loss_weight", 1.0))
    )
    causal_loss_weight = float(
        configured.get(
            "causal_loss_weight", trainer_config.get("causal_loss_weight", 1.0)
        )
    )
    ranking_scope = str(configured.get("ranking_scope", "global")).strip().lower()
    if ranking_scope not in {"global", "within_target"}:
        raise ValueError(
            "loss.ranking_scope must be 'global' or 'within_target'."
        )
    return {
        "type": loss_type,
        "evidential_loss_type": str(
            configured.get("evidential_loss_type", configured.get("edl_type", "digamma"))
        ).strip().lower(),
        "raw_loss_weight": raw_loss_weight,
        "causal_loss_weight": causal_loss_weight,
        "kl_weight": float(configured.get("kl_weight", 0.0)),
        "kl_annealing_epochs": int(configured.get("kl_annealing_epochs", 1)),
        "uncertainty_regularizer_weight": float(
            configured.get("uncertainty_regularizer_weight", 0.0)
        ),
        "ranking_loss_weight": float(
            configured.get("ranking_loss_weight", 0.0)
        ),
        "ranking_temperature": float(
            configured.get("ranking_temperature", 0.2)
        ),
        "ranking_scope": ranking_scope,
        "ranking_group_column": str(
            configured.get("ranking_group_column", "target_id")
        ),
    }


def _resolve_raw_anchor_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize the optional frozen-reference raw-output anchor."""
    configured = config.get("raw_anchor", {})
    if not isinstance(configured, dict):
        configured = {}
    values = {
        "enabled": _as_bool(configured.get("enabled", False)),
        "weight": float(configured.get("weight", 0.1)),
        "probability_weight": float(
            configured.get("probability_weight", 1.0)
        ),
        "uncertainty_weight": float(
            configured.get("uncertainty_weight", 1.0)
        ),
        "evidence_weight": float(configured.get("evidence_weight", 0.0)),
        "require_init_checkpoint": _as_bool(
            configured.get("require_init_checkpoint", True)
        ),
        "reference": "frozen_pre_intervention_checkpoint",
    }
    numeric_keys = (
        "weight",
        "probability_weight",
        "uncertainty_weight",
        "evidence_weight",
    )
    if any(values[key] < 0.0 for key in numeric_keys):
        raise ValueError("raw_anchor weights must be non-negative")
    if values["enabled"] and values["weight"] <= 0.0:
        raise ValueError("enabled raw_anchor requires a positive weight")
    if values["enabled"] and (
        values["probability_weight"]
        + values["uncertainty_weight"]
        + values["evidence_weight"]
        <= 0.0
    ):
        raise ValueError(
            "enabled raw_anchor requires at least one positive component weight"
        )
    return values


def _resolve_target_grouped_batching_config(
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Normalize the optional train-only target-balanced batch sampler."""
    configured = config.get("target_grouped_batching", {})
    if not isinstance(configured, dict):
        configured = {}
    enabled = _as_bool(configured.get("enabled", False))
    samples_per_target = int(configured.get("samples_per_target", 4))
    if samples_per_target < 2:
        raise ValueError(
            "target_grouped_batching.samples_per_target must be at least 2."
        )
    return {
        "enabled": enabled,
        "samples_per_target": samples_per_target,
        "seed": int(configured.get("seed", config.get("seed", 7))),
    }


def _resolve_target_ranking_auxiliary_config(
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Normalize a clean paired batch used only for auxiliary ranking."""
    configured = config.get("target_ranking_auxiliary", {})
    if not isinstance(configured, dict):
        configured = {}
    weight = float(configured.get("weight", 0.03))
    causal_weight = float(configured.get("causal_weight", 0.2))
    temperature = float(configured.get("temperature", 0.2))
    samples_per_target = int(configured.get("samples_per_target", 4))
    if weight < 0.0 or causal_weight < 0.0:
        raise ValueError("target ranking auxiliary weights must be nonnegative.")
    if temperature <= 0.0:
        raise ValueError("target ranking auxiliary temperature must be positive.")
    if samples_per_target < 2:
        raise ValueError(
            "target ranking auxiliary samples_per_target must be at least 2."
        )
    return {
        "enabled": _as_bool(configured.get("enabled", False)),
        "weight": weight,
        "causal_weight": causal_weight,
        "temperature": temperature,
        "samples_per_target": samples_per_target,
        "seed": int(configured.get("seed", config.get("seed", 7))) + 100003,
    }


def _resolve_protein_consistency_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize the optional clean/augmented protein consistency objective."""
    configured = config.get("protein_consistency", {})
    if not isinstance(configured, dict):
        configured = {}
    weight = float(configured.get("weight", 0.05))
    augmented_weight = float(
        configured.get("augmented_supervision_weight", 0.5)
    )
    if weight < 0.0 or augmented_weight < 0.0:
        raise ValueError("protein_consistency weights must be nonnegative.")
    return {
        "enabled": _as_bool(configured.get("enabled", False)),
        "weight": weight,
        "augmented_supervision_weight": augmented_weight,
        "divergence": "symmetric_kl",
    }


def _combined_supervised_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    sample_weights: Optional[torch.Tensor],
    loss_config: Dict[str, Any],
    epoch: int,
    ranking_group_ids: Optional[Sequence[object]] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Return the configured raw plus optional causal supervised objective."""
    raw_loss, raw_components = _configured_branch_loss(
        outputs=outputs,
        branch="raw",
        labels=labels,
        sample_weights=sample_weights,
        loss_config=loss_config,
        epoch=epoch,
        ranking_group_ids=ranking_group_ids,
    )
    total = raw_loss
    components: Dict[str, Any] = {"raw": raw_components}
    if "p_do" in outputs:
        causal_loss, causal_components = _configured_branch_loss(
            outputs=outputs,
            branch="causal",
            labels=labels,
            sample_weights=sample_weights,
            loss_config=loss_config,
            epoch=epoch,
            ranking_group_ids=ranking_group_ids,
        )
        total = total + causal_loss
        components["causal"] = causal_components
    return total, components


def _prediction_consistency_loss(
    clean: Dict[str, torch.Tensor],
    augmented: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Penalize symmetric KL drift between clean and augmented predictions."""
    penalties = []
    values: Dict[str, float] = {}
    for key in ("p_raw", "p_do"):
        if key not in clean or key not in augmented:
            continue
        left = clean[key].clamp_min(1e-8)
        right = augmented[key].clamp_min(1e-8)
        left = left / left.sum(dim=-1, keepdim=True)
        right = right / right.sum(dim=-1, keepdim=True)
        left_to_right = torch.sum(
            left * (torch.log(left) - torch.log(right)), dim=-1
        ).mean()
        right_to_left = torch.sum(
            right * (torch.log(right) - torch.log(left)), dim=-1
        ).mean()
        penalty = 0.5 * (left_to_right + right_to_left)
        penalties.append(penalty)
        values[key] = float(penalty.detach().cpu().item())
    if not penalties:
        raise ValueError("Protein consistency requires p_raw or p_do outputs.")
    return torch.stack(penalties).mean(), values


def _merge_branch_specific_outputs(
    clean: Dict[str, torch.Tensor],
    randomized: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Use clean raw evidence and randomized causal evidence in one objective."""
    required_clean = {"alpha_raw", "p_raw", "u_raw"}
    required_randomized = {"alpha_ci", "alpha_do", "p_do", "u_do"}
    missing = sorted(
        (required_clean - set(clean))
        | (required_randomized - set(randomized))
    )
    if missing:
        raise KeyError("Missing branch-specific output keys: %s" % missing)
    merged = dict(randomized)
    for key in required_clean:
        merged[key] = clean[key]
    merged["pcs"] = merged["p_raw"][:, -1] - merged["p_do"][:, -1]
    return merged


def _configured_branch_loss(
    outputs: Dict[str, torch.Tensor],
    branch: str,
    labels: torch.Tensor,
    sample_weights: Optional[torch.Tensor],
    loss_config: Dict[str, Any],
    epoch: int,
    ranking_group_ids: Optional[Sequence[object]] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Compute NLL or EDL branch loss from the configured training objective."""
    is_raw = branch == "raw"
    branch_weight = float(
        loss_config["raw_loss_weight"] if is_raw else loss_config["causal_loss_weight"]
    )
    if loss_config["type"] == "evidential":
        alpha_key = "alpha_raw" if is_raw else "alpha_do"
        uncertainty_key = "u_raw" if is_raw else "u_do"
        branch_loss, components = evidential_loss(
            alpha=outputs[alpha_key],
            targets=labels,
            loss_type=str(loss_config["evidential_loss_type"]),
            kl_weight=float(loss_config["kl_weight"]),
            kl_annealing_factor=_kl_annealing_factor(epoch, loss_config),
            sample_weights=sample_weights,
            return_components=True,
        )
        uncertainty_weight = float(loss_config["uncertainty_regularizer_weight"])
        uncertainty_penalty = outputs[uncertainty_key].mean()
        total = branch_loss + uncertainty_weight * uncertainty_penalty
        ranking_loss = _pairwise_ranking_loss(
            scores=outputs["p_raw" if is_raw else "p_do"][:, -1],
            labels=labels,
            temperature=float(loss_config["ranking_temperature"]),
            scope=str(loss_config["ranking_scope"]),
            group_ids=ranking_group_ids,
        )
        ranking_weight = float(loss_config["ranking_loss_weight"])
        total = total + ranking_weight * ranking_loss
        components.update(
            {
                "branch": branch,
                "branch_weight": branch_weight,
                "uncertainty_regularizer_weight": uncertainty_weight,
                "uncertainty_penalty": float(
                    uncertainty_penalty.detach().cpu().item()
                ),
                "ranking_loss_weight": ranking_weight,
                "ranking_scope": str(loss_config["ranking_scope"]),
                "ranking_loss": float(ranking_loss.detach().cpu().item()),
                "weighted_loss": float((branch_weight * total).detach().cpu().item()),
            }
        )
        return branch_weight * total, components

    probability_key = "p_raw" if is_raw else "p_do"
    branch_loss = _branch_loss(
        outputs[probability_key],
        labels,
        sample_weights=sample_weights,
    )
    ranking_loss = _pairwise_ranking_loss(
        scores=outputs[probability_key][:, -1],
        labels=labels,
        temperature=float(loss_config["ranking_temperature"]),
        scope=str(loss_config["ranking_scope"]),
        group_ids=ranking_group_ids,
    )
    ranking_weight = float(loss_config["ranking_loss_weight"])
    total = branch_loss + ranking_weight * ranking_loss
    return branch_weight * total, {
        "branch": branch,
        "loss_type": "nll",
        "branch_weight": branch_weight,
        "data_loss": float(branch_loss.detach().cpu().item()),
        "ranking_loss_weight": ranking_weight,
        "ranking_scope": str(loss_config["ranking_scope"]),
        "ranking_loss": float(ranking_loss.detach().cpu().item()),
        "weighted_loss": float((branch_weight * total).detach().cpu().item()),
    }


def _pairwise_ranking_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    scope: str = "global",
    group_ids: Optional[Sequence[object]] = None,
) -> torch.Tensor:
    """Return a differentiable positive-over-negative ranking objective."""
    if scope == "within_target":
        if group_ids is None or len(group_ids) != scores.numel():
            raise ValueError(
                "within_target ranking requires one target_id per score."
            )
        losses = []
        normalized_groups = [str(value) for value in group_ids]
        for group_id in sorted(set(normalized_groups)):
            indices = [
                index
                for index, value in enumerate(normalized_groups)
                if value == group_id
            ]
            index_tensor = torch.as_tensor(
                indices, dtype=torch.long, device=scores.device
            )
            group_scores = scores.index_select(0, index_tensor)
            group_labels = labels.reshape(-1).index_select(0, index_tensor)
            if not bool((group_labels == 1).any()) or not bool(
                (group_labels == 0).any()
            ):
                continue
            losses.append(
                _global_pairwise_ranking_loss(
                    scores=group_scores,
                    labels=group_labels,
                    temperature=temperature,
                )
            )
        if not losses:
            return scores.sum() * 0.0
        return torch.stack(losses).mean()
    if scope != "global":
        raise ValueError("Unknown ranking scope: %s" % scope)
    return _global_pairwise_ranking_loss(scores, labels, temperature)


def _target_ranking_auxiliary_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    target_ids: Sequence[object],
    temperature: float,
    causal_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Rank positive over negative drugs within each sampled target."""
    raw_loss = _pairwise_ranking_loss(
        scores=outputs["p_raw"][:, -1],
        labels=labels,
        temperature=temperature,
        scope="within_target",
        group_ids=target_ids,
    )
    total = raw_loss
    branch_losses = {"raw": float(raw_loss.detach().cpu().item())}
    if "p_do" in outputs:
        do_loss = _pairwise_ranking_loss(
            scores=outputs["p_do"][:, -1],
            labels=labels,
            temperature=temperature,
            scope="within_target",
            group_ids=target_ids,
        )
        total = total + float(causal_weight) * do_loss
        branch_losses["causal"] = float(do_loss.detach().cpu().item())
    return total, branch_losses


def _global_pairwise_ranking_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Compute pairwise ranking for one global or target-conditioned group."""
    positive = scores[labels.reshape(-1) == 1]
    negative = scores[labels.reshape(-1) == 0]
    if positive.numel() == 0 or negative.numel() == 0:
        return scores.sum() * 0.0
    resolved_temperature = max(float(temperature), 1e-6)
    pairwise_margin = (
        positive.reshape(-1, 1) - negative.reshape(1, -1)
    ) / resolved_temperature
    return F.softplus(-pairwise_margin).mean() * resolved_temperature


def _kl_annealing_factor(epoch: int, loss_config: Dict[str, Any]) -> float:
    """Return a simple epoch-based KL annealing factor in [0, 1]."""
    annealing_epochs = int(loss_config.get("kl_annealing_epochs", 1))
    if annealing_epochs <= 0:
        return 1.0
    return min(1.0, float(epoch + 1) / float(annealing_epochs))


def _branch_loss(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute a tiny smoke classification loss from branch probabilities."""
    log_probabilities = torch.log(probabilities.clamp_min(1e-8))
    if sample_weights is None:
        return F.nll_loss(log_probabilities, labels)
    per_sample_loss = F.nll_loss(log_probabilities, labels, reduction="none")
    weights = sample_weights.to(device=per_sample_loss.device, dtype=per_sample_loss.dtype)
    denominator = weights.sum().clamp_min(1e-8)
    return (per_sample_loss * weights).sum() / denominator


def _resolve_device(requested_device: str) -> torch.device:
    """Resolve a requested device with CPU fallback for unavailable CUDA."""
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    return torch.device(requested_device)


def _resolve_repo_path(path: str) -> Path:
    """Resolve config paths relative to the project root."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def _use_target_randomization(config: Dict[str, Any]) -> bool:
    """Return whether smoke training should perturb protein features."""
    if "use_randomization" in config:
        return _as_bool(config["use_randomization"])
    randomization_config = config.get("randomization", {})
    return _as_bool(randomization_config.get("enabled", False))


def _randomize_protein_features(
    protein_features: torch.Tensor,
    randomization_config: Dict[str, Any],
    step: int,
) -> torch.Tensor:
    """Apply deterministic smoke target-feature perturbation."""
    mode = str(randomization_config.get("mode", "delete")).lower()
    if mode == "delete":
        rate = float(randomization_config.get("deletion_rate", 0.15))
    elif mode == "mutate":
        rate = float(randomization_config.get("mutation_rate", 0.15))
    else:
        rate = float(
            randomization_config.get(
                "rate", randomization_config.get("deletion_rate", 0.15)
            )
        )
    rate = min(1.0, max(0.0, rate))
    seed = int(randomization_config.get("seed", 7)) + int(step)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    if mode == "delete":
        keep_mask = torch.rand(
            protein_features.size(0),
            protein_features.size(1),
            1,
            generator=generator,
        ) >= rate
        keep_mask = keep_mask.to(protein_features.device)
        all_deleted = ~keep_mask.any(dim=1, keepdim=True)
        if bool(all_deleted.any().item()):
            first_position = torch.zeros_like(keep_mask)
            first_position[:, :1, :] = True
            keep_mask = torch.where(
                all_deleted.expand_as(keep_mask),
                first_position,
                keep_mask,
            )
        return protein_features * keep_mask.to(protein_features.dtype)

    if mode == "mutate":
        mutate_mask = torch.rand(
            protein_features.size(0),
            protein_features.size(1),
            1,
            generator=generator,
        ) < rate
        replacement = torch.randn(
            protein_features.shape,
            generator=generator,
            dtype=protein_features.dtype,
        ).to(protein_features.device)
        return torch.where(
            mutate_mask.to(protein_features.device),
            replacement,
            protein_features,
        )

    if mode == "shuffle":
        permutation = torch.randperm(
            protein_features.size(1),
            generator=generator,
            device=torch.device("cpu"),
        ).to(protein_features.device)
        return protein_features.index_select(dim=1, index=permutation)

    raise ValueError("Unsupported target randomization mode: %s" % mode)


def _as_bool(value: Any) -> bool:
    """Parse simple bool-like config values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


if __name__ == "__main__":
    main()

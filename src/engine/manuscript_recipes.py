"""Fixed manuscript control recipes retained from their original generators."""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
from typing import Dict,Any
ROOT=Path(__file__).resolve().parents[2]
def relative(path):
    try:return str(Path(path).relative_to(ROOT))
    except ValueError:return str(Path(path))

def apply_ablation(base, variant):
    """Same intervention/removal switches as BindingDB; never reset other weights."""
    c = copy.deepcopy(base)
    if variant in ('no_cam', 'no_backdoor'):
        component = 'cam' if variant == 'no_cam' else 'backdoor'
        c['use_'+component] = False
        for section in ('module_switches', 'ablation'):
            c.setdefault(section, {})[component] = False
        c.setdefault('debiasing', {})['use_'+component] = False
    elif variant == 'no_causal_loss':
        c.setdefault('loss', {})['causal_loss_weight'] = 0.
        c.setdefault('trainer', {})['causal_loss_weight'] = 0.
    elif variant == 'no_target_randomization':
        c['use_randomization'] = False
        c.setdefault('randomization', {})['enabled'] = False
        for section in ('module_switches', 'ablation'):
            c.setdefault(section, {})['target_randomization'] = False
    else:
        raise ValueError(variant)
    return c

def configure_common(
    config: Dict[str, Any],
    args: argparse.Namespace,
    seed: int,
    experiment_name: str,
    init_checkpoint: Path,
    checkpoint: Path,
    prediction: Path,
    residual_scale: float,
    use_norm: bool,
) -> None:
    config["seed"] = seed
    config["experiment_name"] = experiment_name
    config["checkpoint_path"] = relative(checkpoint)
    config["output_prediction_path"] = relative(prediction)
    config["epochs"] = args.epochs
    if getattr(args, "batch_size", None) is not None:
        config["batch_size"] = args.batch_size
    config["submission_control"] = {
        "generator": "run_residual_layernorm_controls.py",
        "condition": experiment_name,
    }
    if isinstance(config.get("randomization"), dict):
        config["randomization"]["seed"] = seed
    debiasing = config.setdefault("debiasing", {})
    debiasing["sample_residual_backdoor_scale"] = float(residual_scale)
    debiasing["use_sample_residual_backdoor"] = residual_scale > 0.0
    debiasing["backdoor_confounder_layernorm"] = bool(use_norm)
    trainer = config.setdefault("trainer", {})
    trainer.update(
        {
            "init_checkpoint": relative(init_checkpoint),
            "trainable_scope": "causal_only",
            "learning_rate": float(args.learning_rate),
            "device": "cuda",
        }
    )
    loss = config.setdefault("loss", {})
    loss["raw_loss_weight"] = 0.0
    loss["causal_loss_weight"] = 1.0
    config["performance_monitor"] = {
        "enabled": True,
        "metric": "valid_do_auc",
        "mode": "max",
        "save_best": True,
        "interval_epochs": 1,
    }


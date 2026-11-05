"""Reject missing scientific inputs instead of using demonstration fallbacks."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def validate_real_inputs(config, prediction=False):
    paths = [config.get("valid_csv")]
    if not prediction:
        paths.append(config.get("train_csv"))
    feature_keys = ["protein_feature_path", "drug2d_feature_path"]
    use_3d = config.get("model", {}).get("use_drug3d", config.get("features", {}).get("use_drug3d", config.get("use_drug3d", True)))
    if isinstance(use_3d, str):
        use_3d = use_3d.strip().lower() in {"true", "1", "yes", "on"}
    if use_3d:
        feature_keys.append("drug3d_feature_path")
    for key in feature_keys:
        paths.append(config.get(key) or config.get("features", {}).get(key))
    if config.get("use_causal_branch", True):
        for key in ("cluster_centers_path", "p_ci_path"):
            paths.append(config.get("debiasing", {}).get(key) or config.get("confounder", {}).get(key))
    missing = []
    for value in paths:
        path = Path(value) if value else None
        if path is not None and not path.is_absolute():
            path = ROOT / path
        if path is None or not path.is_file():
            missing.append(str(value))
    if missing:
        raise FileNotFoundError("Required PRISM-DTI inputs missing: " + ", ".join(missing))

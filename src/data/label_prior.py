"""Training-label priors, distinct from context-cluster masses."""
from pathlib import Path
import pandas as pd


def training_prior_groups(path, threshold=0.7):
    frame = pd.read_csv(Path(path), dtype={"target_id": str}, usecols=["target_id", "label"])
    if frame.target_id.isna().any() or not frame.label.isin([0, 1]).all():
        raise ValueError("Training priors require nonmissing target IDs and binary labels.")
    rates = frame.groupby("target_id").label.mean()
    return {key: "high_prior" if value >= threshold else "other_prior" for key, value in rates.items()}

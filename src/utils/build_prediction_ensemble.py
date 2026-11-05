"""Average aligned raw/do prediction CSVs without changing audit definitions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an aligned ensemble with explicit fixed weights or equal weights.")
    parser.add_argument("--pred-file", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weight", type=float, action="append", dest="weights", help="Repeat in member order; must sum to one. Omit for equal weights.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = [pd.read_csv(path, dtype={"drug_id": str, "target_id": str}) for path in args.pred_file]
    reference = average_predictions(frames, args.weights)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    reference.to_csv(output, index=False)
    labels = reference["label"].astype(int)
    print({"members": len(frames), "rows": len(reference),
           "raw_auc": roc_auc_score(labels, reference["p_raw"]),
           "raw_aupr": average_precision_score(labels, reference["p_raw"]),
           "do_auc": roc_auc_score(labels, reference["p_do"]),
           "do_aupr": average_precision_score(labels, reference["p_do"]),
           "output": str(output)})


def average_predictions(frames, weights=None):
    if len(frames) < 2:
        raise ValueError("At least two --pred-file values are required.")
    weights = np.full(len(frames), 1 / len(frames)) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != (len(frames),) or not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1., atol=1e-12, rtol=0):
        raise ValueError("One finite nonnegative weight per member is required, with total one.")

    reference = frames[0].copy()
    required = {"drug_id", "target_id", "label", "p_raw", "p_do"}
    for index, frame in enumerate(frames):
        path = "member %d" % index
        missing = required - set(frame.columns)
        if missing:
            raise ValueError("%s missing columns: %s" % (path, sorted(missing)))
        if len(frame) != len(reference):
            raise ValueError("Prediction row counts do not match.")
        for column in ("drug_id", "target_id", "label"):
            if not frame[column].astype(str).equals(
                reference[column].astype(str)
            ):
                raise ValueError(
                    "Prediction alignment mismatch in %s for %s"
                    % (path, column)
                )

        for column in ("p_raw", "p_do"):
            values = frame[column].to_numpy(dtype=float)
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError("Probabilities must be finite and in [0, 1].")

    # A member's alpha/evidence columns cannot describe a probability ensemble.
    reference = reference[["drug_id", "target_id", "label"]].copy()

    for column in ("p_raw", "p_do", "u_raw", "u_do"):
        available = [frame[column] for frame in frames if column in frame]
        if len(available) == len(frames):
            reference[column] = sum(w * values.to_numpy() for w, values in zip(weights, available))
    reference["pcs"] = reference["p_raw"] - reference["p_do"]
    if "u_raw" in reference and "u_do" in reference:
        reference["delta_u"] = reference["u_do"] - reference["u_raw"]
    for column in ("target_prior_group", "scaffold_id"):
        if all(column in frame for frame in frames):
            values = frames[0][column].fillna("").astype(str)
            if any(not frame[column].fillna("").astype(str).equals(values) for frame in frames[1:]):
                raise ValueError("Member metadata alignment mismatch: " + column)
            reference[column] = frames[0][column]
    return reference


if __name__ == "__main__":
    main()

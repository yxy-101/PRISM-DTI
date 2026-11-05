#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, chi2_contingency
from src.data.label_prior import training_prior_groups

try:
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        precision_recall_fscore_support,
        accuracy_score,
        brier_score_loss,
    )
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


def binary_ece(y_true, p_pos, n_bins=10):
    y_true = np.asarray(y_true).astype(int)
    p_pos = np.asarray(p_pos).astype(float)
    pred = (p_pos >= 0.5).astype(int)
    conf = np.where(pred == 1, p_pos, 1 - p_pos)
    correct = (pred == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            continue
        acc_bin = correct[mask].mean()
        conf_bin = conf[mask].mean()
        ece += mask.mean() * abs(acc_bin - conf_bin)
    return float(ece)


def aurc(y_true, p_pos):
    y_true = np.asarray(y_true).astype(int)
    p_pos = np.asarray(p_pos).astype(float)
    pred = (p_pos >= 0.5).astype(int)
    conf = np.where(pred == 1, p_pos, 1 - p_pos)
    error = (pred != y_true).astype(float)

    order = np.argsort(-conf)
    sorted_error = error[order]
    cumulative_error = np.cumsum(sorted_error)
    coverage = np.arange(1, len(sorted_error) + 1) / len(sorted_error)
    risk = cumulative_error / np.arange(1, len(sorted_error) + 1)

    return float(np.trapz(risk, coverage))


def make_quantile_labels(bin_count, expected_q=5):
    if bin_count <= 1:
        return ["Q1_all"]
    if bin_count == expected_q:
        return ["Q1_lowest", "Q2", "Q3", "Q4", "Q5_highest"]

    labels = []
    for idx in range(bin_count):
        if idx == 0:
            labels.append(f"Q1_lowest_of_{bin_count}")
        elif idx == bin_count - 1:
            labels.append(f"Q{idx + 1}_highest_of_{bin_count}")
        else:
            labels.append(f"Q{idx + 1}_of_{bin_count}")
    return labels


def assign_quantile_groups(series, q=5):
    numeric = pd.to_numeric(series, errors="coerce")
    result = pd.Series("missing_pcs", index=series.index, dtype="object")
    valid = numeric.notna()

    if not valid.any() or numeric[valid].nunique() < 2:
        result.loc[valid] = "Q1_all"
        categories = ["Q1_all"]
        if (~valid).any():
            categories.append("missing_pcs")
        return pd.Categorical(result, categories=categories, ordered=True)

    try:
        codes = pd.qcut(numeric[valid], q=q, labels=False, duplicates="drop")
    except ValueError:
        result.loc[valid] = "Q1_all"
        categories = ["Q1_all"]
        if (~valid).any():
            categories.append("missing_pcs")
        return pd.Categorical(result, categories=categories, ordered=True)

    if not codes.notna().any():
        result.loc[valid] = "Q1_all"
        categories = ["Q1_all"]
        if (~valid).any():
            categories.append("missing_pcs")
        return pd.Categorical(result, categories=categories, ordered=True)

    labels = make_quantile_labels(int(codes.max()) + 1, expected_q=q)
    result.loc[valid] = [labels[int(code)] if pd.notna(code) else "missing_pcs" for code in codes]
    categories = labels + (["missing_pcs"] if (~valid).any() else [])
    return pd.Categorical(result, categories=categories, ordered=True)


def branch_metrics(df, prob_col, prefix, ece_bins):
    y = df["label"].astype(int).values
    p = df[prob_col].astype(float).values
    pred = (p >= 0.5).astype(int)

    out = {}
    out[f"{prefix}_accuracy"] = float((pred == y).mean())
    out[f"{prefix}_brier"] = float(np.mean((p - y) ** 2))
    out[f"{prefix}_ece"] = binary_ece(y, p, n_bins=ece_bins)
    out[f"{prefix}_aurc"] = aurc(y, p)

    if SKLEARN_OK:
        out[f"{prefix}_roc_auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan
        out[f"{prefix}_aupr"] = float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else np.nan
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, pred, average="binary", zero_division=0
        )
        out[f"{prefix}_precision"] = float(precision)
        out[f"{prefix}_recall"] = float(recall)
        out[f"{prefix}_f1"] = float(f1)
    else:
        out[f"{prefix}_roc_auc"] = np.nan
        out[f"{prefix}_aupr"] = np.nan
        out[f"{prefix}_precision"] = np.nan
        out[f"{prefix}_recall"] = np.nan
        out[f"{prefix}_f1"] = np.nan

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", required=True)
    parser.add_argument("--dataset", default="bindingdb")
    parser.add_argument("--split", default="valid")
    parser.add_argument("--model_name", default="full")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--ece_bins", type=int, default=10)
    parser.add_argument("--out_dir", default="outputs/metrics")
    parser.add_argument("--train-csv", default=None, help="Training CSV for Eq.21 prior groups.")
    args = parser.parse_args()

    pred_path = Path(args.pred_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pred_path, dtype={"drug_id": str, "target_id": str})
    if args.train_csv:
        groups = training_prior_groups(args.train_csv)
        df["target_prior_group"] = df.target_id.map(groups)
    if "target_prior_group" not in df:
        raise ValueError("Provide --train-csv when prediction metadata lacks target_prior_group.")
    df["target_prior_group"] = df["target_prior_group"].fillna("missing_prior_group")
    has_final_head = "p_final" in df.columns and df["p_final"].notna().any()

    required = ["label", "p_raw", "p_do", "pcs", "target_prior_group"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {pred_path}: {missing}")

    df["raw_pred"] = (df["p_raw"] >= 0.5).astype(int)
    df["raw_conf"] = df["p_raw"].where(df["raw_pred"] == 1, 1 - df["p_raw"])
    df["raw_error"] = (df["raw_pred"] != df["label"]).astype(int)

    df["do_pred"] = (df["p_do"] >= 0.5).astype(int)
    df["do_conf"] = df["p_do"].where(df["do_pred"] == 1, 1 - df["p_do"])
    df["do_error"] = (df["do_pred"] != df["label"]).astype(int)

    if has_final_head:
        df["final_pred"] = (df["p_final"] >= 0.5).astype(int)
        df["final_conf"] = df["p_final"].where(
            df["final_pred"] == 1,
            1 - df["p_final"],
        )
        df["final_error"] = (df["final_pred"] != df["label"]).astype(int)

    hce = df[(df["raw_conf"] >= args.threshold) & (df["raw_error"] == 1)].copy()

    ppce = df[
        (df["pcs"] > 0)
        & (df["raw_pred"] == 1)
        & (df["raw_conf"] >= args.threshold)
        & (df["label"] == 0)
    ].copy()

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "model": args.model_name,
        "pred_file": str(pred_path),
        "sample_count": len(df),
        "positive_count": int((df["label"] == 1).sum()),
        "negative_count": int((df["label"] == 0).sum()),
        "positive_rate": float(df["label"].mean()),
        "threshold": args.threshold,
        "all_high_conf_error_count": len(hce),
        "high_conf_false_positive_count": int(((df.label == 0) & (df.p_raw >= args.threshold)).sum()),
        "training_prior_known_count": int(df.target_prior_group.isin(["high_prior", "other_prior"]).sum()),
        "positive_pseudo_error_count": len(ppce),
        "positive_pseudo_error_ratio_in_hce": len(ppce) / len(hce) if len(hce) else 0.0,
        "high_prior_positive_pseudo_error_count": int((ppce["target_prior_group"] == "high_prior").sum()) if len(ppce) else 0,
        "high_prior_ratio_in_positive_pseudo_errors": float((ppce["target_prior_group"] == "high_prior").mean()) if len(ppce) else 0.0,
        "mean_pcs": float(df["pcs"].mean()),
        "mean_positive_pcs": float(df.loc[df["pcs"] > 0, "pcs"].mean()) if (df["pcs"] > 0).any() else 0.0,
    }

    summary.update(branch_metrics(df, "p_raw", "raw", args.ece_bins))
    summary.update(branch_metrics(df, "p_do", "do", args.ece_bins))
    if has_final_head:
        final_hce = df[
            (df["final_conf"] >= args.threshold) & (df["final_error"] == 1)
        ].copy()
        final_positive_hcfp = df[
            (df["final_pred"] == 1)
            & (df["final_conf"] >= args.threshold)
            & (df["label"] == 0)
        ].copy()
        final_pcs_risk_hcfp = final_positive_hcfp[
            final_positive_hcfp["pcs"] > 0
        ].copy()
        summary.update(branch_metrics(df, "p_final", "final", args.ece_bins))
        summary.update(
            {
                "final_high_conf_error_count": len(final_hce),
                "final_high_conf_false_positive_count": len(final_positive_hcfp),
                "final_pcs_risk_false_positive_count": len(final_pcs_risk_hcfp),
                "final_pcs_risk_false_positive_ratio_in_raw_ppce": (
                    len(final_pcs_risk_hcfp) / len(ppce) if len(ppce) else 0.0
                ),
                "mean_p_final": float(df["p_final"].mean()),
                "mean_u_final": float(df["u_final"].mean())
                if "u_final" in df.columns
                else np.nan,
                "mean_final_gate": float(df["final_gate"].mean())
                if "final_gate" in df.columns
                else np.nan,
            }
        )
    else:
        final_hce = pd.DataFrame()
        final_positive_hcfp = pd.DataFrame()
        final_pcs_risk_hcfp = pd.DataFrame()

    # Fisher / chi-square only within high-confidence errors
    known_prior = hce["target_prior_group"].isin(["high_prior", "other_prior"])
    if len(hce) > 0 and bool(known_prior.any()):
        hce["is_ppce"] = (
            (hce["pcs"] > 0)
            & (hce["raw_pred"] == 1)
            & (hce["raw_conf"] >= args.threshold)
            & (hce["label"] == 0)
        ).astype(int)
        hce["is_high_prior"] = (hce["target_prior_group"] == "high_prior").astype(int)

        a = int((known_prior & (hce["is_high_prior"] == 1) & (hce["is_ppce"] == 1)).sum())
        b = int((known_prior & (hce["is_high_prior"] == 1) & (hce["is_ppce"] == 0)).sum())
        c = int((known_prior & (hce["is_high_prior"] == 0) & (hce["is_ppce"] == 1)).sum())
        d = int((known_prior & (hce["is_high_prior"] == 0) & (hce["is_ppce"] == 0)).sum())

        table = [[a, b], [c, d]]
        oddsratio, fisher_p = fisher_exact(table)

        try:
            chi2, chi_p, _, _ = chi2_contingency(table)
        except Exception:
            chi2, chi_p = np.nan, np.nan

        summary["fisher_table"] = str(table)
        summary["fisher_oddsratio"] = float(oddsratio)
        summary["fisher_p"] = float(fisher_p)
        summary["chi2"] = float(chi2)
        summary["chi2_p"] = float(chi_p)
    else:
        summary["fisher_table"] = None
        summary["fisher_oddsratio"] = np.nan
        summary["fisher_p"] = np.nan
        summary["chi2"] = np.nan
        summary["chi2_p"] = np.nan

    base = f"{args.dataset}_{args.model_name}_{args.split}"

    pd.DataFrame([summary]).to_csv(out_dir / f"{base}_summary.csv", index=False)

    prior_agg = {
        "sample_count": ("label", "size"),
        "label_0": ("label", lambda x: int((x == 0).sum())),
        "label_1": ("label", lambda x: int((x == 1).sum())),
        "positive_rate": ("label", "mean"),
        "mean_p_raw": ("p_raw", "mean"),
        "mean_p_do": ("p_do", "mean"),
        "mean_pcs": ("pcs", "mean"),
        "mean_positive_pcs": (
            "pcs",
            lambda x: x[x > 0].mean() if (x > 0).any() else 0,
        ),
        "raw_error_rate": ("raw_error", "mean"),
        "do_error_rate": ("do_error", "mean"),
        "mean_raw_conf": ("raw_conf", "mean"),
    }
    if has_final_head:
        prior_agg.update(
            {
                "mean_p_final": ("p_final", "mean"),
                "final_error_rate": ("final_error", "mean"),
                "mean_final_conf": ("final_conf", "mean"),
            }
        )
        if "final_gate" in df.columns:
            prior_agg["mean_final_gate"] = ("final_gate", "mean")
    prior = df.groupby("target_prior_group").agg(**prior_agg).reset_index()
    prior.to_csv(out_dir / f"{base}_prior_group.csv", index=False)

    df["pcs_group"] = assign_quantile_groups(df["pcs"], q=5)
    pcs_rows = []
    for name, g in df.groupby("pcs_group", observed=False):
        high_conf = g[g["raw_conf"] >= args.threshold]
        row = {
            "pcs_group": str(name),
            "sample_count": len(g),
            "mean_pcs": g["pcs"].mean(),
            "mean_p_raw": g["p_raw"].mean(),
            "mean_p_do": g["p_do"].mean(),
            "raw_error_rate": g["raw_error"].mean(),
            "mean_raw_conf": g["raw_conf"].mean(),
            "high_conf_count": len(high_conf),
            "high_conf_error_count": int(high_conf["raw_error"].sum()),
            "high_conf_error_rate": high_conf["raw_error"].mean() if len(high_conf) else 0,
        }
        if has_final_head:
            row.update(
                {
                    "mean_p_final": g["p_final"].mean(),
                    "final_error_rate": g["final_error"].mean(),
                    "mean_final_conf": g["final_conf"].mean(),
                }
            )
            if "final_gate" in g.columns:
                row["mean_final_gate"] = g["final_gate"].mean()
        pcs_rows.append(row)
    pd.DataFrame(pcs_rows).to_csv(out_dir / f"{base}_pcs_group.csv", index=False)

    ppce.to_csv(out_dir / f"{base}_ppce_errors.csv", index=False)
    hce.to_csv(out_dir / f"{base}_hce_errors.csv", index=False)
    df[(df.label == 0) & (df.p_raw >= args.threshold)].to_csv(out_dir / f"{base}_hcfp_errors.csv", index=False)
    if has_final_head:
        final_hce.to_csv(out_dir / f"{base}_final_hce_errors.csv", index=False)
        final_pcs_risk_hcfp.to_csv(
            out_dir / f"{base}_final_pcs_risk_hcfp.csv",
            index=False,
        )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    print(pd.DataFrame([summary]))
    print("\nSaved:")
    print(out_dir / f"{base}_summary.csv")
    print(out_dir / f"{base}_prior_group.csv")
    print(out_dir / f"{base}_pcs_group.csv")
    print(out_dir / f"{base}_ppce_errors.csv")
    print(out_dir / f"{base}_hce_errors.csv")
    if has_final_head:
        print(out_dir / f"{base}_final_hce_errors.csv")
        print(out_dir / f"{base}_final_pcs_risk_hcfp.csv")


if __name__ == "__main__":
    main()

"""Manuscript analysis functions retained from the research implementation."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import json
KEY=['drug_id','target_id','label']
CONTROLS=['log_train_n','mean_drug_prior','mean_log_drug_n','log_negative_n']

def main():
    parser = argparse.ArgumentParser(description="Eq.23 is not used here: equal-target linear HCFP association for DeepDTA/DrugBAN.")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--pred-file", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-seed", type=int, choices=(7, 42, 2024), required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261016)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train, reference, pred = read(args.train_csv), read(args.split_csv), read(args.pred_file, True)
    if not reference[KEY].equals(pred[KEY]):
        raise ValueError("The model predictions do not match the declared evaluation pairs/labels.")
    if args.bootstrap <= 0:
        raise ValueError("Positive bootstrap count required.")
    if args.output.exists():
        raise FileExistsError("Preserve existing analysis: " + str(args.output))
    args.output.mkdir(parents=True)
    rows = []
    for threshold in (.7, .8, .9):
        table, excluded = target_table(train, pred, threshold)
        for minimum in (0, 20):
            subset = table[table.train_n >= minimum].copy()
            rows.append(dict(dataset=args.dataset, model=args.model, model_seed=args.model_seed,
                             threshold=threshold, minimum_train_pairs=minimum, excluded_unseen_pairs=excluded,
                             **association(subset, args.bootstrap, args.bootstrap_seed)))
    pd.DataFrame(rows).to_csv(args.output / "associations.csv", index=False)
    protocol = dict(bootstrap=args.bootstrap, bootstrap_seed=args.bootstrap_seed,
                    estimator="equal-target OLS risk difference per prior SD; not binomial log-odds",
                    source_hashes={str(p): digest(p) for p in (args.train_csv, args.split_csv, args.pred_file)},
                    historical_training_verified=False, independent_confirmation=False)
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path, prediction=False):
    f = pd.read_csv(path, dtype={'drug_id': str, 'target_id': str})
    required = KEY + (['p_raw'] if prediction else [])
    if not set(required).issubset(f) or f[KEY].isna().any().any():
        raise ValueError('Missing columns/values: ' + str(path))
    if f.duplicated(KEY[:2]).any() or not f.label.isin([0, 1]).all():
        raise ValueError('Duplicate pairs/nonbinary labels: ' + str(path))
    if prediction and (not np.isfinite(f.p_raw).all() or not f.p_raw.between(0, 1).all()):
        raise ValueError('Invalid probability: ' + str(path))
    return f.sort_values(KEY[:2]).reset_index(drop=True)

def target_table(train, pred, threshold):
    ts = train.groupby('target_id').label.agg(['mean', 'size'])
    ds = train.groupby('drug_id').label.agg(['mean', 'size'])
    f = pred[pred.label == 0].copy()
    f['prior'] = f.target_id.map(ts['mean'])
    excluded = int(f.prior.isna().sum())
    f = f[f.prior.notna()].copy()
    f['train_n'] = f.target_id.map(ts['size'])
    # Unseen drugs use training-global values, with coverage recorded below.
    f['unseen_drug'] = ~f.drug_id.isin(ds.index)
    f['drug_prior'] = f.drug_id.map(ds['mean']).fillna(train.label.mean())
    f['log_drug_n'] = np.log1p(f.drug_id.map(ds['size']).fillna(0))
    f['event'] = (f.p_raw >= threshold).astype(int)
    t = f.groupby('target_id').agg(prior=('prior', 'first'), train_n=('train_n', 'first'),
        negative_n=('label', 'size'), events=('event', 'sum'),
        mean_drug_prior=('drug_prior', 'mean'), mean_log_drug_n=('log_drug_n', 'mean'),
        unseen_drug_n=('unseen_drug', 'sum'))
    t['risk'] = t.events / t.negative_n
    t['log_train_n'] = np.log1p(t.train_n)
    t['log_negative_n'] = np.log1p(t.negative_n)
    return t, excluded

def association(t, draws, seed):
    """Equal target weight linear risk association, not a probability model."""
    if len(t) < 20 or t.prior.std(ddof=0) < 1e-12:
        return dict(status='insufficient_targets_or_exposure_variation', targets=len(t))
    exposure = (t.prior.to_numpy() - t.prior.mean()) / t.prior.std(ddof=0)
    controls = []
    for name in CONTROLS:
        x = t[name].to_numpy(float)
        if x.std() > 1e-12:
            controls.append((x - x.mean()) / x.std())
    x = np.column_stack([np.ones(len(t)), exposure] + controls)
    y = t.risk.to_numpy()
    if np.linalg.matrix_rank(x) != x.shape[1]:
        return dict(status='rank_deficient', targets=len(t), design_columns=x.shape[1])
    coef = float(np.linalg.lstsq(x, y, rcond=None)[0][1])
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(draws):
        idx = rng.integers(0, len(t), len(t))
        if np.linalg.matrix_rank(x[idx]) == x.shape[1]:
            boots.append(float(np.linalg.lstsq(x[idx], y[idx], rcond=None)[0][1]))
    high = t.prior >= .7
    lo, hi = np.quantile(boots, [.025, .975]) if boots else (np.nan, np.nan)
    return dict(status='estimated', targets=len(t), design_columns=x.shape[1],
        adjusted_risk_difference_per_prior_sd=coef, ci_low=float(lo), ci_high=float(hi),
        bootstrap_valid=len(boots), high_prior_targets=int(high.sum()),
        other_targets=int((~high).sum()),
        target_equal_high_minus_other=float(t.loc[high, 'risk'].mean()-t.loc[~high, 'risk'].mean())
            if high.any() and (~high).any() else None,
        negative_n=int(t.negative_n.sum()), hcfp=int(t.events.sum()),
        unseen_drug_negative_n=int(t.unseen_drug_n.sum()))

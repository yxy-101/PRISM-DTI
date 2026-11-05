"""Validation-only, equal-opportunity calibration of fixed residual predictions."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

DATASETS = ('davis', 'kiba', 'bindingdb')
SEEDS = (7, 42, 2024)
VARIANTS = ('raw', 'raw_T', 'residual', 'residual_T')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def logits(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1-1e-7)
    return np.log(p)-np.log1p(-p)


def apply_temperature(p, t):
    if not np.isfinite(t) or t <= 0:
        raise ValueError('Temperature must be positive and finite')
    if t == 1:
        return np.asarray(p, dtype=float).copy()
    z = logits(p)/t
    return np.exp(-np.logaddexp(0, -z))


def fit_temperature(y, p):
    """Deterministic bounded golden-section search on log(T), with identity fallback."""
    z = logits(p)
    y = np.asarray(y)
    def loss(x):
        a = z/np.exp(x)
        return float(np.mean(np.logaddexp(0, a)-y*a))
    left, right = -3., 3.
    ratio = (np.sqrt(5)-1)/2
    c, d = right-ratio*(right-left), left+ratio*(right-left)
    for _ in range(100):
        if loss(c) < loss(d):
            right, d = d, c
            c = right-ratio*(right-left)
        else:
            left, c = c, d
            d = left+ratio*(right-left)
    x = (left+right)/2
    return float(np.exp(x)) if loss(x) < loss(0) else 1.


def read(path):
    f = pd.read_csv(path, dtype={'drug_id': str, 'target_id': str})
    required = ['drug_id', 'target_id', 'label', 'p_raw', 'p_do']
    if not set(required).issubset(f) or f.empty:
        raise ValueError('Missing columns or empty prediction: %s' % path)
    if f[required].isna().any().any() or f.duplicated(['drug_id', 'target_id']).any():
        raise ValueError('Missing/duplicate identifiers or values: %s' % path)
    if not f.label.isin([0, 1]).all() or f.label.nunique() != 2:
        raise ValueError('Both binary classes required: %s' % path)
    for column in ('p_raw', 'p_do'):
        if not np.isfinite(f[column]).all() or not f[column].between(0, 1).all():
            raise ValueError('Invalid probability: %s' % path)
    return f.set_index(['drug_id', 'target_id']).sort_index()


def events(y, p, raw):
    error = (p >= .5) != y
    z = logits(p)
    pool = np.flatnonzero(raw >= .8)
    chosen = pool[np.argsort(-p[pool], kind='stable')[:int(np.floor(.9*len(pool)))]]
    fp = np.zeros(len(y))
    fp[chosen] = y[chosen] == 0
    return dict(hce=(error & (np.maximum(p, 1-p) >= .8)).astype(float),
                hcfp=((y == 0) & (p >= .8)).astype(float),
                brier=(p-y)**2, nll=np.logaddexp(0, z)-y*z, retained_fp=fp)


def scores(y, p, raw):
    # Tie-aware AUROC and average precision, matching binary sklearn definitions.
    table = pd.DataFrame({'p': p, 'y': y})
    ranks = table.p.rank(method='average').to_numpy()
    npos = int(y.sum()); nneg = len(y)-npos
    auc = (ranks[y == 1].sum()-npos*(npos+1)/2)/(npos*nneg)
    grouped = table.groupby('p').y.agg(['sum', 'count']).sort_index(ascending=False)
    ap = ((grouped['sum'].cumsum()/grouped['count'].cumsum())*grouped['sum']/npos).sum()
    e = events(y, p, raw)
    return dict(auc=float(auc), aupr=float(ap), errors=int(((p >= .5) != y).sum()),
                hce_count=int(e['hce'].sum()), hcfp_count=int(e['hcfp'].sum()),
                hce_rate=float(e['hce'].mean()), hcfp_rate=float(e['hcfp'].mean()),
                brier=float(e['brier'].mean()), nll=float(e['nll'].mean()),
                high_conf_n=int((np.maximum(p, 1-p) >= .8).sum()),
                retained_fp=int(e['retained_fp'].sum()),
                retained_n=int(np.floor(.9*(raw >= .8).sum())))


def prediction_paths(root, datasets):
    folder = Path(root)/'outputs/predictions/residual_layernorm_controls'
    return [folder/f'{ds}_resnorm_rho025_norm_seed{seed}_{split}_predictions.csv'
            for ds in datasets for seed in SEEDS for split in ('valid', 'test')]


def preflight(root, datasets):
    paths = prediction_paths(root, datasets)
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing existing predictions; no training performed:\n'+'\n'.join(missing))
    return paths


def run(root, draws=5000, datasets=DATASETS, output=None):
    if draws < 1000:
        raise ValueError('At least 1000 bootstrap draws required')
    root = Path(root)
    datasets = tuple(datasets)
    if not datasets or len(set(datasets)) != len(datasets) or not set(datasets).issubset(set(DATASETS) | {'biosnap'}):
        raise ValueError('Invalid/duplicate dataset selection')
    preflight(root, datasets)
    out = Path(output) if output is not None else root/'outputs/metrics/matched_calibration'
    if out.exists():
        raise FileExistsError('Preserve existing calibration output: '+str(out))
    out.mkdir(parents=True)
    (out/'status.json').write_text('{"complete": false}', encoding='utf-8')
    (out/'report.md').write_text('Current run incomplete.\n', encoding='utf-8')
    folder = root/'outputs/predictions/residual_layernorm_controls'
    def path(ds, seed, split):
        return folder/f'{ds}_resnorm_rho025_norm_seed{seed}_{split}_predictions.csv'
    policies, hashes, valid = {}, {}, {}
    # Fit and save all nine validation policies before reading any test labels.
    for ds in datasets:
        for seed in SEEDS:
            f = read(path(ds, seed, 'valid'))
            key = f'{ds}_seed{seed}'
            valid[key] = f
            policies[key] = {branch: fit_temperature(f.label.to_numpy(), f[col].to_numpy())
                             for branch, col in [('raw', 'p_raw'), ('residual', 'p_do')]}
            hashes[str(path(ds, seed, 'valid').relative_to(root))] = sha(path(ds, seed, 'valid'))
    (out/'policies.json').write_text(json.dumps(dict(policies=policies, validation_hashes=hashes), indent=2), encoding='utf-8')
    rows, comparisons = [], []
    rng = np.random.default_rng(20261001)
    pairs = [('raw_T', 'raw'), ('residual_T', 'residual'), ('residual_T', 'raw_T')]
    for ds in datasets:
        for split in ('valid', 'test'):
            reference = None
            deltas = []
            labels = [(a, b, metric) for a, b in pairs for metric in ('hce', 'hcfp', 'brier', 'nll', 'retained_fp')]
            for seed in SEEDS:
                key = f'{ds}_seed{seed}'
                f = valid[key] if split == 'valid' else read(path(ds, seed, split))
                if reference is not None and (not f.index.equals(reference.index) or not np.array_equal(f.label, reference.label)):
                    raise ValueError('Seed evaluation pairs/labels differ')
                reference = f
                hashes[str(path(ds, seed, split).relative_to(root))] = sha(path(ds, seed, split))
                y, raw, residual = f.label.to_numpy(), f.p_raw.to_numpy(), f.p_do.to_numpy()
                probs = dict(raw=raw, residual=residual, raw_T=apply_temperature(raw, policies[key]['raw']),
                             residual_T=apply_temperature(residual, policies[key]['residual']))
                for variant in VARIANTS:
                    rows.append(dict(dataset=ds, split=split, seed=seed, variant=variant,
                                     temperature=policies[key]['raw' if variant.startswith('raw') else 'residual'] if variant.endswith('_T') else 1.,
                                     **scores(y, probs[variant], raw)))
                ev = {v: events(y, p, raw) for v, p in probs.items()}
                deltas.append(np.stack([ev[a][m]-ev[b][m] for a, b, m in labels], axis=1))
            # Same target draw jointly across arms and fitted seeds; no probability ensemble.
            delta = np.mean(deltas, axis=0)
            codes, targets = pd.factorize(reference.index.get_level_values('target_id'), sort=True)
            counts = np.bincount(codes)
            sums = np.stack([np.bincount(codes, weights=delta[:, j]) for j in range(delta.shape[1])], axis=1)
            samples = np.empty((draws, delta.shape[1]))
            for i in range(draws):
                draw = rng.integers(0, len(targets), len(targets))
                samples[i] = sums[draw].sum(axis=0)/counts[draw].sum()
            for j, (a, b, metric) in enumerate(labels):
                lo, hi = np.quantile(samples[:, j], [.025, .975])
                # Family includes every selected dataset's primary test HCE contrast.
                family_lo, family_hi = np.quantile(samples[:, j], [.025/len(datasets), 1-.025/len(datasets)])
                primary = split == 'test' and a == 'residual_T' and b == 'raw_T' and metric == 'hce'
                comparisons.append(dict(dataset=ds, split=split, contrast=a+' minus '+b, metric=metric,
                    mean_delta=float(delta[:, j].mean()), ci_low=float(lo), ci_high=float(hi),
                    primary_family_ci_low=float(family_lo) if primary else None,
                    primary_family_ci_high=float(family_hi) if primary else None))
    frame = pd.DataFrame(rows)
    frame.to_csv(out/'runs.csv', index=False)
    seed_deltas = []
    indexed = frame.set_index(['dataset', 'split', 'seed', 'variant'])
    for ds in datasets:
        for split in ('valid', 'test'):
            for seed in SEEDS:
                for a, b in pairs:
                    first, second = indexed.loc[(ds, split, seed, a)], indexed.loc[(ds, split, seed, b)]
                    seed_deltas.append(dict(dataset=ds, split=split, seed=seed, contrast=a+' minus '+b,
                        **{m: float(first[m]-second[m]) for m in ('auc', 'aupr', 'hce_rate', 'hcfp_rate', 'brier', 'nll', 'retained_fp', 'high_conf_n', 'errors')}))
    pd.DataFrame(seed_deltas).to_csv(out/'seed_deltas.csv', index=False)
    numeric = [c for c in frame if c not in ('dataset', 'split', 'seed', 'variant')]
    summary = frame.groupby(['dataset', 'split', 'variant'])[numeric].agg(['mean', 'std'])
    summary.to_csv(out/'summary.csv')
    pd.DataFrame(comparisons).to_csv(out/'paired.csv', index=False)
    protocol = dict(complete=True, training_performed=False, draws=draws, datasets=list(datasets),
        prediction_count=len(hashes), primary_family_size=len(datasets),
        primary='test HCE prevalence: residual_T minus raw_T; negative favorable',
        temperature='Validation NLL only; log T in [-3,3]; identity fallback; 100 golden-section iterations',
        intervals='Target-cluster bootstrap conditional on three fitted seeds and temperatures; test primary HCE Bonferroni family=%d' % len(datasets),
        independent_confirmation=False, ppce='PCS-defined PPCE not recomputed from calibrated probabilities; not a primary endpoint',
        screening='Same uncalibrated raw>=.8 pool, floor 90% coverage, canonical pair order ties',
        input_hashes=hashes, code_sha256=sha(__file__))
    (out/'status.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    lines = ['# Residual equal-opportunity temperature audit', '',
        'Datasets: '+', '.join(datasets)+'; primary family size: '+str(len(datasets)),
        'Validation-only fits; retrospective fixed tests. No automatic mitigation/superiority claim.',
        'HCE/HCFP are prevalence among all pairs; lower HCE may reflect fewer confident predictions, not corrected classifications.',
        'Inspect high_conf_n, errors, Brier/NLL, AUROC/AUPR and fixed-budget FP together.', '',
        '| Dataset | Variant | AUROC | AUPR | HCE rate | HCFP rate | Brier | NLL |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for (ds, split, variant), row in summary.iterrows():
        if split == 'test':
            lines.append('| %s | %s | %s |' % (ds, variant, ' | '.join(
                '%.6f +/- %.6f' % (row[(m, 'mean')], row[(m, 'std')]) for m in ('auc', 'aupr', 'hce_rate', 'hcfp_rate', 'brier', 'nll'))))
    lines.extend(['', 'Primary paired comparisons (rate units; negative favorable):'])
    for r in comparisons:
        if r['primary_family_ci_low'] is not None:
            lines.append('%s: %.6f [%.6f, %.6f] family-adjusted interval' % (r['dataset'], r['mean_delta'], r['primary_family_ci_low'], r['primary_family_ci_high']))
    (out/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))
    return protocol

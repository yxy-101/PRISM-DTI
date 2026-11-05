"""Train-only target-label propensity weights with class-mass preservation."""
from collections import defaultdict
import math
import random


def compute_weights(records, prior_records, shuffled=False, seed=7):
    totals, positives = defaultdict(int), defaultdict(int)
    if not records or not prior_records:
        raise ValueError('Nonempty training records required')
    for r in prior_records:
        if int(r['label']) not in (0, 1):
            raise ValueError('Binary training labels required')
        t = str(r['target_id']); totals[t] += 1; positives[t] += int(r['label'])
    global_rate = sum(positives.values())/sum(totals.values())
    if not 0 < global_rate < 1:
        raise ValueError('Both training classes required')
    # Fixed pseudo-count 20, target rate shrunk towards training global rate.
    rates = {t: (positives[t]+20*global_rate)/(n+20) for t, n in totals.items()}
    raw, labels = [], []
    for r in records:
        t, y = str(r['target_id']), int(r['label'])
        if y not in (0, 1) or t not in rates:
            raise ValueError('Unknown training target or nonbinary label')
        raw.append(global_rate/rates[t] if y else (1-global_rate)/(1-rates[t]))
        labels.append(y)
    weights = [0.]*len(raw)
    # Solve scale within each class: final bounds [.5,2] AND class mean exactly one.
    for label in (0, 1):
        idx = [i for i, y in enumerate(labels) if y == label]
        if not idx:
            raise ValueError('Both sampled training classes required')
        lo, hi = 0., 2/min(raw[i] for i in idx)
        for _ in range(100):
            mid = (lo+hi)/2
            mass = sum(min(2., max(.5, mid*raw[i])) for i in idx)
            if mass < len(idx): lo = mid
            else: hi = mid
        for i in idx:
            weights[i] = min(2., max(.5, (lo+hi)/2*raw[i]))
        if shuffled:
            values = [weights[i] for i in idx]
            random.Random(seed+label).shuffle(values)
            for i, value in zip(idx, values): weights[i] = value
    if not all(math.isfinite(w) for w in weights):
        raise ValueError('Nonfinite weights')
    return weights

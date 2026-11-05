"""Training-label-aware protection of correct raw decisions; not causal proof."""
import torch


def adjusted_protection_loss(outputs, labels):
    raw = outputs['p_raw'].detach().clamp(1e-7, 1-1e-7)
    adjusted = outputs['p_do'].clamp(1e-7, 1-1e-7)
    correct = raw.argmax(dim=-1) == labels
    divergence = (raw * (raw.log()-adjusted.log())).sum(dim=-1)
    # Mean over the whole batch: no artificial amplification when correct subset is small.
    return (divergence * correct.to(divergence.dtype)).mean()

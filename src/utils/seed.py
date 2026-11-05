"""Seed helpers for reproducible smoke tests and dummy training runs."""

from __future__ import annotations

import random

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency.
    np = None

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency.
    torch = None


def set_seed(seed: int) -> None:
    """Seed Python and optional numerical backends."""
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

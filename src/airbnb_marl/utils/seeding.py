"""Seeding for reproducible runs."""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    """Seed random, numpy and torch (if installed), return a Generator."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return np.random.default_rng(seed)

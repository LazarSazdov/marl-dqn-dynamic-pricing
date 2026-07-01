"""Anchor policy: steer the price back toward the listing base price.

This is the observable part of real host behavior in the data (prices sit at
a chosen base and rarely move), so it serves both as the behavioral cloning
teacher for cold start and as a static baseline agent. Historical daily price
changes do not exist in any snapshot, which is why cloning targets this
anchoring rule instead.
"""

from __future__ import annotations

import numpy as np


def anchor_action(price: float, base_price: float, action_deltas: np.ndarray) -> int:
    """The action whose resulting price lands closest to the base price."""
    outcomes = price * (1.0 + np.asarray(action_deltas))
    return int(np.argmin(np.abs(outcomes - base_price)))

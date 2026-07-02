"""Profit Gain Index and the hypothesis tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def profit_gain_index(rl_profit: float, nash_profit: float,
                      monopoly_profit: float) -> float:
    """Delta = (pi_RL - pi_N) / (pi_M - pi_N).

    0 means perfect competition, 1 means full cartel. Values outside [0, 1]
    are possible (worse than Nash, or above the numeric monopoly bound) and
    are reported as they are.
    """
    return float(
        (rl_profit - nash_profit) / max(monopoly_profit - nash_profit, 1e-12)
    )


def run_delta(metrics_csv: str | Path, nash_profit: float,
              monopoly_profit: float, last_frac: float = 1 / 3) -> dict:
    """Delta for one training run, averaged over the last part of training
    (exploration has decayed there)."""
    df = pd.read_csv(metrics_csv)
    tail = df.iloc[int(len(df) * (1 - last_frac)):]
    rl_profit = float(tail["reward_per_agent_step"].mean())
    return {
        "rl_profit_per_agent_step": rl_profit,
        "delta": profit_gain_index(rl_profit, nash_profit, monopoly_profit),
        "episodes_used": int(len(tail)),
        "mean_price_ratio": float(tail["mean_price_ratio"].mean()),
    }


def h1_collusion_test(deltas: list[float], threshold: float = 0.15) -> dict:
    """One sided t test: is mean delta greater than the threshold?"""
    deltas = np.asarray(deltas, dtype=np.float64)
    t_stat, p_value = stats.ttest_1samp(deltas, popmean=threshold,
                                        alternative="greater")
    return {
        "mean_delta": float(deltas.mean()),
        "std_delta": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
        "n_runs": int(len(deltas)),
        "threshold": threshold,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "rejected_h0": bool(p_value < 0.05),
    }


def mann_whitney_greater(a: list[float], b: list[float]) -> dict:
    """One sided Mann Whitney U: are values in a greater than in b?"""
    stat, p_value = stats.mannwhitneyu(a, b, alternative="greater")
    return {
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "u_stat": float(stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }

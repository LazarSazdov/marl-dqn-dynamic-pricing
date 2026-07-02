"""Empirical Nash and monopoly bounds for a simulated market.

Profits use the same normalization as the training reward, booked revenue
divided by the listing base price, and are averaged over sampled days of
the year with occupancy fixed at each listing's historical level. The Nash
point comes from iterated best response until a fixed point (a single pass
optimization is not an equilibrium). The monopoly point maximizes the JOINT
profit by coordinate ascent from several starts.
"""

from __future__ import annotations

import numpy as np


def _profit_matrix(env, prices: np.ndarray, days: np.ndarray) -> np.ndarray:
    """Per agent expected normalized profit, averaged over the given days."""
    profits = np.zeros(env.n)
    for day in days:
        probs = env.demand_probs(prices, int(day))
        profits += (prices / env.base_prices) * probs
    return profits / len(days)


def _best_response(env, prices: np.ndarray, j: int, grid: np.ndarray,
                   days: np.ndarray, joint: bool = False) -> float:
    """Price on the grid maximizing agent j's (or the joint) profit."""
    best_price, best_value = prices[j], -np.inf
    for candidate in grid:
        trial = prices.copy()
        trial[j] = candidate
        profit = _profit_matrix(env, trial, days)
        value = float(profit.sum()) if joint else float(profit[j])
        if value > best_value:
            best_value, best_price = value, float(candidate)
    return best_price


def _iterate(env, start: np.ndarray, grid: np.ndarray, days: np.ndarray,
             joint: bool, max_rounds: int = 60) -> np.ndarray:
    """Round robin best responses until no agent moves (or averaging a cycle)."""
    prices = start.copy()
    history = [prices.copy()]
    for _ in range(max_rounds):
        moved = False
        for j in range(env.n):
            new_price = _best_response(env, prices, j, grid, days, joint=joint)
            if abs(new_price - prices[j]) > 1e-9:
                prices[j] = new_price
                moved = True
        if not moved:
            return prices
        for past in history:
            if np.allclose(past, prices):
                # limit cycle (Edgeworth style), average the cycle segment
                start_idx = next(
                    i for i, p in enumerate(history) if np.allclose(p, prices)
                )
                return np.mean(history[start_idx:], axis=0)
        history.append(prices.copy())
    return np.mean(history[-4:], axis=0)


def compute_bounds(env, n_days: int = 26, grid_step: float = 0.025) -> dict:
    """Nash and monopoly prices and profits for the env's market."""
    lo = env.cfg["price_min_ratio"] * env.cluster_median
    hi = env.cfg["price_max_ratio"] * env.cluster_median
    grid = np.arange(lo, hi + 1e-9, grid_step * env.cluster_median)
    days = np.linspace(0, env.episode_length - 1, n_days).astype(int)

    nash_prices = _iterate(env, env.base_prices.copy(), grid, days, joint=False)
    nash_profit = _profit_matrix(env, nash_prices, days)

    starts = [
        nash_prices.copy(),
        np.full(env.n, hi, dtype=np.float64),
        env.base_prices.copy(),
    ]
    best_prices, best_joint = None, -np.inf
    for start in starts:
        candidate = _iterate(env, start, grid, days, joint=True)
        joint_profit = _profit_matrix(env, candidate, days).sum()
        if joint_profit > best_joint:
            best_joint, best_prices = float(joint_profit), candidate
    monopoly_prices = best_prices
    monopoly_profit = _profit_matrix(env, monopoly_prices, days)

    return {
        "n_agents": env.n,
        "cluster_median": env.cluster_median,
        "nash_prices": nash_prices.tolist(),
        "nash_price_ratios": (nash_prices / env.cluster_median).round(4).tolist(),
        "nash_profit_per_agent_step": float(nash_profit.mean()),
        "monopoly_prices": monopoly_prices.tolist(),
        "monopoly_price_ratios": (monopoly_prices / env.cluster_median).round(4).tolist(),
        "monopoly_profit_per_agent_step": float(monopoly_profit.mean()),
        "n_days_sampled": int(n_days),
        "grid_step_ratio": grid_step,
    }

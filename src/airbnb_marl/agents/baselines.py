"""Static baseline policies and their evaluation.

These policies read the environment state directly (own price, base price,
rival prices), which is fine because they are evaluated, never trained.
"""

from __future__ import annotations

import numpy as np

from airbnb_marl.agents.anchor import anchor_action


def random_policy(env, rng: np.random.Generator) -> dict:
    return {a: int(rng.integers(len(env.action_deltas))) for a in env.agents}


def median_seeker_policy(env, rng=None) -> dict:
    """Move own price toward the median of rival prices."""
    actions = {}
    for j, agent in enumerate(env.possible_agents):
        rival_median = float(np.median(np.delete(env.prices, j)))
        outcomes = env.prices[j] * (1.0 + env.action_deltas)
        actions[agent] = int(np.argmin(np.abs(outcomes - rival_median)))
    return actions


def anchor_policy(env, rng=None) -> dict:
    """Hold the listing base price (the static human heuristic)."""
    return {
        agent: anchor_action(env.prices[j], env.base_prices[j], env.action_deltas)
        for j, agent in enumerate(env.possible_agents)
    }


BASELINES = {
    "random": random_policy,
    "median_seeker": median_seeker_policy,
    "anchor": anchor_policy,
}


def evaluate_policy(env, policy_fn, episodes: int = 3, seed: int = 0) -> dict:
    """Mean per agent per step reward and price ratio for a fixed policy."""
    rng = np.random.default_rng(seed)
    rewards, ratios = [], []
    for _ in range(episodes):
        env.reset(seed=int(rng.integers(1 << 31)))
        while env.agents:
            _, r, _, _, _ = env.step(policy_fn(env, rng))
            rewards.append(sum(r.values()) / env.n)
            ratios.append(float(np.mean(env.prices)) / env.cluster_median)
    return {
        "profit_per_agent_step": float(np.mean(rewards)),
        "mean_price_ratio": float(np.mean(ratios)),
    }


def evaluate_all_baselines(env, episodes: int = 3, seed: int = 0) -> dict:
    return {
        name: evaluate_policy(env, fn, episodes=episodes, seed=seed)
        for name, fn in BASELINES.items()
    }

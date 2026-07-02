"""TQL, PPO, baselines, bounds, and the Profit Gain Index."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from airbnb_marl.agents.baselines import evaluate_all_baselines
from airbnb_marl.agents.ppo import PPOAgent
from airbnb_marl.agents.tql import TQLAgent
from airbnb_marl.analysis.bounds import compute_bounds
from airbnb_marl.analysis.metrics import (
    h1_collusion_test,
    mann_whitney_greater,
    profit_gain_index,
    run_delta,
)
from airbnb_marl.env.market_env import PricingMarketEnv
from airbnb_marl.training.loop import run_seed

from tests.test_env import ENV_CFG, StubDemand, _listings

TQL_CFG = {
    "gamma": 0.95, "alpha": 0.2,
    "epsilon_start": 1.0, "epsilon_min": 0.05, "epsilon_decay": 0.8,
    "bins": {"price_ratio": 8, "market_dispersion": 4, "market_trend": 4,
             "occupancy": 4, "season": 6, "is_weekend": 2},
}

PPO_CFG = {
    "gamma": 0.95, "lr": 0.003, "clip_range": 0.2, "gae_lambda": 0.95,
    "rollout_steps": 64, "minibatch_size": 32, "update_epochs": 3,
    "entropy_coef": 0.01, "value_coef": 0.5, "max_grad_norm": 0.5,
    "hidden_dim": 32,
}


def test_tql_learns_state_dependent_bandit():
    rng = np.random.default_rng(0)
    agent = TQLAgent(5, TQL_CFG, rng)
    lo = np.array([0.6, 0.1, 0.0, 0.5, 0.0, 1.0, 0.0], dtype=np.float32)
    hi = np.array([1.9, 0.1, 0.0, 0.5, 0.0, 1.0, 0.0], dtype=np.float32)
    for _ in range(500):
        for state, good in ((lo, 4), (hi, 0)):
            a = int(rng.integers(5))
            agent.update(state, a, 1.0 if a == good else 0.0, state, True)
    assert agent.act(lo, epsilon=0.0) == 4
    assert agent.act(hi, epsilon=0.0) == 0


def test_tql_index_within_table():
    agent = TQLAgent(5, TQL_CFG, np.random.default_rng(0))
    rng = np.random.default_rng(1)
    for _ in range(200):
        state = np.array([
            rng.uniform(0, 6), rng.uniform(0, 3), rng.uniform(-0.5, 0.5),
            rng.uniform(0, 1), rng.uniform(-1, 1), rng.uniform(-1, 1),
            float(rng.integers(2)),
        ], dtype=np.float32)
        assert 0 <= agent._index(state) < len(agent.q)


def test_ppo_update_runs_and_improves_bandit():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    agent = PPOAgent(4, 5, PPO_CFG, rng)
    s = np.zeros(4, dtype=np.float32)
    for _ in range(12):
        while not agent.rollout_full():
            a = agent.act(s)
            agent.store(s, a, 1.0 if a == 2 else 0.0, False)
        agent.update(s)
    counts = np.bincount([agent.act(s) for _ in range(200)], minlength=5)
    assert counts[2] > 120  # the paying action dominates the policy


def test_tql_and_ppo_full_runs_write_metrics(tmp_path):
    for algo, block in (("tql", {"tql": TQL_CFG}), ("ppo", {"ppo": PPO_CFG})):
        cfg = {"algorithm": algo, "total_steps": 2 * ENV_CFG["episode_length"],
               "logging": {"checkpoint_every_episodes": 1}, **block}
        env = PricingMarketEnv(StubDemand(), _listings(2), ENV_CFG)
        out = tmp_path / algo
        summary = run_seed(cfg, env, out, seed=3, log=lambda *_: None)
        assert summary["algorithm"] == algo
        df = pd.read_csv(out / "metrics.csv")
        assert len(df) == 2
        trace = np.load(out / "eval_trace.npz")
        assert trace["prices"].shape == (ENV_CFG["episode_length"], 2)


def test_baselines_have_expected_ordering():
    env = PricingMarketEnv(StubDemand(), _listings(3), ENV_CFG)
    results = evaluate_all_baselines(env, episodes=2, seed=0)
    assert set(results) == {"random", "median_seeker", "anchor"}
    for r in results.values():
        assert r["profit_per_agent_step"] > 0
    # the median seeker holds ratio near 1 by construction
    assert abs(results["median_seeker"]["mean_price_ratio"] - 1.0) < 0.25


def test_bounds_monopoly_at_least_nash():
    env = PricingMarketEnv(StubDemand(), _listings(2), ENV_CFG)
    bounds = compute_bounds(env, n_days=4, grid_step=0.1)
    assert bounds["monopoly_profit_per_agent_step"] >= \
        bounds["nash_profit_per_agent_step"] - 1e-9
    for ratio in bounds["nash_price_ratios"] + bounds["monopoly_price_ratios"]:
        assert 0.4 <= ratio <= 2.6


def test_profit_gain_index_anchors():
    assert profit_gain_index(0.3, 0.3, 0.6) == pytest.approx(0.0)
    assert profit_gain_index(0.6, 0.3, 0.6) == pytest.approx(1.0)
    assert profit_gain_index(0.45, 0.3, 0.6) == pytest.approx(0.5)


def test_run_delta_uses_last_third(tmp_path):
    df = pd.DataFrame({
        "reward_per_agent_step": [0.1] * 20 + [0.5] * 10,
        "mean_price_ratio": [1.0] * 30,
    })
    path = tmp_path / "metrics.csv"
    df.to_csv(path, index=False)
    result = run_delta(path, nash_profit=0.2, monopoly_profit=0.6)
    assert result["rl_profit_per_agent_step"] == pytest.approx(0.5)
    assert result["delta"] == pytest.approx(0.75)


def test_hypothesis_helpers():
    strong = [0.4, 0.5, 0.45, 0.55, 0.5, 0.42, 0.48, 0.52]
    weak = [0.05, 0.1, 0.02, 0.08, 0.06, 0.04, 0.09, 0.03]
    assert h1_collusion_test(strong)["rejected_h0"]
    assert not h1_collusion_test(weak)["rejected_h0"]
    mw = mann_whitney_greater(strong, weak)
    assert mw["significant"]

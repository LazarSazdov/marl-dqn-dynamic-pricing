"""D3QN agent, replay buffer, anchor teacher, and the training loop."""

import csv

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from airbnb_marl.agents.anchor import anchor_action
from airbnb_marl.agents.dqn import D3QNAgent, DuelingDQN, ReplayBuffer
from airbnb_marl.env.market_env import PricingMarketEnv
from airbnb_marl.training.loop import pretrain_bc, run_seed

from tests.test_env import ENV_CFG, StubDemand, _listings

DQN_CFG = {
    "gamma": 0.95,
    "lr": 0.001,
    "batch_size": 32,
    "buffer_size": 1000,
    "learning_starts": 50,
    "train_every": 1,
    "target_update_every": 100,
    "epsilon_start": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.9,
    "hidden_dim": 32,
    "dueling": True,
    "double": True,
    "bc_pretrain": False,
}


def test_dueling_shapes_and_grad():
    net = DuelingDQN(7, 5, hidden_dim=16)
    q = net(torch.randn(4, 7))
    assert q.shape == (4, 5)
    q.sum().backward()


def test_replay_buffer_wraps_and_samples():
    buf = ReplayBuffer(10, 3, np.random.default_rng(0))
    for i in range(25):
        buf.add(np.full(3, i, dtype=np.float32), i % 5, float(i), np.zeros(3), False)
    assert buf.size == 10
    states, actions, rewards, _, _ = buf.sample(8)
    assert states.shape == (8, 3)
    assert rewards.min() >= 15.0  # oldest entries were overwritten


def test_agent_learns_trivial_bandit():
    # single state, action 3 always pays 1, others pay 0
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    agent = D3QNAgent(4, 5, DQN_CFG, rng)
    s = np.zeros(4, dtype=np.float32)
    for _ in range(400):
        a = int(rng.integers(5))
        agent.store(s, a, 1.0 if a == 3 else 0.0, s, True)
    for _ in range(300):
        agent.learn()
    assert agent.act(s, epsilon=0.0) == 3


def test_anchor_action_pulls_toward_base():
    deltas = np.array([-0.10, -0.05, 0.0, 0.05, 0.10])
    assert anchor_action(150.0, 100.0, deltas) == 0   # way above base: -10%
    assert anchor_action(104.0, 100.0, deltas) == 1   # a bit above: -5%
    assert anchor_action(100.0, 100.0, deltas) == 2   # at base: hold
    assert anchor_action(95.5, 100.0, deltas) == 3    # a bit below: +5%
    assert anchor_action(60.0, 100.0, deltas) == 4    # far below: +10%


def test_bc_pretrain_matches_teacher():
    env = PricingMarketEnv(StubDemand(), _listings(2), ENV_CFG)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    agents = {a: D3QNAgent(7, 5, DQN_CFG, rng) for a in env.possible_agents}
    logs = []
    pretrain_bc(agents, env, epochs=6, rng=rng, log=logs.append)
    matches = [float(line.split("teacher match ")[1].rstrip("%")) for line in logs]
    assert all(m > 60.0 for m in matches)


def test_run_seed_writes_metrics_checkpoint_and_resumes(tmp_path):
    cfg = {
        "algorithm": "dqn",
        "total_steps": 3 * ENV_CFG["episode_length"],
        "dqn": DQN_CFG,
        "logging": {"checkpoint_every_episodes": 1},
    }
    env = PricingMarketEnv(StubDemand(), _listings(2), ENV_CFG)
    summary = run_seed(cfg, env, tmp_path, seed=1, log=lambda *_: None)
    assert summary["episodes"] == 3
    assert (tmp_path / "checkpoints" / "latest.pt").exists()

    with open(tmp_path / "metrics.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3
    assert float(rows[-1]["epsilon"]) < 1.0
    assert all(float(r["reward_per_agent_step"]) >= 0 for r in rows)

    # a rerun resumes from the checkpoint instead of restarting
    env2 = PricingMarketEnv(StubDemand(), _listings(2), ENV_CFG)
    logs = []
    summary2 = run_seed(cfg, env2, tmp_path, seed=1, log=logs.append)
    assert any("resumed from episode 3" in line for line in logs)
    assert summary2["episodes"] == 3

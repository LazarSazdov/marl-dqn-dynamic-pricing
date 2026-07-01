"""Generic multi agent training loop for one experiment seed.

Handles behavioral cloning pretraining, epsilon greedy exploration with per
episode decay, target network syncs, per episode CSV metrics, checkpoints
and resume. Episode truncation at the horizon is stored as done, standard
DQN practice for time limited tasks.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from airbnb_marl.agents.anchor import anchor_action
from airbnb_marl.agents.dqn import D3QNAgent

CSV_FIELDS = [
    "episode", "global_step", "epsilon", "mean_loss",
    "reward_per_agent_step", "mean_price", "mean_price_ratio",
    "mean_booking_prob", "boundary_hits", "seconds",
]


def pretrain_bc(agents: dict, env, epochs: int, rng: np.random.Generator,
                episodes: int = 4, explore: float = 0.3, log=print) -> None:
    """Clone the anchor policy into every agent network.

    Rolls the environment under the anchor teacher with exploration noise,
    then fits each network to the teacher actions with cross entropy on the
    Q values used as logits.
    """
    states, labels = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(1 << 31)))
        while env.agents:
            actions = {}
            for j, agent in enumerate(env.possible_agents):
                teacher = anchor_action(
                    env.prices[j], env.base_prices[j], env.action_deltas
                )
                states.append(obs[agent])
                labels.append(teacher)
                actions[agent] = (
                    int(rng.integers(len(env.action_deltas)))
                    if rng.random() < explore else teacher
                )
            obs, _, _, _, _ = env.step(actions)

    X = torch.from_numpy(np.asarray(states, dtype=np.float32))
    y = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    for name, agent in agents.items():
        optimizer = torch.optim.Adam(agent.online.parameters(), lr=1e-3)
        for epoch in range(epochs):
            perm = torch.randperm(len(X))
            total = 0.0
            for start in range(0, len(perm), 512):
                idx = perm[start : start + 512]
                loss = F.cross_entropy(agent.online(X[idx]), y[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(idx)
            if epoch == epochs - 1:
                acc = float((agent.online(X).argmax(1) == y).float().mean())
                log(f"  bc {name}: final loss {total / len(X):.4f}, "
                    f"teacher match {acc:.2%}")
        agent.sync_target()


def run_seed(cfg: dict, env, out_dir: Path, seed: int, log=print,
             resume: bool = True) -> dict:
    """Train all agents in `env` for one seed. Returns a summary dict."""
    out_dir = Path(out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.csv"

    algo_cfg = cfg["dqn"]
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    state_dim = env.observation_space(env.possible_agents[0]).shape[0]
    n_actions = env.action_space(env.possible_agents[0]).n
    agents = {
        name: D3QNAgent(state_dim, n_actions, algo_cfg, np.random.default_rng(rng.integers(1 << 31)))
        for name in env.possible_agents
    }

    episode_length = env.episode_length
    n_episodes = max(1, int(np.ceil(cfg["total_steps"] / episode_length)))
    start_episode = 0
    epsilon = float(algo_cfg["epsilon_start"])

    latest = ckpt_dir / "latest.pt"
    if resume and latest.exists():
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        for name, agent in agents.items():
            agent.load_state_dict(payload["agents"][name])
        start_episode = payload["episode"] + 1
        epsilon = payload["epsilon"]
        log(f"  resumed from episode {start_episode}")
    elif algo_cfg.get("bc_pretrain", False):
        log("  behavioral cloning pretrain (anchor policy)")
        pretrain_bc(agents, env, epochs=5, rng=rng, log=log)

    write_header = not metrics_path.exists() or start_episode == 0
    csv_file = open(metrics_path, "w" if start_episode == 0 else "a",
                    newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()

    global_step = start_episode * episode_length
    for episode in range(start_episode, n_episodes):
        t_start = time.time()
        obs, _ = env.reset(seed=int(rng.integers(1 << 31)))
        ep_reward = 0.0
        ep_prices, ep_ratios, ep_probs, losses = [], [], [], []

        while env.agents:
            actions = {a: agents[a].act(obs[a], epsilon) for a in env.agents}
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            done = not env.agents
            for a in agents:
                agents[a].store(obs[a], actions[a], rewards[a], next_obs[a], done)
            obs = next_obs

            if (global_step >= algo_cfg["learning_starts"]
                    and global_step % algo_cfg["train_every"] == 0):
                for a in agents:
                    loss = agents[a].learn()
                    if loss is not None:
                        losses.append(loss)
            if global_step % algo_cfg["target_update_every"] == 0:
                for a in agents:
                    agents[a].sync_target()

            global_step += 1
            ep_reward += sum(rewards.values())
            ep_prices.append(np.mean([i["price"] for i in infos.values()]))
            ep_probs.append(np.mean([i["booking_prob"] for i in infos.values()]))
            ep_ratios.append(float(np.mean(env.prices)) / env.cluster_median)

        epsilon = max(
            float(algo_cfg["epsilon_min"]),
            epsilon * float(algo_cfg["epsilon_decay"]),
        )

        writer.writerow({
            "episode": episode,
            "global_step": global_step,
            "epsilon": round(epsilon, 4),
            "mean_loss": round(float(np.mean(losses)), 6) if losses else "",
            "reward_per_agent_step": round(
                ep_reward / (len(agents) * episode_length), 6
            ),
            "mean_price": round(float(np.mean(ep_prices)), 2),
            "mean_price_ratio": round(float(np.mean(ep_ratios)), 4),
            "mean_booking_prob": round(float(np.mean(ep_probs)), 4),
            "boundary_hits": int(env.boundary_hits.sum()),
            "seconds": round(time.time() - t_start, 2),
        })
        csv_file.flush()

        if (episode + 1) % cfg["logging"]["checkpoint_every_episodes"] == 0 \
                or episode == n_episodes - 1:
            torch.save({
                "agents": {a: agents[a].state_dict() for a in agents},
                "episode": episode,
                "epsilon": epsilon,
            }, latest)

    csv_file.close()
    summary = {
        "seed": seed,
        "episodes": n_episodes,
        "total_steps": global_step,
        "final_epsilon": epsilon,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary

"""Training loops for one experiment seed, all algorithms.

Every algorithm writes the same metrics.csv schema so evaluation and
figures are uniform. After training, one greedy evaluation episode is
recorded to eval_trace.npz (per step prices, actions, rewards, booking
probabilities) for the trajectory and joint action figures. The horizon
is a time limit rather than a terminal state, so stored transitions keep
done False and targets bootstrap through the episode boundary.
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
from airbnb_marl.agents.ppo import PPOAgent
from airbnb_marl.agents.tql import TQLAgent

CSV_FIELDS = [
    "episode", "global_step", "epsilon", "mean_loss",
    "reward_per_agent_step", "mean_price", "mean_price_ratio",
    "mean_booking_prob", "boundary_hits", "seconds",
]


class _EpisodeLogger:
    def __init__(self, out_dir: Path, fresh: bool):
        self.path = out_dir / "metrics.csv"
        self.file = open(self.path, "w" if fresh else "a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=CSV_FIELDS)
        if fresh:
            self.writer.writeheader()

    def write(self, **row) -> None:
        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def _record_eval_trace(env, act_fn, out_dir: Path, seed: int) -> None:
    """One greedy episode, per step arrays for figures."""
    obs, _ = env.reset(seed=seed)
    prices, actions_log, rewards_log, probs_log = [], [], [], []
    while env.agents:
        actions = {a: act_fn(a, obs[a]) for a in env.agents}
        obs, rewards, _, _, infos = env.step(actions)
        prices.append([infos[a]["price"] for a in env.possible_agents])
        actions_log.append([actions[a] for a in env.possible_agents])
        rewards_log.append([rewards[a] for a in env.possible_agents])
        probs_log.append([infos[a]["booking_prob"] for a in env.possible_agents])
    np.savez_compressed(
        out_dir / "eval_trace.npz",
        prices=np.asarray(prices),
        actions=np.asarray(actions_log),
        rewards=np.asarray(rewards_log),
        booking_probs=np.asarray(probs_log),
        cluster_median=env.cluster_median,
        base_prices=env.base_prices,
    )


def _write_summary(out_dir: Path, **summary) -> dict:
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def pretrain_bc(agents: dict, env, epochs: int, rng: np.random.Generator,
                episodes: int = 4, explore: float = 0.3, log=print) -> None:
    """Clone the anchor policy into every DQN network."""
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

    X_cpu = torch.from_numpy(np.asarray(states, dtype=np.float32))
    y_cpu = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    for name, agent in agents.items():
        X = X_cpu.to(agent.device)
        y = y_cpu.to(agent.device)
        optimizer = torch.optim.Adam(agent.online.parameters(), lr=1e-3)
        for epoch in range(epochs):
            perm = torch.randperm(len(X), device=agent.device)
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
    algorithm = cfg["algorithm"]
    out_dir = Path(out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    if algorithm == "dqn":
        return _run_dqn(cfg, env, out_dir, seed, log, resume)
    if algorithm == "tql":
        return _run_tql(cfg, env, out_dir, seed, log)
    if algorithm == "ppo":
        return _run_ppo(cfg, env, out_dir, seed, log)
    raise ValueError(f"unknown algorithm {algorithm}")


def _run_dqn(cfg, env, out_dir, seed, log, resume) -> dict:
    algo_cfg = cfg["dqn"]
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    ckpt = out_dir / "checkpoints" / "latest.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dim = env.observation_space(env.possible_agents[0]).shape[0]
    n_actions = env.action_space(env.possible_agents[0]).n
    agents = {
        name: D3QNAgent(state_dim, n_actions, algo_cfg,
                        np.random.default_rng(rng.integers(1 << 31)),
                        device=device)
        for name in env.possible_agents
    }

    episode_length = env.episode_length
    n_episodes = max(1, int(np.ceil(cfg["total_steps"] / episode_length)))
    start_episode = 0
    epsilon = float(algo_cfg["epsilon_start"])

    if resume and ckpt.exists():
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        for name, agent in agents.items():
            agent.load_state_dict(payload["agents"][name])
        start_episode = payload["episode"] + 1
        epsilon = payload["epsilon"]
        log(f"  resumed from episode {start_episode}")
    elif algo_cfg.get("bc_pretrain", False):
        log("  behavioral cloning pretrain (anchor policy)")
        pretrain_bc(agents, env, epochs=5, rng=rng, log=log)

    logger = _EpisodeLogger(out_dir, fresh=start_episode == 0)
    global_step = start_episode * episode_length
    for episode in range(start_episode, n_episodes):
        t_start = time.time()
        obs, _ = env.reset(seed=int(rng.integers(1 << 31)))
        ep_reward, losses = 0.0, []
        ep_prices, ep_ratios, ep_probs = [], [], []

        while env.agents:
            actions = {a: agents[a].act(obs[a], epsilon) for a in env.agents}
            next_obs, rewards, _, _, infos = env.step(actions)
            # the horizon is a time limit, not a terminal state, so targets
            # bootstrap through it (stored done stays False)
            for a in agents:
                agents[a].store(obs[a], actions[a], rewards[a], next_obs[a], False)
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

        epsilon = max(float(algo_cfg["epsilon_min"]),
                      epsilon * float(algo_cfg["epsilon_decay"]))
        logger.write(
            episode=episode, global_step=global_step, epsilon=round(epsilon, 4),
            mean_loss=round(float(np.mean(losses)), 6) if losses else "",
            reward_per_agent_step=round(ep_reward / (env.n * episode_length), 6),
            mean_price=round(float(np.mean(ep_prices)), 2),
            mean_price_ratio=round(float(np.mean(ep_ratios)), 4),
            mean_booking_prob=round(float(np.mean(ep_probs)), 4),
            boundary_hits=int(env.boundary_hits.sum()),
            seconds=round(time.time() - t_start, 2),
        )
        if (episode + 1) % 50 == 0 or episode == n_episodes - 1:
            log(f"  episode {episode + 1}/{n_episodes} "
                f"eps {epsilon:.3f} reward {ep_reward / (env.n * episode_length):.4f} "
                f"ratio {float(np.mean(ep_ratios)):.3f}")
        if (episode + 1) % cfg["logging"]["checkpoint_every_episodes"] == 0 \
                or episode == n_episodes - 1:
            torch.save({
                "agents": {a: agents[a].state_dict() for a in agents},
                "episode": episode, "epsilon": epsilon,
            }, ckpt)
    logger.close()

    _record_eval_trace(
        env, lambda a, s: agents[a].act(s, epsilon=0.0), out_dir,
        seed=int(rng.integers(1 << 31)),
    )
    return _write_summary(out_dir, seed=seed, algorithm="dqn",
                          episodes=n_episodes, total_steps=global_step,
                          final_epsilon=epsilon)


def _run_tql(cfg, env, out_dir, seed, log) -> dict:
    algo_cfg = cfg["tql"]
    rng = np.random.default_rng(seed)
    n_actions = env.action_space(env.possible_agents[0]).n
    agents = {
        name: TQLAgent(n_actions, algo_cfg,
                       np.random.default_rng(rng.integers(1 << 31)))
        for name in env.possible_agents
    }

    episode_length = env.episode_length
    n_episodes = max(1, int(np.ceil(cfg["total_steps"] / episode_length)))
    epsilon = float(algo_cfg["epsilon_start"])
    logger = _EpisodeLogger(out_dir, fresh=True)
    global_step = 0

    for episode in range(n_episodes):
        t_start = time.time()
        obs, _ = env.reset(seed=int(rng.integers(1 << 31)))
        ep_reward, tds = 0.0, []
        ep_prices, ep_ratios, ep_probs = [], [], []
        while env.agents:
            actions = {a: agents[a].act(obs[a], epsilon) for a in env.agents}
            next_obs, rewards, _, _, infos = env.step(actions)
            for a in agents:
                tds.append(agents[a].update(obs[a], actions[a], rewards[a],
                                            next_obs[a], False))
            obs = next_obs
            global_step += 1
            ep_reward += sum(rewards.values())
            ep_prices.append(np.mean([i["price"] for i in infos.values()]))
            ep_probs.append(np.mean([i["booking_prob"] for i in infos.values()]))
            ep_ratios.append(float(np.mean(env.prices)) / env.cluster_median)

        epsilon = max(float(algo_cfg["epsilon_min"]),
                      epsilon * float(algo_cfg["epsilon_decay"]))
        logger.write(
            episode=episode, global_step=global_step, epsilon=round(epsilon, 4),
            mean_loss=round(float(np.mean(tds)), 6),
            reward_per_agent_step=round(ep_reward / (env.n * episode_length), 6),
            mean_price=round(float(np.mean(ep_prices)), 2),
            mean_price_ratio=round(float(np.mean(ep_ratios)), 4),
            mean_booking_prob=round(float(np.mean(ep_probs)), 4),
            boundary_hits=int(env.boundary_hits.sum()),
            seconds=round(time.time() - t_start, 2),
        )
        if (episode + 1) % 50 == 0 or episode == n_episodes - 1:
            log(f"  episode {episode + 1}/{n_episodes} "
                f"eps {epsilon:.3f} reward {ep_reward / (env.n * episode_length):.4f}")
    logger.close()

    _record_eval_trace(env, lambda a, s: agents[a].act(s, epsilon=0.0),
                       out_dir, seed=int(rng.integers(1 << 31)))
    return _write_summary(out_dir, seed=seed, algorithm="tql",
                          episodes=n_episodes, total_steps=global_step,
                          final_epsilon=epsilon)


def _run_ppo(cfg, env, out_dir, seed, log) -> dict:
    algo_cfg = cfg["ppo"]
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    state_dim = env.observation_space(env.possible_agents[0]).shape[0]
    n_actions = env.action_space(env.possible_agents[0]).n
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agents = {
        name: PPOAgent(state_dim, n_actions, algo_cfg,
                       np.random.default_rng(rng.integers(1 << 31)),
                       device=device)
        for name in env.possible_agents
    }

    episode_length = env.episode_length
    n_episodes = max(1, int(np.ceil(cfg["total_steps"] / episode_length)))
    logger = _EpisodeLogger(out_dir, fresh=True)
    global_step = 0

    for episode in range(n_episodes):
        t_start = time.time()
        obs, _ = env.reset(seed=int(rng.integers(1 << 31)))
        ep_reward, losses = 0.0, []
        ep_prices, ep_ratios, ep_probs = [], [], []
        while env.agents:
            actions = {a: agents[a].act(obs[a]) for a in env.agents}
            next_obs, rewards, _, _, infos = env.step(actions)
            for a in agents:
                agents[a].store(obs[a], actions[a], rewards[a], False)
                if agents[a].rollout_full():
                    losses.append(agents[a].update(next_obs[a])["loss"])
            obs = next_obs
            global_step += 1
            ep_reward += sum(rewards.values())
            ep_prices.append(np.mean([i["price"] for i in infos.values()]))
            ep_probs.append(np.mean([i["booking_prob"] for i in infos.values()]))
            ep_ratios.append(float(np.mean(env.prices)) / env.cluster_median)

        logger.write(
            episode=episode, global_step=global_step, epsilon="",
            mean_loss=round(float(np.mean(losses)), 6) if losses else "",
            reward_per_agent_step=round(ep_reward / (env.n * episode_length), 6),
            mean_price=round(float(np.mean(ep_prices)), 2),
            mean_price_ratio=round(float(np.mean(ep_ratios)), 4),
            mean_booking_prob=round(float(np.mean(ep_probs)), 4),
            boundary_hits=int(env.boundary_hits.sum()),
            seconds=round(time.time() - t_start, 2),
        )
        if (episode + 1) % 50 == 0 or episode == n_episodes - 1:
            log(f"  episode {episode + 1}/{n_episodes} "
                f"reward {ep_reward / (env.n * episode_length):.4f}")
    logger.close()

    def greedy(agent_name, state):
        agent = agents[agent_name]
        with torch.no_grad():
            logits, _ = agent.net(
                torch.from_numpy(np.asarray(state, dtype=np.float32))
                .unsqueeze(0).to(agent.device)
            )
        return int(logits.argmax(dim=-1).item())

    _record_eval_trace(env, greedy, out_dir, seed=int(rng.integers(1 << 31)))
    return _write_summary(out_dir, seed=seed, algorithm="ppo",
                          episodes=n_episodes, total_steps=global_step)

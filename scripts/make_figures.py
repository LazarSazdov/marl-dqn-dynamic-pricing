#!/usr/bin/env python3
"""Render the report figures from the experiment results.

Figure 1: learning curves per experiment (mean over seeds, one std band,
          Nash and monopoly reference lines)
Figure 2: Profit Gain Index distribution per experiment (violins)
Figure 3: price trajectories from the greedy evaluation episode
Figure 4: joint action heatmap (2 agent experiments)

Everything is read from results/, figures land in results/figures/.

Usage:
    python3 scripts/make_figures.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from airbnb_marl.analysis.plotting import (
    INK_SECONDARY,
    PALETTE,
    SEQUENTIAL_CMAP,
    apply_style,
)
from airbnb_marl.utils.paths import results_dir

ALGO_COLOR = {"dqn": PALETTE[0], "ppo": PALETTE[1], "tql": PALETTE[2]}
ACTION_LABELS = ["-10%", "-5%", "0%", "+5%", "+10%"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _experiments() -> list[Path]:
    return sorted((results_dir() / "experiments").glob("E*"))


def _bounds_for(exp_dir: Path) -> dict | None:
    config = json.loads((exp_dir / "config.json").read_text())
    path = exp_dir.parent / f"bounds_n{config['experiment']['n_agents']}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _seed_frames(exp_dir: Path) -> list[pd.DataFrame]:
    return [pd.read_csv(p) for p in sorted(exp_dir.glob("seed_*/metrics.csv"))]


def learning_curves(out_dir: Path) -> None:
    for exp_dir in _experiments():
        frames = _seed_frames(exp_dir)
        if not frames:
            continue
        config = json.loads((exp_dir / "config.json").read_text())
        algo = config["experiment"]["algorithm"]
        bounds = _bounds_for(exp_dir)
        n = min(len(f) for f in frames)
        rewards = np.stack([f["reward_per_agent_step"].to_numpy()[:n] for f in frames])
        window = max(1, n // 40)
        mean = pd.Series(rewards.mean(axis=0)).rolling(window, min_periods=1).mean()
        std = pd.Series(rewards.std(axis=0)).rolling(window, min_periods=1).mean()

        fig, ax = plt.subplots(figsize=(9, 4.2))
        episodes = np.arange(n)
        color = ALGO_COLOR.get(algo, PALETTE[0])
        ax.plot(episodes, mean, color=color, label=f"{algo} mean ({len(frames)} seeds)")
        ax.fill_between(episodes, mean - std, mean + std, color=color, alpha=0.2)
        if bounds:
            ax.axhline(bounds["nash_profit_per_agent_step"], color=PALETTE[3],
                       linestyle="--", linewidth=1.5, label="Nash")
            ax.axhline(bounds["monopoly_profit_per_agent_step"], color=PALETTE[5],
                       linestyle="--", linewidth=1.5, label="Monopoly")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Profit per agent per step (normalized)")
        ax.set_title(f"{exp_dir.name}: learning curve")
        ax.legend()
        fig.savefig(out_dir / f"learning_curve_{exp_dir.name}.png")
        plt.close(fig)
        log(f"  learning_curve_{exp_dir.name}.png")


def delta_violins(out_dir: Path) -> None:
    deltas_path = results_dir() / "evaluation" / "deltas.csv"
    if not deltas_path.exists():
        log("  skipping violins, run scripts/evaluate.py first")
        return
    deltas = pd.read_csv(deltas_path)
    experiments = sorted(deltas["experiment"].unique())
    data = [deltas.loc[deltas["experiment"] == e, "delta"].to_numpy()
            for e in experiments]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    parts = ax.violinplot(data, showmeans=True, showmedians=True)
    for body, exp in zip(parts["bodies"], experiments):
        algo = deltas.loc[deltas["experiment"] == exp, "algorithm"].iloc[0]
        body.set_facecolor(ALGO_COLOR.get(algo, PALETTE[0]))
        body.set_alpha(0.7)
    ax.axhline(0.0, color=PALETTE[3], linestyle="--", linewidth=1.5, label="Nash (0)")
    ax.axhline(1.0, color=PALETTE[5], linestyle="--", linewidth=1.5, label="Monopoly (1)")
    ax.axhline(0.15, color=PALETTE[7], linestyle=":", linewidth=1.5, label="H1 threshold")
    ax.set_xticks(range(1, len(experiments) + 1))
    ax.set_xticklabels(experiments)
    ax.set_ylabel("Profit Gain Index")
    ax.set_title("Profit Gain Index distribution per experiment")
    ax.legend()
    fig.savefig(out_dir / "delta_violins.png")
    plt.close(fig)
    log("  delta_violins.png")


def price_trajectories(out_dir: Path, experiment: str = "E1") -> None:
    exp_dir = results_dir() / "experiments" / experiment
    traces = sorted(exp_dir.glob("seed_*/eval_trace.npz"))
    if not traces:
        log(f"  skipping trajectories, no eval traces in {experiment}")
        return
    trace = np.load(traces[0])
    prices = trace["prices"]
    bounds = _bounds_for(exp_dir)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    for j in range(prices.shape[1]):
        ax.plot(prices[:, j], color=PALETTE[j % len(PALETTE)],
                label=f"agent {j}", alpha=0.9)
    if bounds:
        ax.axhline(float(np.mean(bounds["nash_prices"])), color=PALETTE[3],
                   linestyle="--", linewidth=1.5, label="Nash mean price")
        ax.axhline(float(np.mean(bounds["monopoly_prices"])), color=PALETTE[5],
                   linestyle="--", linewidth=1.5, label="Monopoly mean price")
    ax.set_xlabel("Day")
    ax.set_ylabel("Price (GBP)")
    ax.set_title(f"{experiment}: greedy policy price trajectory (first seed)")
    ax.legend()
    fig.savefig(out_dir / f"price_trajectory_{experiment}.png")
    plt.close(fig)
    log(f"  price_trajectory_{experiment}.png")


def joint_action_heatmap(out_dir: Path, experiment: str = "E1") -> None:
    exp_dir = results_dir() / "experiments" / experiment
    counts = np.zeros((5, 5), dtype=np.int64)
    found = False
    for trace_path in exp_dir.glob("seed_*/eval_trace.npz"):
        actions = np.load(trace_path)["actions"]
        if actions.shape[1] < 2:
            continue
        found = True
        for a0, a1 in actions[:, :2]:
            counts[int(a0), int(a1)] += 1
    if not found:
        log(f"  skipping heatmap, no eval traces in {experiment}")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(counts, cmap=SEQUENTIAL_CMAP, origin="lower")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(counts[i, j]), ha="center", va="center",
                    fontsize=9, color=INK_SECONDARY)
    ax.set_xticks(range(5), ACTION_LABELS)
    ax.set_yticks(range(5), ACTION_LABELS)
    ax.set_xlabel("Agent 1 action")
    ax.set_ylabel("Agent 0 action")
    ax.set_title(f"{experiment}: joint actions in greedy evaluation")
    ax.grid(visible=False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(out_dir / f"joint_actions_{experiment}.png")
    plt.close(fig)
    log(f"  joint_actions_{experiment}.png")


def main() -> int:
    apply_style()
    out_dir = results_dir() / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    log("rendering figures")
    learning_curves(out_dir)
    delta_violins(out_dir)
    for experiment in ("E1", "E2", "E3", "E4", "E5"):
        if (results_dir() / "experiments" / experiment).exists():
            price_trajectories(out_dir, experiment)
    joint_action_heatmap(out_dir, "E1")
    log(f"figures in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Aggregate all experiment runs: Profit Gain Index per seed, hypothesis tests.

Reads results/experiments/<EXP>/seed_*/metrics.csv and the matching
bounds_n<agents>.json, computes delta per seed over the last third of
training, then runs:
  H1  DQN 2 agents collude (one sided t test, delta > 0.15)
  H2  delta depends on the algorithm (Kruskal Wallis over DQN, PPO, TQL)
  H3  DQN 2 agents > DQN 4 agents on delta (Mann Whitney)

Writes results/evaluation/deltas.csv and results/evaluation/summary.json.

Usage:
    python3 scripts/evaluate.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from airbnb_marl.analysis.metrics import (
    h1_collusion_test,
    kruskal_groups,
    mann_whitney_greater,
    run_delta,
)
from airbnb_marl.utils.paths import results_dir


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    exp_root = results_dir() / "experiments"
    rows = []
    for exp_dir in sorted(exp_root.glob("E*")):
        config_path = exp_dir / "config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text())
        n_agents = config["experiment"]["n_agents"]
        bounds_path = exp_root / f"bounds_n{n_agents}.json"
        if not bounds_path.exists():
            log(f"skipping {exp_dir.name}: {bounds_path.name} missing "
                f"(run scripts/compute_bounds.py)")
            continue
        bounds = json.loads(bounds_path.read_text())

        episode_length = config["env"]["episode_length"]
        expected_episodes = max(1, -(-config["experiment"]["total_steps"] // episode_length))
        for seed_dir in sorted(exp_dir.glob("seed_*")):
            metrics_csv = seed_dir / "metrics.csv"
            if not metrics_csv.exists():
                continue
            n_episodes = sum(1 for _ in open(metrics_csv)) - 1
            if n_episodes < expected_episodes:
                log(f"skipping {exp_dir.name}/{seed_dir.name}: incomplete run "
                    f"({n_episodes}/{expected_episodes} episodes)")
                continue
            result = run_delta(
                metrics_csv,
                bounds["nash_profit_per_agent_step"],
                bounds["monopoly_profit_per_agent_step"],
                trace_path=seed_dir / "eval_trace.npz",
            )
            rows.append({
                "experiment": exp_dir.name,
                "algorithm": config["experiment"]["algorithm"],
                "n_agents": n_agents,
                "seed": int(seed_dir.name.split("_")[1]),
                **result,
            })

    if not rows:
        log("no completed runs found")
        return 1
    deltas = pd.DataFrame(rows)
    out_dir = results_dir() / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    deltas.to_csv(out_dir / "deltas.csv", index=False)
    log(f"wrote deltas.csv ({len(deltas)} runs)")

    MIN_SEEDS = 5  # below this the tests are meaningless, do not report them

    def exp_deltas(name: str) -> list[float]:
        values = deltas.loc[deltas["experiment"] == name, "delta"].tolist()
        if 0 < len(values) < MIN_SEEDS:
            log(f"{name}: only {len(values)} seeds, hypothesis tests need "
                f"at least {MIN_SEEDS}, skipping")
            return []
        return values

    summary: dict = {
        "per_experiment": {
            name: {
                "n_runs": int(len(grp)),
                "mean_delta": float(grp["delta"].mean()),
                "std_delta": float(grp["delta"].std(ddof=1)) if len(grp) > 1 else 0.0,
                "mean_price_ratio": float(grp["mean_price_ratio"].mean()),
                "mean_profit": float(grp["rl_profit_per_agent_step"].mean()),
            }
            for name, grp in deltas.groupby("experiment")
        },
        "hypothesis_tests": {},
    }

    if exp_deltas("E1"):
        summary["hypothesis_tests"]["H1_dqn_collusion"] = h1_collusion_test(exp_deltas("E1"))
    if exp_deltas("E1") and exp_deltas("E3") and exp_deltas("E4"):
        summary["hypothesis_tests"]["H2_algorithm_dependence"] = kruskal_groups({
            "dqn_E1": exp_deltas("E1"),
            "ppo_E3": exp_deltas("E3"),
            "tql_E4": exp_deltas("E4"),
        })
    # direction expected by the literature, reported as a secondary check
    if exp_deltas("E4") and exp_deltas("E1"):
        summary["hypothesis_tests"]["literature_order_tql_gt_dqn"] = mann_whitney_greater(
            exp_deltas("E4"), exp_deltas("E1"))
    if exp_deltas("E1") and exp_deltas("E3"):
        summary["hypothesis_tests"]["literature_order_dqn_gt_ppo"] = mann_whitney_greater(
            exp_deltas("E1"), exp_deltas("E3"))
    if exp_deltas("E1") and exp_deltas("E2"):
        summary["hypothesis_tests"]["H3_n2_gt_n4"] = mann_whitney_greater(
            exp_deltas("E1"), exp_deltas("E2"))

    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    log("wrote summary.json")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

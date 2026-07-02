#!/usr/bin/env python3
"""Compute the Nash and monopoly bounds for an experiment's market.

E1, E3, E4 and E5 share the 2 agent market so their bounds are identical;
the output file is keyed by agent count. Baseline policies are evaluated on
the same market for the comparison table.

Usage:
    python3 scripts/compute_bounds.py --config configs/experiments/E1_dqn_n2.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from airbnb_marl.agents.baselines import evaluate_all_baselines
from airbnb_marl.analysis.bounds import compute_bounds
from airbnb_marl.config import load_config
from airbnb_marl.demand.interface import DemandModel
from airbnb_marl.env.market import select_market
from airbnb_marl.env.market_env import PricingMarketEnv
from airbnb_marl.utils.paths import processed_dir, results_dir


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--baseline-episodes", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    env_cfg = load_config("configs/env.yaml", overrides=cfg.get("env") or {})
    artifact = args.artifact_dir or (results_dir() / "demand")
    demand_model = DemandModel.load(artifact)
    listings = pd.read_parquet(processed_dir() / "listings_clean.parquet")
    market = select_market(
        listings,
        n_agents=cfg["n_agents"],
        neighbourhood=cfg["market"]["neighbourhood"],
        room_type=cfg["market"]["room_type"],
    )
    env = PricingMarketEnv(demand_model, market, env_cfg)

    log(f"computing bounds for n_agents={cfg['n_agents']} "
        f"market {market['listing_id'].tolist()}")
    bounds = compute_bounds(env)
    log(f"  nash ratios {bounds['nash_price_ratios']} "
        f"profit {bounds['nash_profit_per_agent_step']:.4f}")
    log(f"  monopoly ratios {bounds['monopoly_price_ratios']} "
        f"profit {bounds['monopoly_profit_per_agent_step']:.4f}")

    log("evaluating baseline policies")
    bounds["baselines"] = evaluate_all_baselines(
        env, episodes=args.baseline_episodes, seed=123
    )
    for name, result in bounds["baselines"].items():
        log(f"  {name}: profit {result['profit_per_agent_step']:.4f} "
            f"ratio {result['mean_price_ratio']:.3f}")

    bounds["market_listing_ids"] = market["listing_id"].tolist()
    out_dir = results_dir() / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"bounds_n{cfg['n_agents']}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(bounds, fh, indent=2)
    log(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

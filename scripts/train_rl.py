#!/usr/bin/env python3
"""Run one RL experiment config (all seeds, or a single seed).

Builds the market from the processed listings, loads the demand artifact,
and trains independent agents per the experiment config. Each seed writes
results/experiments/<experiment>/seed_<n>/ with metrics.csv, checkpoints
and summary.json. Reruns resume from the latest checkpoint.

Usage:
    python3 scripts/train_rl.py --config configs/experiments/E1_dqn_n2.yaml
    python3 scripts/train_rl.py --config configs/experiments/E1_dqn_n2.yaml \
        --override configs/smoke/rl_overrides.yaml --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from airbnb_marl.config import load_config
from airbnb_marl.demand.interface import DemandModel
from airbnb_marl.env.market import select_market
from airbnb_marl.env.market_env import PricingMarketEnv
from airbnb_marl.training.loop import run_seed
from airbnb_marl.utils.paths import processed_dir, results_dir


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="run only this seed instead of all")
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    env_cfg = load_config("configs/env.yaml", overrides=cfg.get("env") or {})
    if cfg["algorithm"] != "dqn":
        raise SystemExit(
            f"algorithm {cfg['algorithm']} lands in phase 5, only dqn runs for now"
        )

    artifact = args.artifact_dir or (results_dir() / "demand")
    demand_model = DemandModel.load(artifact)
    listings = pd.read_parquet(processed_dir() / "listings_clean.parquet")
    market = select_market(
        listings,
        n_agents=cfg["n_agents"],
        neighbourhood=cfg["market"]["neighbourhood"],
        room_type=cfg["market"]["room_type"],
    )
    log(
        f"market: {market['listing_id'].tolist()} "
        f"base prices {market['price_gbp'].round(0).tolist()} "
        f"span {market.attrs['group_span_km']:.3f} km"
    )

    exp_dir = results_dir() / "experiments" / cfg["experiment"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "config.json", "w", encoding="utf-8") as fh:
        json.dump({"experiment": cfg.to_dict(), "env": env_cfg.to_dict(),
                   "market_listing_ids": market["listing_id"].tolist()}, fh, indent=2)

    seeds = ([args.seed] if args.seed is not None
             else [cfg["seed_base"] + i for i in range(cfg["n_seeds"])])
    for seed in seeds:
        log(f"=== {cfg['experiment']} seed {seed} ===")
        env = PricingMarketEnv(
            demand_model, market, env_cfg,
            inflation_rate=(cfg["inflation"]["annual_rate"]
                            if cfg["inflation"]["enabled"] else 0.0),
        )
        summary = run_seed(
            cfg.to_dict(), env, exp_dir / f"seed_{seed}", seed,
            log=log, resume=not args.no_resume,
        )
        log(f"  done: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

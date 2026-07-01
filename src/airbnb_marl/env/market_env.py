"""PettingZoo parallel environment for the simulated pricing market.

Each agent is one listing. Every step all agents pick a relative price
change at the same time, the demand model turns each price into a booking
probability, and bookings are sampled. Observations for the next decision
are built from the prices just set, so an agent only ever sees rival prices
from the previous step and the simultaneous action circularity is avoided.

Rewards are booked revenue normalized by the listing base price, which
keeps scales comparable across listings and avoids long flat zero reward
stretches destabilizing the Q values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from airbnb_marl.features.schema import (
    DEMAND_FEATURES,
    build_demand_features,
    temporal_features,
)

OBS_NAMES = [
    "price_ratio",        # own price / median of rival prices (previous step)
    "market_dispersion",  # rival price std / median (previous step)
    "market_trend",       # one step relative change of the market median
    "occupancy_recent",   # own booked share over the occupancy window
    "sin_doy",
    "cos_doy",
    "is_weekend_night",
]

_OBS_LOW = np.array([0.0, 0.0, -0.5, 0.0, -1.0, -1.0, 0.0], dtype=np.float32)
_OBS_HIGH = np.array([6.0, 3.0, 0.5, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)


class PricingMarketEnv(ParallelEnv):
    metadata = {"name": "airbnb_pricing_v1"}

    def __init__(
        self,
        demand_model,
        listings: pd.DataFrame,
        env_cfg: dict,
        start_date: str = "2026-06-19",
        inflation_rate: float = 0.0,
    ):
        """
        demand_model: object with predict_proba_features(X) over DEMAND_FEATURES
        listings: one row per agent with the raw static columns and price_gbp
        env_cfg: configs/env.yaml content
        inflation_rate: yearly drift applied to the price bounds and the
            reward normalizer (exogenous shock experiment)
        """
        self.demand_model = demand_model
        self.cfg = env_cfg
        self.inflation_rate = float(inflation_rate)
        self.action_deltas = np.asarray(env_cfg["action_deltas"], dtype=np.float64)
        self.episode_length = int(env_cfg["episode_length"])
        self.occupancy_window = int(env_cfg["occupancy_window"])

        self.n = len(listings)
        if self.n < 2:
            raise ValueError("the market needs at least 2 agents")
        self.possible_agents = [f"agent_{i}" for i in range(self.n)]
        self.agents: list[str] = []

        self.base_prices = listings["price_gbp"].to_numpy(np.float64)
        self.cluster_median = float(np.median(self.base_prices))

        # static part of the demand features, dynamic columns get overwritten
        # every step through the shared schema indices
        raw = listings.copy().reset_index(drop=True)
        raw["price_gbp"] = self.base_prices
        raw["competitor_median_price"] = self.cluster_median
        raw["competitor_price_std"] = float(np.std(self.base_prices))
        raw["occupancy_recent"] = 0.5
        raw["date"] = pd.Timestamp(start_date)
        raw["lead_days"] = int(env_cfg["reference_lead_days"])
        self._X_base = build_demand_features(raw).to_numpy(np.float64)

        self._ix = {name: DEMAND_FEATURES.index(name) for name in (
            "price_ratio", "log_price", "market_dispersion", "occupancy_recent",
            "sin_doy", "cos_doy", "sin_dow", "cos_dow", "is_weekend_night",
            "hist_occupancy",
        )}

        dates = pd.date_range(start_date, periods=self.episode_length + 1, freq="D")
        self._temporal = temporal_features(pd.Series(dates)).to_numpy(np.float64)

        self._obs_space = spaces.Box(low=_OBS_LOW, high=_OBS_HIGH, dtype=np.float32)
        self._act_space = spaces.Discrete(len(self.action_deltas))
        self.observation_spaces = {a: self._obs_space for a in self.possible_agents}
        self.action_spaces = {a: self._act_space for a in self.possible_agents}

    def observation_space(self, agent):
        return self._obs_space

    def action_space(self, agent):
        return self._act_space

    def _inflation_factor(self) -> float:
        if self.inflation_rate == 0.0:
            return 1.0
        return float((1.0 + self.inflation_rate) ** (self.t / 365.0))

    def _rival_stats(self) -> tuple[np.ndarray, np.ndarray]:
        """Leave one out median and std of current prices, per agent."""
        medians = np.empty(self.n)
        stds = np.empty(self.n)
        for j in range(self.n):
            others = np.delete(self.prices, j)
            medians[j] = np.median(others)
            stds[j] = np.std(others)
        return medians, stds

    def _observations(self) -> dict:
        rival_median, rival_std = self._rival_stats()
        occ = self._occ_history.mean(axis=1)
        tf = self._temporal[self.t]
        obs = {}
        for j, agent in enumerate(self.possible_agents):
            vec = np.array([
                self.prices[j] / max(rival_median[j], 1e-8),
                rival_std[j] / max(rival_median[j], 1e-8),
                self._trend,
                occ[j],
                tf[0],   # sin_doy
                tf[1],   # cos_doy
                tf[4],   # is_weekend_night
            ], dtype=np.float32)
            obs[agent] = np.clip(vec, _OBS_LOW, _OBS_HIGH)
        return obs

    def reset(self, seed=None, options=None):
        self.rng = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        self.t = 0
        self.prices = self.base_prices.copy()
        self._market_median_prev = float(np.median(self.prices))
        self._trend = 0.0
        init_occ = self._X_base[:, self._ix["hist_occupancy"]]
        self._occ_history = np.tile(
            init_occ[:, None], (1, self.occupancy_window)
        ).astype(np.float64)
        self.boundary_hits = np.zeros(self.n, dtype=np.int64)

        observations = self._observations()
        infos = {a: {} for a in self.agents}
        return observations, infos

    def step(self, actions: dict):
        deltas = np.array([
            self.action_deltas[int(actions[a])] for a in self.possible_agents
        ])

        factor = self._inflation_factor()
        lo = self.cfg["price_min_ratio"] * self.cluster_median * factor
        hi = self.cfg["price_max_ratio"] * self.cluster_median * factor
        proposed = self.prices * (1.0 + deltas)
        self.prices = np.clip(proposed, lo, hi)
        self.boundary_hits += (proposed != self.prices).astype(np.int64)

        # demand features from the prices everyone just posted
        rival_median, rival_std = self._rival_stats()
        X = self._X_base.copy()
        X[:, self._ix["price_ratio"]] = self.prices / np.maximum(rival_median, 1e-8)
        X[:, self._ix["log_price"]] = np.log1p(self.prices)
        X[:, self._ix["market_dispersion"]] = rival_std / np.maximum(rival_median, 1e-8)
        X[:, self._ix["occupancy_recent"]] = self._occ_history.mean(axis=1)
        tf = self._temporal[self.t]
        X[:, self._ix["sin_doy"]] = tf[0]
        X[:, self._ix["cos_doy"]] = tf[1]
        X[:, self._ix["sin_dow"]] = tf[2]
        X[:, self._ix["cos_dow"]] = tf[3]
        X[:, self._ix["is_weekend_night"]] = tf[4]

        probs = np.asarray(self.demand_model.predict_proba_features(X), dtype=np.float64)
        booked = self.rng.random(self.n) < probs

        if self.cfg.get("normalize_reward", True):
            revenue = self.prices / (self.base_prices * factor)
        else:
            revenue = self.prices
        reward_vec = revenue * booked

        self._occ_history = np.roll(self._occ_history, -1, axis=1)
        self._occ_history[:, -1] = booked.astype(np.float64)

        market_median = float(np.median(self.prices))
        self._trend = float(np.clip(
            (market_median - self._market_median_prev)
            / max(self._market_median_prev, 1e-8),
            -0.5, 0.5,
        ))
        self._market_median_prev = market_median

        self.t += 1
        done = self.t >= self.episode_length

        rewards = {a: float(reward_vec[j]) for j, a in enumerate(self.possible_agents)}
        terminations = {a: False for a in self.possible_agents}
        truncations = {a: done for a in self.possible_agents}
        infos = {
            a: {
                "price": float(self.prices[j]),
                "booking_prob": float(probs[j]),
                "booked": bool(booked[j]),
                "boundary_hits": int(self.boundary_hits[j]),
            }
            for j, a in enumerate(self.possible_agents)
        }

        observations = self._observations()
        if done:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

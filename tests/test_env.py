"""Market environment: PettingZoo API compliance, determinism, market rules."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pettingzoo")

from airbnb_marl.env.market import select_market
from airbnb_marl.env.market_env import PricingMarketEnv
from airbnb_marl.features.schema import DEMAND_FEATURES

ENV_CFG = {
    "action_deltas": [-0.10, -0.05, 0.0, 0.05, 0.10],
    "price_min_ratio": 0.5,
    "price_max_ratio": 2.5,
    "episode_length": 40,
    "occupancy_window": 7,
    "competitor_price_lag": 1,
    "reference_lead_days": 30,
    "normalize_reward": True,
}

I_RATIO = DEMAND_FEATURES.index("price_ratio")


class StubDemand:
    """Deterministic logistic demand in the price ratio."""

    def __init__(self, prob=None):
        self.prob = prob

    def predict_proba_features(self, X, apply_correction=True):
        if self.prob is not None:
            return np.full(len(X), self.prob)
        return 0.3 / (1.0 + np.exp(2.0 * (X[:, I_RATIO] - 1.0)))


def _listings(n=3, price=100.0):
    return pd.DataFrame({
        "listing_id": np.arange(1, n + 1),
        "price_gbp": [price * (1 + 0.1 * i) for i in range(n)],
        "latitude": [51.5 + 0.001 * i for i in range(n)],
        "longitude": [-0.1] * n,
        "neighbourhood_cleansed": ["Westminster"] * n,
        "review_scores_rating": [4.8] * n,
        "number_of_reviews": [25] * n,
        "host_is_superhost": [False] * n,
        "accommodates": [2] * n,
        "room_type": ["Entire home/apt"] * n,
        "minimum_nights": [2] * n,
        "instant_bookable": [True] * n,
        "calculated_host_listings_count": [1] * n,
        "estimated_occupancy_l365d": [120] * n,
        "reviews_per_month": [1.0] * n,
    })


def _env(n=3, prob=None, **kwargs):
    return PricingMarketEnv(StubDemand(prob), _listings(n), ENV_CFG, **kwargs)


def test_parallel_api_compliance():
    from pettingzoo.test import parallel_api_test

    parallel_api_test(_env(), num_cycles=100)


def test_same_seed_same_trajectory():
    rewards_by_run = []
    for _ in range(2):
        env = _env()
        env.reset(seed=7)
        total = []
        rng = np.random.default_rng(0)
        for _ in range(20):
            actions = {a: int(rng.integers(5)) for a in env.agents}
            _, rewards, _, _, _ = env.step(actions)
            total.append(sum(rewards.values()))
        rewards_by_run.append(total)
    assert rewards_by_run[0] == rewards_by_run[1]


def test_prices_stay_inside_bounds_and_hits_are_counted():
    env = _env(prob=0.0)
    env.reset(seed=1)
    hi = ENV_CFG["price_max_ratio"] * env.cluster_median
    for _ in range(30):
        _, _, _, _, infos = env.step({a: 4 for a in env.agents})  # +10% forever
    for agent in env.possible_agents:
        assert infos[agent]["price"] <= hi + 1e-9
        assert infos[agent]["boundary_hits"] > 0


def test_observation_uses_rival_prices_from_last_step():
    env = _env(n=2, prob=0.0)
    env.reset(seed=3)
    obs, _, _, _, _ = env.step({"agent_0": 4, "agent_1": 0})  # +10% vs hold
    p0 = env.prices[0]
    p1 = env.prices[1]
    # for two agents the rival median is just the other price
    assert obs["agent_0"][0] == pytest.approx(p0 / p1, abs=1e-6)
    assert obs["agent_1"][0] == pytest.approx(p1 / p0, abs=1e-6)


def test_reward_is_normalized_booked_revenue():
    env = _env(prob=1.0)
    env.reset(seed=2)
    _, rewards, _, _, infos = env.step({a: 2 for a in env.agents})  # hold
    for j, agent in enumerate(env.possible_agents):
        assert infos[agent]["booked"]
        assert rewards[agent] == pytest.approx(env.prices[j] / env.base_prices[j])

    env_never = _env(prob=0.0)
    env_never.reset(seed=2)
    _, rewards, _, _, _ = env_never.step({a: 2 for a in env_never.agents})
    assert all(r == 0.0 for r in rewards.values())


def test_occupancy_fills_after_constant_booking():
    env = _env(prob=1.0)
    env.reset(seed=5)
    for _ in range(ENV_CFG["occupancy_window"]):
        obs, _, _, _, _ = env.step({a: 2 for a in env.agents})
    for agent in env.possible_agents:
        assert obs[agent][3] == pytest.approx(1.0)


def test_episode_truncates_at_horizon():
    env = _env()
    env.reset(seed=0)
    truncated = {}
    for _ in range(ENV_CFG["episode_length"]):
        _, _, terminations, truncated, _ = env.step({a: 2 for a in env.agents})
        assert not any(terminations.values())
    assert all(truncated.values())
    assert env.agents == []


def test_inflation_moves_price_ceiling():
    env = _env(prob=0.0, inflation_rate=0.10)
    env.reset(seed=1)
    for _ in range(ENV_CFG["episode_length"] - 1):
        env.step({a: 4 for a in env.agents})
    late_price = env.prices.max()
    plain = _env(prob=0.0)
    plain.reset(seed=1)
    for _ in range(ENV_CFG["episode_length"] - 1):
        plain.step({a: 4 for a in plain.agents})
    assert late_price > plain.prices.max()


def test_select_market_handles_identical_coordinates():
    listings = _listings(4)
    # airbnb anonymizes coordinates, identical points are common
    listings.loc[:, "latitude"] = 51.5
    listings.loc[:, "longitude"] = -0.1
    market = select_market(listings, n_agents=3)
    assert market["listing_id"].nunique() == 3


def test_select_market_prefers_dense_compatible_group():
    listings = _listings(6)
    # push one listing far away and make one incompatible in size
    listings.loc[5, "latitude"] = 52.5
    listings.loc[2, "accommodates"] = 8
    market = select_market(listings, n_agents=3, neighbourhood="Westminster",
                           room_type="Entire home/apt")
    assert len(market) == 3
    assert 6 not in market["listing_id"].tolist()  # the faraway listing
    assert 3 not in market["listing_id"].tolist()  # the incompatible one
    assert market.attrs["group_span_km"] < 1.0

"""Tabular Q-learning agent, the bridge to Calvano et al. 2020.

The continuous observation is discretized into a small grid. The table is
updated online after every transition.
"""

from __future__ import annotations

import numpy as np

# observation layout comes from market_env.OBS_NAMES:
# [price_ratio, market_dispersion, market_trend, occupancy, sin_doy, cos_doy, weekend]


class TQLAgent:
    def __init__(self, n_actions: int, cfg: dict, rng: np.random.Generator):
        self.cfg = cfg
        self.n_actions = n_actions
        self.rng = rng
        self.alpha = float(cfg["alpha"])
        self.gamma = float(cfg["gamma"])

        bins = cfg["bins"]
        self.edges_ratio = np.linspace(0.4, 2.6, int(bins["price_ratio"]) - 1)
        self.edges_disp = np.linspace(0.0, 0.6, int(bins["market_dispersion"]) - 1)
        self.edges_trend = np.linspace(-0.06, 0.06, int(bins["market_trend"]) - 1)
        self.edges_occ = np.linspace(0.0, 1.0, int(bins["occupancy"]) + 1)[1:-1]
        self.n_season = int(bins["season"])
        self.dims = (
            int(bins["price_ratio"]), int(bins["market_dispersion"]),
            int(bins["market_trend"]), int(bins["occupancy"]),
            self.n_season, int(bins["is_weekend"]),
        )
        self.q = np.zeros((int(np.prod(self.dims)), n_actions), dtype=np.float64)

    def _index(self, state: np.ndarray) -> int:
        ratio = int(np.digitize(state[0], self.edges_ratio))
        disp = int(np.digitize(state[1], self.edges_disp))
        trend = int(np.digitize(state[2], self.edges_trend))
        occ = int(np.digitize(state[3], self.edges_occ))
        angle = float(np.arctan2(state[4], state[5]))  # sin, cos of day of year
        season = int((angle + np.pi) / (2 * np.pi) * self.n_season) % self.n_season
        weekend = int(state[6] > 0.5)
        return int(np.ravel_multi_index(
            (ratio, disp, trend, occ, season, weekend), self.dims
        ))

    def act(self, state: np.ndarray, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(self.q[self._index(state)].argmax())

    def update(self, state, action, reward, next_state, done) -> float:
        s = self._index(state)
        s_next = self._index(next_state)
        target = reward + self.gamma * (1.0 - float(done)) * self.q[s_next].max()
        td = target - self.q[s, action]
        self.q[s, action] += self.alpha * td
        return float(abs(td))

    def state_dict(self) -> dict:
        return {"q": self.q.copy()}

    def load_state_dict(self, state: dict) -> None:
        self.q = np.asarray(state["q"], dtype=np.float64).copy()

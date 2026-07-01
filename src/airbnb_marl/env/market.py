"""Selection of the simulated market: a tight group of comparable listings."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


def select_market(
    listings: pd.DataFrame,
    n_agents: int,
    neighbourhood: str | None = None,
    room_type: str | None = None,
    accommodates_tol: int = 1,
) -> pd.DataFrame:
    """Pick n_agents comparable listings that sit closest together.

    Filters to the given neighbourhood and room type, then anchors on the
    listing whose (n_agents - 1)th nearest compatible neighbour is closest,
    which is the densest spot of the filtered market. Returns the anchor and
    its neighbours as a DataFrame, one row per agent.
    """
    pool = listings
    if neighbourhood is not None:
        pool = pool[pool["neighbourhood_cleansed"] == neighbourhood]
    if room_type is not None:
        pool = pool[pool["room_type"] == room_type]
    pool = pool.dropna(subset=["latitude", "longitude", "price_gbp"]).reset_index(drop=True)
    if len(pool) < n_agents:
        raise ValueError(
            f"only {len(pool)} listings match the market filter, need {n_agents}"
        )

    coords = np.radians(pool[["latitude", "longitude"]].to_numpy(np.float64))
    tree = BallTree(coords, metric="haversine")
    k_query = min(len(pool), max(n_agents * 5, n_agents + 1))
    distances, indices = tree.query(coords, k=k_query)

    accommodates = pool["accommodates"].to_numpy(np.int64)
    best_anchor, best_group, best_span = None, None, np.inf
    for i in range(len(pool)):
        group = [i]
        span = 0.0
        for j, d in zip(indices[i], distances[i]):
            # with anonymized coordinates several listings share the exact
            # same point, so self is not always at position 0
            if j == i or j in group:
                continue
            if abs(accommodates[j] - accommodates[i]) <= accommodates_tol:
                group.append(int(j))
                span = float(d)
                if len(group) == n_agents:
                    break
        if len(group) == n_agents and span < best_span:
            best_anchor, best_group, best_span = i, group, span

    if best_group is None:
        raise ValueError("no compatible group found, relax the filters")
    market = pool.iloc[best_group].reset_index(drop=True)
    market.attrs["anchor_listing_id"] = int(pool.iloc[best_anchor]["listing_id"])
    market.attrs["group_span_km"] = best_span * 6371.0
    return market

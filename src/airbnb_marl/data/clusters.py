"""K nearest geographic competitors per listing.

Competitors must share room_type and have accommodates within the tolerance.
Distances are haversine via BallTree on lat and lon in radians.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_RADIUS_KM = 6371.0


def build_clusters(
    listings: pd.DataFrame,
    k: int = 5,
    accommodates_tol: int = 1,
    query_multiplier: int = 10,
) -> pd.DataFrame:
    """One row per listing with its competitor set and price stats.

    The accommodates filter is applied after the spatial query, so each
    listing queries k * query_multiplier neighbours and keeps the first k
    that fit.
    """
    records = []
    for room_type, group in listings.groupby("room_type", observed=True):
        group = group.reset_index(drop=True)
        n = len(group)
        if n < 2:
            for _, row in group.iterrows():
                records.append(_record(row, [], [], group))
            continue

        coords = np.radians(group[["latitude", "longitude"]].to_numpy(np.float64))
        tree = BallTree(coords, metric="haversine")
        k_query = min(n, k * query_multiplier + 1)
        distances, indices = tree.query(coords, k=k_query)

        accommodates = group["accommodates"].to_numpy(np.int64)
        for i in range(n):
            # skip self; with duplicate coordinates it is not always index 0
            neighbor_idx = [j for j in indices[i] if j != i]
            neighbor_dist = [
                d for j, d in zip(indices[i], distances[i]) if j != i
            ]
            keep_idx, keep_dist = [], []
            for j, d in zip(neighbor_idx, neighbor_dist):
                if abs(accommodates[j] - accommodates[i]) <= accommodates_tol:
                    keep_idx.append(j)
                    keep_dist.append(d * EARTH_RADIUS_KM)
                    if len(keep_idx) == k:
                        break
            records.append(_record(group.iloc[i], keep_idx, keep_dist, group))

    return pd.DataFrame.from_records(records)


def _record(row: pd.Series, comp_idx: list, comp_dist_km: list, group: pd.DataFrame) -> dict:
    comp_prices = group["price_gbp"].to_numpy(np.float64)[comp_idx] if comp_idx else np.array([])
    comp_ids = group["listing_id"].to_numpy(np.int64)[comp_idx] if comp_idx else np.array([], dtype=np.int64)
    return {
        "listing_id": int(row["listing_id"]),
        "competitor_ids": comp_ids.tolist(),
        "n_competitors": len(comp_ids),
        "competitor_median_price": float(np.median(comp_prices)) if len(comp_prices) else np.nan,
        "competitor_price_std": float(np.std(comp_prices)) if len(comp_prices) else np.nan,
        "max_competitor_distance_km": float(comp_dist_km[-1]) if comp_dist_km else np.nan,
    }

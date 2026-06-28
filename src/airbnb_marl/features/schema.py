"""Shared demand model feature schema.

Both preprocessing and the market environment build model inputs through
build_demand_features, so training and simulation always use the same layout.
Lead time is a real feature because we train on two snapshot pairs from
different seasons; the simulation evaluates it at a fixed reference value
(configs/env.yaml, reference_lead_days).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ROOM_TYPE_ENCODING = {
    "Shared room": 0,
    "Private room": 1,
    "Hotel room": 1,
    "Entire home/apt": 2,
}

RAW_INPUT_COLUMNS = [
    "price_gbp",
    "competitor_median_price",
    "competitor_price_std",
    "occupancy_recent",
    "review_scores_rating",
    "number_of_reviews",
    "host_is_superhost",
    "accommodates",
    "room_type",
    "minimum_nights",
    "instant_bookable",
    "calculated_host_listings_count",
    "estimated_occupancy_l365d",
    "reviews_per_month",
    "date",
    "lead_days",
]

DEMAND_FEATURES = [
    "price_ratio",
    "log_price",
    "market_dispersion",
    "occupancy_recent",
    "review_score",
    "log_reviews",
    "is_superhost",
    "accommodates",
    "room_type_enc",
    "log_min_nights",
    "instant_bookable",
    "log_host_listings",
    "hist_occupancy",
    "log_reviews_rate",
    "sin_doy",
    "cos_doy",
    "sin_dow",
    "cos_dow",
    "is_weekend_night",
    "log_lead",
]

REVIEW_SCORE_FILL = 4.5


def temporal_features(dates: pd.Series) -> pd.DataFrame:
    """Cyclical calendar encodings. Weekend night means Friday or Saturday."""
    doy = dates.dt.dayofyear.to_numpy(dtype=np.float64)
    dow = dates.dt.dayofweek.to_numpy(dtype=np.float64)
    return pd.DataFrame(
        {
            "sin_doy": np.sin(2 * np.pi * doy / 365.25),
            "cos_doy": np.cos(2 * np.pi * doy / 365.25),
            "sin_dow": np.sin(2 * np.pi * dow / 7.0),
            "cos_dow": np.cos(2 * np.pi * dow / 7.0),
            "is_weekend_night": ((dow == 4) | (dow == 5)).astype(np.float64),
        },
        index=dates.index,
    )


def build_demand_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Map raw columns to the model feature matrix (float32, no NaN)."""
    missing = [c for c in RAW_INPUT_COLUMNS if c not in raw.columns]
    if missing:
        raise KeyError(f"build_demand_features: missing raw columns {missing}")

    comp_median = raw["competitor_median_price"].to_numpy(dtype=np.float64)
    price = raw["price_gbp"].to_numpy(dtype=np.float64)

    out = pd.DataFrame(index=raw.index)
    out["price_ratio"] = price / np.maximum(comp_median, 1e-8)
    out["log_price"] = np.log1p(price)
    out["market_dispersion"] = (
        raw["competitor_price_std"].to_numpy(dtype=np.float64)
        / np.maximum(comp_median, 1e-8)
    )
    out["occupancy_recent"] = raw["occupancy_recent"].astype(np.float64)

    out["review_score"] = (
        raw["review_scores_rating"].astype(np.float64).fillna(REVIEW_SCORE_FILL)
    )
    out["log_reviews"] = np.log1p(raw["number_of_reviews"].astype(np.float64))
    out["log_reviews"] = out["log_reviews"].fillna(0.0)
    out["is_superhost"] = (
        raw["host_is_superhost"].fillna(False).astype(np.float64)
    )
    out["accommodates"] = (
        raw["accommodates"].astype(np.float64).fillna(2.0)
    )
    out["room_type_enc"] = (
        raw["room_type"].map(ROOM_TYPE_ENCODING).fillna(1).astype(np.float64)
    )
    out["log_min_nights"] = np.log1p(
        raw["minimum_nights"].astype(np.float64).fillna(1).clip(lower=1, upper=365)
    )
    out["instant_bookable"] = (
        raw["instant_bookable"].fillna(False).astype(np.float64)
    )
    out["log_host_listings"] = np.log1p(
        raw["calculated_host_listings_count"].astype(np.float64).fillna(1).clip(lower=1)
    )
    out["hist_occupancy"] = (
        raw["estimated_occupancy_l365d"].astype(np.float64).fillna(0.0) / 365.0
    ).clip(0.0, 1.0)
    out["log_reviews_rate"] = np.log1p(
        raw["reviews_per_month"].astype(np.float64).fillna(0.0).clip(lower=0)
    )

    out = pd.concat([out, temporal_features(raw["date"])], axis=1)
    out["log_lead"] = np.log1p(
        raw["lead_days"].astype(np.float64).clip(lower=0)
    )

    out = out[DEMAND_FEATURES].astype(np.float32)
    if out.isna().any().any():
        bad = out.columns[out.isna().any()].tolist()
        raise ValueError(f"NaNs in demand features after fill: {bad}")
    return out

"""Shared demand feature schema."""

import numpy as np
import pandas as pd
import pytest

from airbnb_marl.features.schema import (
    DEMAND_FEATURES,
    RAW_INPUT_COLUMNS,
    build_demand_features,
    temporal_features,
)


def _raw(n=3):
    return pd.DataFrame({
        "price_gbp": [100.0, 200.0, 50.0][:n],
        "competitor_median_price": [100.0, 100.0, 100.0][:n],
        "competitor_price_std": [10.0, 20.0, 5.0][:n],
        "occupancy_recent": [0.5, 0.0, 1.0][:n],
        "review_scores_rating": [4.8, np.nan, 4.2][:n],
        "number_of_reviews": [10, 0, 250][:n],
        "host_is_superhost": [True, None, False][:n],
        "accommodates": [2, 4, np.nan][:n],
        "room_type": ["Entire home/apt", "Private room", "Unknown type"][:n],
        "minimum_nights": [1, np.nan, 1000][:n],
        "instant_bookable": [False, True, None][:n],
        "calculated_host_listings_count": [1, 40, 2][:n],
        "estimated_occupancy_l365d": [73, np.nan, 400][:n],
        "reviews_per_month": [1.2, np.nan, 0.3][:n],
        "date": pd.to_datetime(["2026-07-03", "2026-07-06", "2026-12-25"][:n]),
        "lead_days": [0, 30, 180][:n],
    })


def test_output_layout_and_no_nans():
    out = build_demand_features(_raw())
    assert list(out.columns) == DEMAND_FEATURES
    assert out.dtypes.eq(np.float32).all()
    assert not out.isna().any().any()


def test_price_ratio_and_fills():
    out = build_demand_features(_raw())
    assert out["price_ratio"].tolist() == pytest.approx([1.0, 2.0, 0.5])
    assert out["review_score"].iloc[1] == pytest.approx(4.5)  # median-ish fill
    assert out["is_superhost"].tolist() == [1.0, 0.0, 0.0]
    assert out["room_type_enc"].tolist() == [2.0, 1.0, 1.0]  # unknown -> 1
    assert out["log_min_nights"].iloc[2] == pytest.approx(np.log1p(365))  # capped
    assert out["log_lead"].tolist() == pytest.approx(
        [0.0, np.log1p(30), np.log1p(180)]
    )
    assert out["hist_occupancy"].tolist() == pytest.approx(
        [73 / 365, 0.0, 1.0]  # NaN -> 0, over-365 capped at 1
    )
    assert out["log_reviews_rate"].iloc[1] == 0.0  # NaN -> 0


def test_missing_raw_column_raises():
    raw = _raw().drop(columns=["occupancy_recent"])
    with pytest.raises(KeyError):
        build_demand_features(raw)


def test_weekend_nights():
    # 2026-07-03 = Friday, 2026-07-05 = Sunday, 2026-07-04 = Saturday
    dates = pd.Series(pd.to_datetime(["2026-07-03", "2026-07-04", "2026-07-05"]))
    tf = temporal_features(dates)
    assert tf["is_weekend_night"].tolist() == [1.0, 1.0, 0.0]
    for col in ("sin_doy", "cos_doy", "sin_dow", "cos_dow"):
        assert tf[col].abs().max() <= 1.0


def test_raw_input_columns_documented():
    assert set(RAW_INPUT_COLUMNS) == set(_raw().columns)

"""Competitor cluster construction."""

import numpy as np
import pandas as pd

from airbnb_marl.data.clusters import build_clusters


def _listings():
    # Five entire homes on a line (0.01 deg lon apart ~ 1.1 km at lat 0),
    # one with incompatible accommodates, plus one private room nearby.
    return pd.DataFrame({
        "listing_id": [1, 2, 3, 4, 5, 6],
        "latitude": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "longitude": [0.00, 0.01, 0.02, 0.03, 0.04, 0.001],
        "room_type": ["Entire home/apt"] * 5 + ["Private room"],
        "accommodates": [2, 2, 3, 2, 8, 2],
        "price_gbp": [100.0, 110.0, 120.0, 130.0, 500.0, 50.0],
    })


def test_room_type_and_accommodates_constraints():
    clusters = build_clusters(_listings(), k=3, accommodates_tol=1)
    row = clusters.set_index("listing_id").loc[1]
    # Listing 5 (accommodates 8) and listing 6 (private room) must be excluded.
    assert set(row["competitor_ids"]) == {2, 3, 4}
    assert row["n_competitors"] == 3
    assert row["competitor_median_price"] == 120.0


def test_self_never_a_competitor():
    clusters = build_clusters(_listings(), k=3, accommodates_tol=1)
    for _, row in clusters.iterrows():
        assert row["listing_id"] not in row["competitor_ids"]


def test_nearest_selected_first():
    clusters = build_clusters(_listings(), k=2, accommodates_tol=1)
    row = clusters.set_index("listing_id").loc[1]
    # The two nearest compatible neighbours of listing 1 are 2 and 3.
    assert set(row["competitor_ids"]) == {2, 3}


def test_isolated_room_type_gets_empty_cluster():
    clusters = build_clusters(_listings(), k=3, accommodates_tol=1)
    row = clusters.set_index("listing_id").loc[6]  # only private room
    assert row["n_competitors"] == 0
    assert np.isnan(row["competitor_median_price"])

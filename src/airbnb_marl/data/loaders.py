"""Readers for raw InsideAirbnb files.

Calendars run to tens of millions of rows, so they are read in chunks with
minimal columns and filtered while streaming.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from airbnb_marl.utils.paths import raw_snapshot_dir

LISTING_USECOLS = [
    "id",
    "price",
    "latitude",
    "longitude",
    "room_type",
    "accommodates",
    "minimum_nights",
    "number_of_reviews",
    "review_scores_rating",
    "host_is_superhost",
    "calculated_host_listings_count",
    "neighbourhood_cleansed",
    "instant_bookable",
    "last_review",
    "estimated_occupancy_l365d",
    "reviews_per_month",
]

_BOOL_TF = {"t": True, "f": False}


def parse_price(series: pd.Series) -> pd.Series:
    """Parse price strings like $1,234.56 into float."""
    return pd.to_numeric(
        series.astype("string").str.replace(r"[$,]", "", regex=True),
        errors="coerce",
    )


def load_listings(snapshot: str, usecols: list[str] | None = None) -> pd.DataFrame:
    """Load one snapshot's listings with parsed prices and clean dtypes."""
    path = raw_snapshot_dir(snapshot) / "listings.csv.gz"
    df = pd.read_csv(path, usecols=usecols or LISTING_USECOLS, low_memory=False)
    df = df.rename(columns={"id": "listing_id"})
    df["price_gbp"] = parse_price(df["price"])
    df = df.drop(columns=["price"])
    for col in ("host_is_superhost", "instant_bookable"):
        if col in df.columns:
            df[col] = df[col].map(_BOOL_TF).astype("boolean")
    if "last_review" in df.columns:
        df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")
    return df


def load_calendar_window(
    snapshot: str,
    date_max: pd.Timestamp | str | None = None,
    listing_ids: Iterable[int] | None = None,
    chunksize: int = 2_000_000,
) -> pd.DataFrame:
    """Stream a calendar, keeping only listing_id, date and available.

    Rows are filtered while streaming to dates up to date_max and optionally
    to the given listing ids.
    """
    path = raw_snapshot_dir(snapshot) / "calendar.csv.gz"
    date_max = pd.Timestamp(date_max) if date_max is not None else None
    id_set = None
    if listing_ids is not None:
        id_set = pd.Index(np.asarray(list(listing_ids), dtype=np.int64))

    parts = []
    reader = pd.read_csv(
        path,
        usecols=["listing_id", "date", "available"],
        dtype={"listing_id": np.int64, "available": "category"},
        parse_dates=["date"],
        chunksize=chunksize,
    )
    for chunk in reader:
        if date_max is not None:
            chunk = chunk[chunk["date"] <= date_max]
        if id_set is not None:
            chunk = chunk[chunk["listing_id"].isin(id_set)]
        if len(chunk):
            chunk = chunk.copy()
            chunk["available"] = (chunk["available"] == "t").astype(bool)
            parts.append(chunk)
    if not parts:
        return pd.DataFrame(columns=["listing_id", "date", "available"])
    return pd.concat(parts, ignore_index=True)


def load_review_dates(snapshot: str, chunksize: int = 2_000_000) -> pd.DataFrame:
    """Stream the reviews file, returning only listing_id and date."""
    path = raw_snapshot_dir(snapshot) / "reviews.csv.gz"
    parts = []
    reader = pd.read_csv(
        path,
        usecols=["listing_id", "date"],
        dtype={"listing_id": np.int64},
        parse_dates=["date"],
        chunksize=chunksize,
    )
    for chunk in reader:
        parts.append(chunk)
    return pd.concat(parts, ignore_index=True)

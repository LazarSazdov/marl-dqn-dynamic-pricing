"""Booking labels from a two snapshot calendar diff.

A night that was open at t0 and unavailable at t1 counts as booked during
the window. A night open in both is not booked. Nights already closed at t0
tell us nothing about demand at the observed price, so they are dropped.
The signal still mixes real bookings with host blocks; run length stats and
the paused listing filter keep that noise measurable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_labels(
    cal_t0: pd.DataFrame,
    cal_t1: pd.DataFrame,
    max_lead_days: int,
) -> tuple[pd.DataFrame, dict]:
    """Join the two calendars and derive labels, with exclusion counts."""
    t1_start = cal_t1["date"].min()
    label_end = t1_start + pd.Timedelta(days=max_lead_days)

    merged = cal_t0.merge(
        cal_t1,
        on=["listing_id", "date"],
        how="inner",
        suffixes=("_t0", "_t1"),
    )
    stats = {
        "t1_first_date": str(t1_start.date()),
        "label_window_end": str(label_end.date()),
        "nights_in_both_calendars": int(len(merged)),
    }

    in_window = merged["date"] <= label_end
    merged = merged[in_window]
    stats["nights_within_lead_window"] = int(len(merged))

    open_at_t0 = merged["available_t0"]
    labels = merged[open_at_t0].copy()
    stats["nights_excluded_closed_at_t0"] = int((~open_at_t0).sum())

    labels["booked"] = (~labels["available_t1"]).astype(np.int8)
    labels = labels[["listing_id", "date", "booked"]].reset_index(drop=True)

    stats["labelled_nights"] = int(len(labels))
    stats["booked_nights"] = int(labels["booked"].sum())
    stats["booking_rate"] = float(labels["booked"].mean()) if len(labels) else float("nan")
    return labels, stats


def detect_paused_listings(
    labels: pd.DataFrame,
    min_open_nights: int = 30,
    flip_rate_threshold: float = 0.95,
) -> tuple[np.ndarray, dict]:
    """Find listings whose whole open calendar flipped to unavailable.

    That pattern means the host paused or delisted, not that 30 plus nights
    sold out in under a month, so those nights are not booking labels.
    """
    per_listing = labels.groupby("listing_id")["booked"].agg(["size", "mean"])
    paused = per_listing[
        (per_listing["size"] >= min_open_nights)
        & (per_listing["mean"] > flip_rate_threshold)
    ]
    stats = {
        "paused_listings_detected": int(len(paused)),
        "paused_nights_removed": int(
            labels["listing_id"].isin(paused.index).sum()
        ),
        "min_open_nights": min_open_nights,
        "flip_rate_threshold": flip_rate_threshold,
    }
    return paused.index.to_numpy(), stats


def booked_run_lengths(labels: pd.DataFrame) -> dict:
    """Stats on consecutive booked night runs.

    Real bookings come in short runs close to stay length. Long streaks look
    like host blocks, so their share is our noise indicator.
    """
    booked = labels[labels["booked"] == 1].sort_values(["listing_id", "date"])
    if booked.empty:
        return {"runs": 0}
    same_listing = booked["listing_id"].eq(booked["listing_id"].shift())
    consecutive = booked["date"].diff().eq(pd.Timedelta(days=1))
    new_run = ~(same_listing & consecutive)
    run_id = new_run.cumsum()
    lengths = booked.groupby(run_id).size()

    nights_total = int(lengths.sum())
    return {
        "runs": int(len(lengths)),
        "median_run_nights": float(lengths.median()),
        "share_nights_in_runs_le_14": round(
            float(lengths[lengths <= 14].sum() / nights_total), 4
        ),
        "share_nights_in_runs_gt_28": round(
            float(lengths[lengths > 28].sum() / nights_total), 4
        ),
    }


def attach_recent_occupancy(
    labels: pd.DataFrame,
    cal_t0: pd.DataFrame,
    window: int = 7,
) -> pd.DataFrame:
    """Add occupancy_recent: share of the previous 7 nights unavailable at t0.

    The simulation computes the same quantity from simulated bookings.
    """
    cal = cal_t0.sort_values(["listing_id", "date"]).copy()
    cal["unavailable"] = ~cal["available"]
    rolled = (
        cal.groupby("listing_id")["unavailable"]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    )
    cal["occupancy_recent"] = rolled
    global_mean = float(cal["unavailable"].mean())
    cal["occupancy_recent"] = cal["occupancy_recent"].fillna(global_mean)

    out = labels.merge(
        cal[["listing_id", "date", "occupancy_recent"]],
        on=["listing_id", "date"],
        how="left",
    )
    out["occupancy_recent"] = out["occupancy_recent"].fillna(global_mean)
    return out


def review_noise_report(
    labels: pd.DataFrame,
    listings: pd.DataFrame,
    review_dates: pd.DataFrame,
    t0_scrape: pd.Timestamp,
    t1_scrape: pd.Timestamp,
    review_rate: float = 0.5,
    default_stay_nights: float = 3.0,
) -> dict:
    """Compare diff labelled bookings with review activity.

    Uses the InsideAirbnb assumption that about half of stays leave a review
    and a stay lasts max(minimum_nights, 3) nights. Reviews in the window
    reflect stays that were mostly booked before t0, while diff positives are
    bookings made during the window, so this is a volume sanity check rather
    than a direct noise bound.
    """
    window_reviews = review_dates[
        (review_dates["date"] > t0_scrape)
        & (review_dates["date"] <= t1_scrape + pd.Timedelta(days=7))
    ]
    reviews_per_listing = window_reviews.groupby("listing_id").size().rename("n_reviews_window")

    per_listing = (
        labels[labels["booked"] == 1]
        .groupby("listing_id")
        .size()
        .rename("diff_booked_nights")
        .to_frame()
        .join(reviews_per_listing, how="left")
        .fillna({"n_reviews_window": 0})
    )
    stay_nights = (
        listings.set_index("listing_id")["minimum_nights"]
        .clip(lower=default_stay_nights)
        .rename("stay_nights")
    )
    per_listing = per_listing.join(stay_nights, how="left")
    per_listing["stay_nights"] = per_listing["stay_nights"].fillna(default_stay_nights)
    per_listing["implied_booked_nights"] = (
        per_listing["n_reviews_window"] / review_rate * per_listing["stay_nights"]
    )

    total_diff = float(per_listing["diff_booked_nights"].sum())
    total_implied = float(per_listing["implied_booked_nights"].sum())
    corr = float(
        per_listing["diff_booked_nights"].corr(per_listing["implied_booked_nights"])
    ) if len(per_listing) > 2 else float("nan")

    return {
        "listings_with_diff_bookings": int(len(per_listing)),
        "diff_booked_nights_total": int(total_diff),
        "review_implied_booked_nights_window_stays": round(total_implied),
        "implied_over_diff_volume_ratio": round(total_implied / total_diff, 4)
        if total_diff
        else None,
        "per_listing_correlation": round(corr, 4),
        "review_rate_assumed": review_rate,
        "cohort_note": (
            "Reviews in the window reflect stays booked mostly before t0, "
            "while diff positives are bookings made during the window for "
            "future dates. Volume sanity check only; run lengths and the "
            "paused listing filter are the main noise controls."
        ),
    }

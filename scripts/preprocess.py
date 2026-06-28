#!/usr/bin/env python3
"""Preprocess raw InsideAirbnb snapshots into the demand model dataset.

For each snapshot pair in configs/data.yaml: load and filter listings,
build competitor clusters, diff the two calendars into booking labels,
drop paused listings and attach recent occupancy and lead time. Pairs are
then concatenated, features built through the shared schema, and split
chronologically within each pair. Writes parquet artifacts plus a JSON
report with every count.

Usage:
    python3 scripts/preprocess.py
    python3 scripts/preprocess.py --override configs/smoke/data_overrides.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from airbnb_marl.config import Config, load_config
from airbnb_marl.data.clusters import build_clusters
from airbnb_marl.data.labels import (
    attach_recent_occupancy,
    booked_run_lengths,
    build_labels,
    detect_paused_listings,
    review_noise_report,
)
from airbnb_marl.data.loaders import (
    load_calendar_window,
    load_listings,
    load_review_dates,
)
from airbnb_marl.features.schema import RAW_INPUT_COLUMNS, build_demand_features
from airbnb_marl.utils.paths import processed_dir, raw_snapshot_dir

META_COLUMNS = [
    "listing_id",
    "pair_id",
    "date",
    "lead_days",
    "booked",
    "split",
    "neighbourhood_cleansed",
    "price_gbp",
    "competitor_median_price",
]

STATIC_COLS = [
    "listing_id", "price_gbp", "competitor_median_price", "competitor_price_std",
    "review_scores_rating", "number_of_reviews", "host_is_superhost",
    "accommodates", "room_type", "minimum_nights", "instant_bookable",
    "calculated_host_listings_count", "estimated_occupancy_l365d",
    "reviews_per_month", "neighbourhood_cleansed",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_pair_dataset(
    t0_date: str, t1_date: str, prep: Config, run_review_noise: bool
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build the labelled night level dataset for one snapshot pair."""
    report: dict = {"t0": t0_date, "t1": t1_date, "stages": {}}
    stages = report["stages"]

    log(f"[pair {t1_date}] loading listings t1={t1_date}, t0={t0_date}")
    listings = load_listings(t1_date)
    stages["listings_raw"] = len(listings)

    t0_prices = load_listings(t0_date, usecols=["id", "price"])
    t0_prices = t0_prices.dropna(subset=["price_gbp"]).rename(
        columns={"price_gbp": "price_gbp_t0"}
    )

    listings = listings[listings["room_type"].isin(prep["room_types_keep"])]
    stages["after_room_type_filter"] = len(listings)

    listings = listings.dropna(subset=["latitude", "longitude"])
    stages["after_coordinates_filter"] = len(listings)

    active_cutoff = pd.Timestamp(t1_date) - pd.Timedelta(
        days=prep["active_review_window_days"]
    )
    listings = listings[listings["last_review"] >= active_cutoff]
    stages["after_review_activity_filter"] = len(listings)

    listings = listings.merge(
        t0_prices[["listing_id", "price_gbp_t0"]], on="listing_id", how="inner"
    )
    stages["after_present_in_t0_with_price"] = len(listings)

    # the t0 price is the price in force during the label window and becomes
    # the price feature, no nightly price series exists in any snapshot
    listings["price_gbp"] = listings["price_gbp_t0"]
    price_cap = float(listings["price_gbp"].quantile(prep["price_max_quantile"]))
    listings = listings[
        (listings["price_gbp"] >= prep["price_min"])
        & (listings["price_gbp"] <= price_cap)
    ]
    stages["after_price_sanity"] = len(listings)
    report["price_cap_gbp"] = round(price_cap, 2)

    if prep.get("sample_listings"):
        rng = np.random.default_rng(42)
        take = min(int(prep["sample_listings"]), len(listings))
        listings = listings.iloc[rng.permutation(len(listings))[:take]]
        stages["after_smoke_subsample"] = len(listings)

    log(f"[pair {t1_date}] building K={prep.k_competitors} clusters for {len(listings)} listings")
    clusters = build_clusters(
        listings,
        k=prep["k_competitors"],
        accommodates_tol=prep["accommodates_tolerance"],
    )
    listings = listings.merge(clusters, on="listing_id", how="left")
    listings = listings[listings["n_competitors"] >= prep["min_competitors"]]
    stages["after_min_competitors"] = len(listings)

    active_ids = listings["listing_id"].to_numpy()
    label_end = pd.Timestamp(t1_date) + pd.Timedelta(days=prep["max_lead_days"])

    log(f"[pair {t1_date}] loading calendars (dates <= {label_end.date()})")
    cal_t0 = load_calendar_window(t0_date, date_max=label_end, listing_ids=active_ids)
    cal_t1 = load_calendar_window(t1_date, date_max=label_end, listing_ids=active_ids)
    log(f"[pair {t1_date}]   t0 {len(cal_t0):,} rows, t1 {len(cal_t1):,} rows")

    labels, label_stats = build_labels(cal_t0, cal_t1, prep["max_lead_days"])
    report["labels"] = label_stats

    paused_ids, paused_stats = detect_paused_listings(labels)
    labels = labels[~labels["listing_id"].isin(paused_ids)]
    listings = listings[~listings["listing_id"].isin(paused_ids)]
    report["paused_listings"] = paused_stats
    stages["after_paused_listing_filter"] = len(listings)

    label_stats["labelled_nights_after_pause_filter"] = int(len(labels))
    label_stats["booking_rate_after_pause_filter"] = float(labels["booked"].mean())
    report["booked_run_lengths"] = booked_run_lengths(labels)
    log(
        f"[pair {t1_date}] labels: {len(labels):,} nights, booking rate "
        f"{label_stats['booking_rate_after_pause_filter']:.3f} "
        f"({paused_stats['paused_listings_detected']} paused listings removed)"
    )

    labels = attach_recent_occupancy(labels, cal_t0)

    if run_review_noise and (raw_snapshot_dir(t1_date) / "reviews.csv.gz").exists():
        log(f"[pair {t1_date}] scanning reviews for the noise report")
        review_dates = load_review_dates(t1_date)
        report["noise"] = review_noise_report(
            labels,
            listings,
            review_dates,
            t0_scrape=cal_t0["date"].min(),
            t1_scrape=pd.Timestamp(t1_date),
        )
        del review_dates
    del cal_t0, cal_t1

    dataset = labels.merge(listings[STATIC_COLS], on="listing_id", how="inner")
    dataset["pair_id"] = t1_date
    dataset["lead_days"] = (
        dataset["date"] - pd.Timestamp(t1_date)
    ).dt.days.clip(lower=0)
    return listings, dataset, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--override", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    prep = cfg.preprocessing
    primary = cfg["primary_snapshot"]
    report: dict = {"snapshot_pairs": cfg["snapshot_pairs"], "pairs": {}}

    primary_listings = None
    datasets = []
    for t0_date, t1_date in cfg["snapshot_pairs"]:
        listings, dataset, pair_report = build_pair_dataset(
            t0_date, t1_date, prep, run_review_noise=(t1_date == primary)
        )
        report["pairs"][t1_date] = pair_report
        datasets.append(dataset)
        if t1_date == primary:
            primary_listings = listings

    if primary_listings is None:
        raise SystemExit(f"primary snapshot {primary} not among snapshot_pairs")

    log("assembling combined demand dataset")
    dataset = pd.concat(datasets, ignore_index=True)
    features = build_demand_features(dataset[RAW_INPUT_COLUMNS])

    # chronological split within each pair, whole dates per split
    split_cfg = load_config("configs/demand.yaml")["split"]
    dataset["split"] = ""
    report["split_boundaries"] = {}
    for pair_id, grp in dataset.groupby("pair_id"):
        t_train = grp["date"].quantile(split_cfg["train_frac"])
        t_val = grp["date"].quantile(split_cfg["train_frac"] + split_cfg["val_frac"])
        dataset.loc[grp.index, "split"] = np.where(
            grp["date"] <= t_train, "train",
            np.where(grp["date"] <= t_val, "val", "test"),
        )
        report["split_boundaries"][pair_id] = {
            "train_end": str(t_train.date()), "val_end": str(t_val.date())
        }
    report["split_stats"] = {
        name: {"rows": int(len(g)), "booking_rate": round(float(g["booked"].mean()), 4)}
        for name, g in dataset.groupby("split")
    }
    report["dataset_rows"] = int(len(dataset))
    report["overall_booking_rate"] = round(float(dataset["booked"].mean()), 4)

    out = pd.concat([dataset[META_COLUMNS], features], axis=1)

    processed_dir().mkdir(parents=True, exist_ok=True)
    listings_path = processed_dir() / "listings_clean.parquet"
    dataset_path = processed_dir() / "demand_dataset.parquet"
    report_path = processed_dir() / "preprocess_report.json"

    primary_listings.drop(columns=["price_gbp_t0"]).to_parquet(listings_path, index=False)
    out.to_parquet(dataset_path, index=False)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    log(f"wrote {listings_path.name} ({len(primary_listings):,} primary pair listings)")
    log(f"wrote {dataset_path.name} ({len(out):,} rows)")
    log(f"wrote {report_path.name}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

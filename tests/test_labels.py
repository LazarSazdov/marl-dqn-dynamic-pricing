"""Two-snapshot diff label construction."""

import pandas as pd

from airbnb_marl.data.labels import attach_recent_occupancy, build_labels


def _cal(rows):
    df = pd.DataFrame(rows, columns=["listing_id", "date", "available"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_diff_labels():
    # t0: listing 1 open on d1..d3; closed on d4.
    cal_t0 = _cal([
        (1, "2026-07-01", True),
        (1, "2026-07-02", True),
        (1, "2026-07-03", True),
        (1, "2026-07-04", False),
    ])
    # t1: d1 got taken, d2 still open, d3 taken, d4 still closed.
    cal_t1 = _cal([
        (1, "2026-07-01", False),
        (1, "2026-07-02", True),
        (1, "2026-07-03", False),
        (1, "2026-07-04", False),
    ])
    labels, stats = build_labels(cal_t0, cal_t1, max_lead_days=180)

    assert len(labels) == 3  # d4 excluded: closed at t0 carries no signal
    by_date = labels.set_index(labels["date"].dt.strftime("%Y-%m-%d"))["booked"]
    assert by_date["2026-07-01"] == 1
    assert by_date["2026-07-02"] == 0
    assert by_date["2026-07-03"] == 1
    assert stats["nights_excluded_closed_at_t0"] == 1
    assert stats["booked_nights"] == 2


def test_lead_window_cut():
    cal_t0 = _cal([(1, "2026-07-01", True), (1, "2026-07-20", True)])
    cal_t1 = _cal([(1, "2026-07-01", False), (1, "2026-07-20", False)])
    labels, stats = build_labels(cal_t0, cal_t1, max_lead_days=10)
    # window starts at t1's first date (07-01); 07-20 is beyond 10 lead days
    assert len(labels) == 1
    assert stats["nights_within_lead_window"] == 1


def test_dates_only_in_one_calendar_are_dropped():
    cal_t0 = _cal([(1, "2026-06-25", True), (1, "2026-07-01", True)])
    cal_t1 = _cal([(1, "2026-07-01", True), (1, "2026-08-01", True)])
    labels, _ = build_labels(cal_t0, cal_t1, max_lead_days=180)
    assert list(labels["date"].dt.strftime("%Y-%m-%d")) == ["2026-07-01"]


def test_paused_listing_detection():
    from airbnb_marl.data.labels import detect_paused_listings

    days = pd.date_range("2026-07-01", periods=40, freq="D")
    labels = pd.DataFrame({
        "listing_id": [1] * 40 + [2] * 40,
        "date": list(days) * 2,
        "booked": [1] * 40 + [1, 0] * 20,  # listing 1 fully flipped, 2 half
    })
    labels["date"] = pd.to_datetime(labels["date"])
    paused, stats = detect_paused_listings(labels, min_open_nights=30)
    assert list(paused) == [1]
    assert stats["paused_nights_removed"] == 40


def test_booked_run_lengths():
    from airbnb_marl.data.labels import booked_run_lengths

    days = pd.date_range("2026-07-01", periods=10, freq="D")
    labels = pd.DataFrame({
        "listing_id": [1] * 10,
        "date": days,
        # two runs: nights 0-2 (len 3) and nights 5-6 (len 2)
        "booked": [1, 1, 1, 0, 0, 1, 1, 0, 0, 0],
    })
    stats = booked_run_lengths(labels)
    assert stats["runs"] == 2
    assert stats["median_run_nights"] == 2.5
    assert stats["share_nights_in_runs_le_14"] == 1.0


def test_recent_occupancy_rolling():
    days = pd.date_range("2026-07-01", periods=10, freq="D")
    cal_t0 = _cal([(1, d, i >= 5) for i, d in enumerate(days)])  # first 5 closed
    labels = pd.DataFrame({
        "listing_id": [1, 1],
        "date": [pd.Timestamp("2026-07-06"), pd.Timestamp("2026-07-10")],
        "booked": [0, 0],
    })
    out = attach_recent_occupancy(labels, cal_t0, window=7)
    # 2026-07-06: previous 5 nights all unavailable -> occupancy 1.0
    assert out.loc[out["date"] == "2026-07-06", "occupancy_recent"].iloc[0] == 1.0
    # 2026-07-10: previous 7 nights = 07-03..07-09, of which 07-03/04/05 closed -> 3/7
    assert abs(out.loc[out["date"] == "2026-07-10", "occupancy_recent"].iloc[0] - 3 / 7) < 1e-9

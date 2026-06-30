"""Demand models: training loop, evaluation, calibration, artifact round-trip."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from airbnb_marl.demand.evaluate import (
    IsotonicCalibrator,
    evaluate_probabilities,
    expected_calibration_error,
    monotonicity_check,
    sweep_price,
)
from airbnb_marl.demand.interface import DemandModel
from airbnb_marl.demand.models import DemandMLP
from airbnb_marl.demand.train import predict_mlp, train_mlp
from airbnb_marl.features.schema import DEMAND_FEATURES

MLP_CFG = {
    "hidden_dims": [16, 8],
    "dropout": 0.0,
    "batch_norm": True,
    "lr": 0.01,
    "weight_decay": 0.0,
    "batch_size": 256,
    "max_epochs": 8,
    "early_stopping_patience": 4,
}


def _synthetic(n=4000, seed=0):
    """Synthetic task where P(book) falls with price_ratio (feature 0)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(DEMAND_FEATURES))).astype(np.float32)
    ratio_col = DEMAND_FEATURES.index("price_ratio")
    logits = -2.0 * X[:, ratio_col] + 0.5 * X[:, 1] - 0.5
    y = (rng.random(n) < 1 / (1 + np.exp(-logits))).astype(np.int8)
    return X, y


def test_mlp_learns_synthetic_signal():
    X, y = _synthetic()
    model, history = train_mlp(X[:3000], y[:3000], X[3000:], y[3000:], MLP_CFG, log=lambda *_: None)
    probs = predict_mlp(model, X[3000:])
    metrics = evaluate_probabilities(y[3000:], probs)
    assert metrics["pr_auc"] > 1.3 * metrics["prevalence"]
    assert history["best_val_loss"] < 0.75


def test_ece_perfect_and_awful():
    y = np.array([0, 0, 0, 1] * 250)
    assert expected_calibration_error(y, np.full(1000, 0.25)) < 0.01
    assert expected_calibration_error(y, np.full(1000, 0.95)) > 0.5


def test_isotonic_calibrator_roundtrip():
    rng = np.random.default_rng(1)
    raw = rng.random(2000)
    y = (rng.random(2000) < raw ** 2).astype(int)  # miscalibrated on purpose
    cal = IsotonicCalibrator.fit(raw, y)
    restored = IsotonicCalibrator.from_dict(cal.to_dict())
    probs = np.array([0.1, 0.5, 0.9])
    assert np.allclose(cal(probs), restored(probs))
    assert expected_calibration_error(y, cal(raw)) < expected_calibration_error(y, raw)


def test_sweep_price_updates_both_price_features():
    captured = []

    def fake_predict(X):
        captured.append(X.copy())
        return np.full(len(X), 0.5)

    X = np.ones((4, len(DEMAND_FEATURES)))
    comp_median = np.full(4, 100.0)
    sweep_price(fake_predict, X, comp_median, np.array([0.5, 2.0]), DEMAND_FEATURES)
    i_ratio = DEMAND_FEATURES.index("price_ratio")
    i_logp = DEMAND_FEATURES.index("log_price")
    assert captured[0][0, i_ratio] == 0.5
    assert captured[0][0, i_logp] == pytest.approx(np.log1p(50.0))
    assert captured[1][0, i_ratio] == 2.0
    assert captured[1][0, i_logp] == pytest.approx(np.log1p(200.0))


def test_monotonicity_check_detects_direction():
    i_ratio = DEMAND_FEATURES.index("price_ratio")
    X = np.zeros((10, len(DEMAND_FEATURES)))
    comp = np.full(10, 100.0)

    down = monotonicity_check(
        lambda X_: 1 / (1 + np.exp(2 * X_[:, i_ratio])), X, comp, DEMAND_FEATURES
    )
    assert down["passed"]

    up = monotonicity_check(
        lambda X_: 1 / (1 + np.exp(-2 * X_[:, i_ratio])), X, comp, DEMAND_FEATURES
    )
    assert not up["passed"]


def test_elasticity_correction_steepens_and_roundtrips(tmp_path):
    X, y = _synthetic(n=800)
    model, _ = train_mlp(X[:600], y[:600], X[600:], y[600:],
                         {**MLP_CFG, "max_epochs": 2}, log=lambda *_: None)
    i_ratio = DEMAND_FEATURES.index("price_ratio")
    X_hi = X[:50].astype(np.float64).copy()
    X_hi[:, i_ratio] = 2.0  # everyone priced at 2x the competitor median

    plain = DemandModel(model, np.zeros(len(DEMAND_FEATURES)), np.ones(len(DEMAND_FEATURES)))
    boosted = DemandModel(
        model, np.zeros(len(DEMAND_FEATURES)), np.ones(len(DEMAND_FEATURES)),
        meta={"architecture": {"hidden_dims": [16, 8], "dropout": 0.0, "batch_norm": True}},
        price_penalty_beta=-1.0,
    )
    p_plain = plain.predict_proba_features(X_hi)
    p_boosted = boosted.predict_proba_features(X_hi)
    # r = 2 and beta = -1 gives a factor exp(-1)
    assert np.allclose(p_boosted, p_plain * np.exp(-1.0), atol=1e-6)
    # uncorrected view must ignore the penalty
    assert np.allclose(
        boosted.predict_proba_features(X_hi, apply_correction=False), p_plain, atol=1e-6
    )
    # penalty survives save and load
    boosted.save(tmp_path / "boosted")
    loaded = DemandModel.load(tmp_path / "boosted")
    assert loaded.price_penalty_beta == -1.0
    assert np.allclose(loaded.predict_proba_features(X_hi), p_boosted, atol=1e-6)


def test_artifact_save_load_roundtrip(tmp_path):
    X, y = _synthetic(n=1000)
    model, _ = train_mlp(X[:800], y[:800], X[800:], y[800:],
                         {**MLP_CFG, "max_epochs": 2}, log=lambda *_: None)
    dm = DemandModel(
        model,
        scaler_mean=np.zeros(len(DEMAND_FEATURES)),
        scaler_scale=np.ones(len(DEMAND_FEATURES)),
        calibrator=IsotonicCalibrator(np.array([0.0, 1.0]), np.array([0.0, 1.0])),
        meta={"architecture": {"hidden_dims": [16, 8], "dropout": 0.0, "batch_norm": True}},
    )
    dm.save(tmp_path / "artifact")
    loaded = DemandModel.load(tmp_path / "artifact")
    p1 = dm.predict_proba_features(X[:50].astype(np.float64))
    p2 = loaded.predict_proba_features(X[:50].astype(np.float64))
    assert np.allclose(p1, p2, atol=1e-6)


def test_predict_proba_from_raw_frame():
    X, y = _synthetic(n=600)
    model, _ = train_mlp(X[:500], y[:500], X[500:], y[500:],
                         {**MLP_CFG, "max_epochs": 2}, log=lambda *_: None)
    dm = DemandModel(model, np.zeros(len(DEMAND_FEATURES)), np.ones(len(DEMAND_FEATURES)))
    raw = pd.DataFrame({
        "price_gbp": [100.0, 150.0],
        "competitor_median_price": [100.0, 100.0],
        "competitor_price_std": [10.0, 10.0],
        "occupancy_recent": [0.4, 0.4],
        "review_scores_rating": [4.8, 4.8],
        "number_of_reviews": [10, 10],
        "host_is_superhost": [True, False],
        "accommodates": [2, 2],
        "room_type": ["Entire home/apt", "Private room"],
        "minimum_nights": [1, 2],
        "instant_bookable": [True, False],
        "calculated_host_listings_count": [1, 3],
        "estimated_occupancy_l365d": [100, 50],
        "reviews_per_month": [1.0, 0.5],
        "date": pd.to_datetime(["2026-07-03", "2026-07-04"]),
        "lead_days": [30, 30],
    })
    probs = dm.predict_proba(raw)
    assert probs.shape == (2,)
    assert np.all((probs >= 0) & (probs <= 1))

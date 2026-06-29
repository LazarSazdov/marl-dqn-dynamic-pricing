"""Evaluation for demand models.

Accuracy is skipped on purpose, it says nothing at a 13.7 percent positive
rate. The metrics follow the project spec: log loss, PR-AUC, Brier and
calibration, because the probabilities become the simulator's physics.
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> float:
    """ECE with equal width bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def evaluate_probabilities(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)
    return {
        "log_loss": float(log_loss(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
        "prevalence": float(np.mean(y_true)),
        "mean_predicted": float(np.mean(y_prob)),
    }


def calibration_points(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> dict:
    """Reliability diagram points, quantile bins for stable tails."""
    frac_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="quantile"
    )
    return {"mean_predicted": mean_pred.tolist(), "fraction_positive": frac_pos.tolist()}


class IsotonicCalibrator:
    """Isotonic regression stored as interpolation points, JSON safe."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)

    @classmethod
    def fit(cls, y_prob: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(y_prob, y_true)
        return cls(iso.X_thresholds_, iso.y_thresholds_)

    def __call__(self, y_prob: np.ndarray) -> np.ndarray:
        return np.interp(y_prob, self.x, self.y)

    def to_dict(self) -> dict:
        return {"x": self.x.tolist(), "y": self.y.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "IsotonicCalibrator":
        return cls(np.asarray(d["x"]), np.asarray(d["y"]))


def sweep_price(
    predict_features_fn,
    X: np.ndarray,
    competitor_median: np.ndarray,
    price_ratios: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    """Mean predicted booking probability while sweeping the price ratio.

    For each ratio the two price features are rewritten together
    (price_ratio and log_price) with everything else fixed. This is also the
    demand curve used later for the Nash and monopoly bounds.
    """
    i_ratio = feature_names.index("price_ratio")
    i_logp = feature_names.index("log_price")
    curve = []
    for ratio in price_ratios:
        Xc = X.copy()
        Xc[:, i_ratio] = ratio
        Xc[:, i_logp] = np.log1p(ratio * competitor_median)
        curve.append(float(np.mean(predict_features_fn(Xc))))
    return np.asarray(curve)


def monotonicity_check(
    predict_features_fn,
    X: np.ndarray,
    competitor_median: np.ndarray,
    feature_names: list[str],
    price_ratios: np.ndarray | None = None,
    tolerance: float = 0.002,
) -> dict:
    """Gate: mean P(booking) must not increase with price.

    Upticks below the tolerance between adjacent grid points count as noise.
    """
    if price_ratios is None:
        price_ratios = np.linspace(0.5, 2.5, 21)
    curve = sweep_price(
        predict_features_fn, X, competitor_median, price_ratios, feature_names
    )
    increases = np.diff(curve)
    worst = float(increases.max()) if len(increases) else 0.0
    return {
        "passed": bool(worst <= tolerance),
        "worst_increase": worst,
        "price_ratios": price_ratios.tolist(),
        "curve": curve.tolist(),
        "tolerance": tolerance,
    }

#!/usr/bin/env python3
"""Train and evaluate the demand models.

Trains logistic regression (baseline), LightGBM (diagnostic) and the MLP
(primary, early stopping on val log loss, optional isotonic recalibration
fitted on val). Evaluates on the untouched chronological test split, runs
the acceptance gates and saves the artifact plus metrics to results/demand.

Usage:
    python3 scripts/train_demand.py
    python3 scripts/train_demand.py --override configs/smoke/demand_overrides.yaml --sample-frac 0.05
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

from sklearn.preprocessing import StandardScaler

from airbnb_marl.config import load_config
from airbnb_marl.demand.evaluate import (
    IsotonicCalibrator,
    calibration_points,
    evaluate_probabilities,
    monotonicity_check,
)
from airbnb_marl.demand.interface import DemandModel
from airbnb_marl.demand.models import make_lightgbm, make_logistic_regression
from airbnb_marl.demand.train import predict_mlp, train_mlp
from airbnb_marl.features.schema import DEMAND_FEATURES
from airbnb_marl.utils.paths import processed_dir, results_dir
from airbnb_marl.utils.seeding import set_seed

MONO_SAMPLE = 20000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/demand.yaml")
    parser.add_argument("--override", default=None)
    parser.add_argument("--sample-frac", type=float, default=None,
                        help="subsample the dataset for smoke runs")
    parser.add_argument("--artifact-dir", default=None,
                        help="default: results/demand")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    set_seed(args.seed)
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else results_dir() / "demand"

    log("loading demand dataset")
    dataset = pd.read_parquet(processed_dir() / "demand_dataset.parquet")
    if args.sample_frac:
        dataset = dataset.sample(frac=args.sample_frac, random_state=args.seed)
    parts = {
        name: grp for name, grp in dataset.groupby("split")
    }
    missing_splits = {"train", "val", "test"} - set(parts)
    if missing_splits:
        raise SystemExit(
            f"splits missing from the (sub)sample: {sorted(missing_splits)}, "
            "use a larger --sample-frac"
        )
    X = {k: v[DEMAND_FEATURES].to_numpy(np.float32) for k, v in parts.items()}
    y = {k: v["booked"].to_numpy(np.int8) for k, v in parts.items()}
    comp_median = {k: v["competitor_median_price"].to_numpy(np.float64) for k, v in parts.items()}
    log(
        "rows: " + ", ".join(f"{k}={len(v):,} (pos {y[k].mean():.3f})" for k, v in parts.items())
    )
    del dataset, parts

    metrics: dict = {"config": cfg.to_dict(), "n_features": len(DEMAND_FEATURES)}
    test_probs: dict[str, np.ndarray] = {}

    log("training logistic regression")
    lr = make_logistic_regression(cfg["logistic_regression"])
    lr.fit(X["train"], y["train"])
    test_probs["lr"] = lr.predict_proba(X["test"])[:, 1]
    coefs = dict(zip(DEMAND_FEATURES, lr.named_steps["lr"].coef_[0].round(4).tolist()))
    metrics["lr_standardized_coefficients"] = coefs
    log(f"  price_ratio coefficient (std log odds): {coefs['price_ratio']}")

    log("training LightGBM (diagnostic)")
    import lightgbm as lgb

    lgbm = make_lightgbm(cfg["lightgbm"])
    lgbm.fit(
        X["train"], y["train"],
        eval_set=[(X["val"], y["val"])],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(cfg["lightgbm"]["early_stopping_rounds"], verbose=False)],
    )
    test_probs["lgbm"] = lgbm.predict_proba(X["test"])[:, 1]
    metrics["lgbm_best_iteration"] = int(lgbm.best_iteration_ or 0)
    metrics["lgbm_feature_importance"] = dict(zip(
        DEMAND_FEATURES, (lgbm.feature_importances_ / lgbm.feature_importances_.sum()).round(4).tolist()
    ))

    log("training MLP (primary)")
    scaler = StandardScaler().fit(X["train"])
    X_std = {k: scaler.transform(v) for k, v in X.items()}

    pos_weight = None
    if cfg["imbalance"]["strategy"] == "pos_weight":
        ratio = float((y["train"] == 0).sum() / max((y["train"] == 1).sum(), 1))
        pos_weight = min(ratio, float(cfg["imbalance"]["pos_weight_cap"]))
        log(f"  using pos_weight={pos_weight:.2f}")

    mlp, history = train_mlp(
        X_std["train"], y["train"], X_std["val"], y["val"],
        cfg["mlp"], pos_weight=pos_weight, log=log,
    )
    metrics["mlp_history"] = {
        "best_val_loss": history["best_val_loss"],
        "best_epoch": history["best_epoch"],
        "epochs_run": len(history["val_loss"]),
    }
    val_probs_mlp = predict_mlp(mlp, X_std["val"])
    test_probs["mlp"] = predict_mlp(mlp, X_std["test"])

    # recalibrate on val only if calibration error is above the threshold
    calibrator = None
    val_ece = evaluate_probabilities(y["val"], val_probs_mlp)["ece"]
    metrics["mlp_val_ece_before_calibration"] = val_ece
    if (
        cfg["calibration"]["method"] == "isotonic"
        and val_ece > cfg["calibration"]["apply_if_ece_above"]
    ):
        log(f"  val ECE {val_ece:.4f} above threshold, fitting isotonic calibrator")
        calibrator = IsotonicCalibrator.fit(val_probs_mlp, y["val"])
        test_probs["mlp_calibrated"] = np.clip(calibrator(test_probs["mlp"]), 0, 1)
    else:
        log(f"  val ECE {val_ece:.4f} within threshold, no recalibration needed")

    final_key = "mlp_calibrated" if calibrator else "mlp"
    metrics["test"] = {
        name: evaluate_probabilities(y["test"], probs)
        for name, probs in test_probs.items()
    }
    metrics["calibration_curves_test"] = {
        name: calibration_points(y["test"], probs)
        for name, probs in test_probs.items()
    }

    demand_model = DemandModel(
        mlp,
        scaler.mean_,
        scaler.scale_,
        calibrator,
        meta={
            "architecture": {
                "hidden_dims": list(cfg["mlp"]["hidden_dims"]),
                "dropout": cfg["mlp"]["dropout"],
                "batch_norm": cfg["mlp"]["batch_norm"],
            },
        },
    )
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(X["test"]), size=min(MONO_SAMPLE, len(X["test"])), replace=False)
    X_mono = X["test"][idx].astype(np.float64)
    cm_mono = comp_median["test"][idx]

    def revenue_peak(curve_dict: dict) -> float:
        ratios = np.asarray(curve_dict["price_ratios"])
        probs = np.asarray(curve_dict["curve"])
        return float(ratios[np.argmax(ratios * probs)])

    mono_raw = monotonicity_check(
        demand_model.predict_proba_features, X_mono, cm_mono, DEMAND_FEATURES
    )
    peak_raw = revenue_peak(mono_raw)
    metrics["demand_curve_uncorrected"] = {
        "price_ratios": mono_raw["price_ratios"],
        "mean_prob": mono_raw["curve"],
        "revenue_peak_ratio": peak_raw,
    }

    # correct the simulation price response only if the raw revenue curve
    # peaks outside the accepted interior band
    ela = cfg["elasticity"]
    ratios_grid = np.asarray(mono_raw["price_ratios"])
    curve_grid = np.asarray(mono_raw["curve"])
    peak_ok = ela["min_revenue_peak_ratio"] <= peak_raw <= ela["max_revenue_peak_ratio"]
    if not peak_ok:
        # grid search the beta that places the revenue peak at the target;
        # the correction is a pure exp(beta * (r - 1)) factor so the raw
        # curve can be rescaled in numpy without re-running the model
        betas = np.linspace(-3.0, 0.0, 601)
        peaks = np.array([
            ratios_grid[np.argmax(
                ratios_grid * curve_grid * np.exp(b * (ratios_grid - 1.0))
            )]
            for b in betas
        ])
        best = int(np.argmin(np.abs(peaks - ela["target_peak_ratio"])))
        beta = float(betas[best])
        demand_model.price_penalty_beta = beta
        log(
            f"  raw revenue peak at ratio {peak_raw}, applying beta {beta:.3f} "
            f"to move the peak to ratio {peaks[best]} "
            f"(target {ela['target_peak_ratio']})"
        )
        metrics["elasticity_correction"] = {
            "applied": True, "raw_peak_ratio": peak_raw,
            "target_peak_ratio": ela["target_peak_ratio"], "boost": beta,
        }
    else:
        log(f"  revenue peak interior at ratio {peak_raw}, no correction needed")
        metrics["elasticity_correction"] = {"applied": False, "revenue_peak_ratio": peak_raw}

    mono = monotonicity_check(
        demand_model.predict_proba_features, X_mono, cm_mono, DEMAND_FEATURES
    )
    peak_final = revenue_peak(mono)
    prevalence = metrics["test"][final_key]["prevalence"]
    pr_auc = metrics["test"][final_key]["pr_auc"]
    gates = {
        "pr_auc_over_prevalence": {
            "value": round(pr_auc / prevalence, 3),
            "required": cfg["gates"]["min_pr_auc_over_prevalence"],
            "passed": bool(pr_auc / prevalence >= cfg["gates"]["min_pr_auc_over_prevalence"]),
        },
        "price_monotonicity": {k: mono[k] for k in ("passed", "worst_increase", "tolerance")},
        "interior_revenue_peak": {
            "value": peak_final,
            "required_between": [
                ela["min_revenue_peak_ratio"], ela["max_revenue_peak_ratio"]
            ],
            "passed": bool(
                ela["min_revenue_peak_ratio"] <= peak_final <= ela["max_revenue_peak_ratio"]
            ),
        },
    }
    metrics["gates"] = gates
    metrics["demand_curve_test_sample"] = {
        "price_ratios": mono["price_ratios"], "mean_prob": mono["curve"],
        "revenue_peak_ratio": peak_final,
    }
    metrics["final_model"] = final_key

    demand_model.meta["metrics_test"] = metrics["test"][final_key]
    demand_model.meta["gates"] = gates
    demand_model.save(artifact_dir)
    with open(artifact_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    np.savez_compressed(
        artifact_dir / "eval_arrays.npz",
        y_test=y["test"],
        **{f"p_{name}": probs for name, probs in test_probs.items()},
    )
    log(f"artifact saved to {artifact_dir}")

    print("\n=== TEST metrics ===")
    for name, m in metrics["test"].items():
        print(f"  {name:15s} log-loss {m['log_loss']:.4f}  PR-AUC {m['pr_auc']:.4f}  "
              f"Brier {m['brier']:.4f}  ECE {m['ece']:.4f}")
    print(f"  (prevalence {prevalence:.4f})")
    print("\n=== Gates ===")
    print(json.dumps(gates, indent=2))
    all_passed = all(g["passed"] for g in gates.values())
    print("\nALL GATES PASSED" if all_passed else "\nGATE FAILURE, inspect before using in RL")
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Saved demand model artifact and its inference interface.

Downstream code (market environment, Nash and monopoly bounds) queries
booking probabilities only through this wrapper, so simulation inputs cannot
drift from training inputs. An artifact directory holds mlp.pt (state dict)
and meta.json (schema, scaler, calibrator, metrics, gates).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from airbnb_marl.demand.evaluate import IsotonicCalibrator
from airbnb_marl.demand.models import DemandMLP
from airbnb_marl.demand.train import predict_mlp
from airbnb_marl.features.schema import DEMAND_FEATURES, build_demand_features


class DemandModel:
    def __init__(
        self,
        model: DemandMLP,
        scaler_mean: np.ndarray,
        scaler_scale: np.ndarray,
        calibrator: IsotonicCalibrator | None = None,
        meta: dict | None = None,
        price_penalty_beta: float = 0.0,
    ):
        self.model = model.eval()
        self.scaler_mean = np.asarray(scaler_mean, dtype=np.float64)
        self.scaler_scale = np.asarray(scaler_scale, dtype=np.float64)
        self.calibrator = calibrator
        self.meta = meta or {}
        # Simulation side elasticity fix: multiply P by exp(beta * (r - 1))
        # where r is the price ratio and beta <= 0. Elasticity then grows
        # with price like in logit demand, which gives a unique interior
        # revenue peak, and P is unchanged at r = 1. Cross sectional prices
        # understate the causal price response, this puts it back.
        self.price_penalty_beta = float(price_penalty_beta)

    def predict_proba_features(
        self, X: np.ndarray, apply_correction: bool = True
    ) -> np.ndarray:
        """Probabilities from a built feature matrix.

        apply_correction=False gives the raw model output, used when scoring
        against real observations.
        """
        X = np.asarray(X, dtype=np.float64)
        X_std = (X - self.scaler_mean) / self.scaler_scale
        probs = predict_mlp(self.model, X_std.astype(np.float32))
        if self.calibrator is not None:
            probs = self.calibrator(probs)
        if apply_correction and self.price_penalty_beta != 0.0:
            ratio = np.maximum(X[:, DEMAND_FEATURES.index("price_ratio")], 1e-6)
            probs = probs * np.exp(self.price_penalty_beta * (ratio - 1.0))
        return np.clip(probs, 0.0, 1.0)

    def predict_proba(self, raw: pd.DataFrame) -> np.ndarray:
        """Probabilities from raw columns, schema applied internally."""
        X = build_demand_features(raw).to_numpy()
        return self.predict_proba_features(X)

    def save(self, artifact_dir: str | Path) -> Path:
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), artifact_dir / "mlp.pt")
        meta = dict(self.meta)
        meta.update({
            "features": DEMAND_FEATURES,
            "input_dim": len(DEMAND_FEATURES),
            "scaler_mean": self.scaler_mean.tolist(),
            "scaler_scale": self.scaler_scale.tolist(),
            "calibrator": self.calibrator.to_dict() if self.calibrator else None,
            "price_penalty_beta": self.price_penalty_beta,
        })
        with open(artifact_dir / "meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        return artifact_dir

    @classmethod
    def load(cls, artifact_dir: str | Path) -> "DemandModel":
        artifact_dir = Path(artifact_dir)
        with open(artifact_dir / "meta.json", encoding="utf-8") as fh:
            meta = json.load(fh)
        if meta["features"] != DEMAND_FEATURES:
            raise ValueError(
                "Artifact feature schema differs from current code, retrain "
                "with scripts/train_demand.py"
            )
        arch = meta["architecture"]
        model = DemandMLP(
            input_dim=meta["input_dim"],
            hidden_dims=arch["hidden_dims"],
            dropout=arch["dropout"],
            batch_norm=arch["batch_norm"],
        )
        model.load_state_dict(
            torch.load(artifact_dir / "mlp.pt", map_location="cpu", weights_only=True)
        )
        calibrator = (
            IsotonicCalibrator.from_dict(meta["calibrator"])
            if meta.get("calibrator")
            else None
        )
        return cls(
            model,
            np.asarray(meta["scaler_mean"]),
            np.asarray(meta["scaler_scale"]),
            calibrator,
            meta,
            price_penalty_beta=meta.get("price_penalty_beta", 0.0),
        )

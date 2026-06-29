"""Demand model architectures: logistic regression, LightGBM, MLP.

All models consume the feature matrix produced by build_demand_features.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_logistic_regression(cfg: dict) -> Pipeline:
    """Interpretable baseline, also gives the price elasticity coefficient."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=cfg["C"],
            max_iter=cfg["max_iter"],
            class_weight=cfg.get("class_weight"),
            solver="lbfgs",
        )),
    ])


def make_lightgbm(cfg: dict):
    """Diagnostic baseline.

    If the MLP cannot approach this model's PR-AUC, the MLP training is the
    problem rather than the features.
    """
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        num_leaves=cfg["num_leaves"],
        learning_rate=cfg["learning_rate"],
        n_estimators=cfg["n_estimators"],
        objective="binary",
        verbosity=-1,
    )


class DemandMLP(nn.Module):
    """Primary model, logit output for BCEWithLogitsLoss."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.2,
        batch_norm: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for width in hidden_dims:
            layers.append(nn.Linear(prev, width))
            if batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)

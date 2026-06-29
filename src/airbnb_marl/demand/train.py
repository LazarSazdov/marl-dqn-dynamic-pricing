"""Training routines for the demand models."""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn

from airbnb_marl.demand.models import DemandMLP


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: dict,
    pos_weight: float | None = None,
    device: str | None = None,
    log=print,
) -> tuple[DemandMLP, dict]:
    """Train the MLP with early stopping on validation log loss.

    Inputs must already be standardized. Returns the best model on CPU and
    a history dict.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = DemandMLP(
        input_dim=X_train.shape[1],
        hidden_dims=list(cfg["hidden_dims"]),
        dropout=cfg["dropout"],
        batch_norm=cfg["batch_norm"],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device) if pos_weight else None
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    # manual batching, a DataLoader is slow on millions of small rows
    X_train_t = torch.from_numpy(X_train.astype(np.float32)).to(device)
    y_train_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
    batch_size = int(cfg["batch_size"])
    X_val_t = torch.from_numpy(X_val.astype(np.float32)).to(device)
    y_val_t = torch.from_numpy(y_val.astype(np.float32)).to(device)
    val_criterion = nn.BCEWithLogitsLoss()

    best_state, best_val, best_epoch = None, float("inf"), -1
    history = {"train_loss": [], "val_loss": []}
    patience = int(cfg["early_stopping_patience"])

    for epoch in range(int(cfg["max_epochs"])):
        t_start = time.time()
        model.train()
        total, count = 0.0, 0
        perm = torch.randperm(len(X_train_t), device=device)
        for start in range(0, len(perm) - 1, batch_size):
            idx = perm[start : start + batch_size]
            if len(idx) < 2:  # BatchNorm needs more than one sample
                continue
            xb, yb = X_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
            count += len(xb)

        model.eval()
        with torch.no_grad():
            val_chunks = [
                float(val_criterion(model(xc), yc)) * len(xc)
                for xc, yc in zip(X_val_t.split(65536), y_val_t.split(65536))
            ]
            val_loss = sum(val_chunks) / len(X_val_t)
        scheduler.step(val_loss)

        history["train_loss"].append(total / max(count, 1))
        history["val_loss"].append(val_loss)
        log(
            f"  epoch {epoch + 1:>3}/{cfg['max_epochs']} "
            f"train {history['train_loss'][-1]:.4f} val {val_loss:.4f} "
            f"({time.time() - t_start:.0f}s)"
        )

        if val_loss < best_val - 1e-5:
            best_val, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            log(f"  early stop at epoch {epoch + 1} (best epoch {best_epoch + 1})")
            break

    model.load_state_dict(best_state)
    model.to("cpu").eval()
    history["best_val_loss"] = best_val
    history["best_epoch"] = best_epoch + 1
    return model, history


@torch.no_grad()
def predict_mlp(model: DemandMLP, X: np.ndarray, batch: int = 65536) -> np.ndarray:
    """Predicted probabilities on CPU, batched."""
    model.eval()
    outputs = []
    for chunk in torch.from_numpy(X.astype(np.float32)).split(batch):
        outputs.append(torch.sigmoid(model(chunk)).numpy())
    return np.concatenate(outputs)

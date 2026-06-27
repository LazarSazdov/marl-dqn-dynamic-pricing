"""Path resolution that works locally and on Colab.

Everything is derived from the repo root. AIRBNB_MARL_DATA_DIR can redirect
the data directory, which is useful on Colab where the Drive mount is slow
and data is copied to the local disk first.
"""

from __future__ import annotations

import os
from pathlib import Path

_ROOT_MARKERS = ("pyproject.toml", ".git")


def repo_root(start: Path | None = None) -> Path:
    """Walk upwards until a repo marker is found."""
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in _ROOT_MARKERS):
            return candidate
    raise FileNotFoundError(
        f"Could not locate repository root from {current} "
        f"(looked for {', '.join(_ROOT_MARKERS)})"
    )


def data_dir() -> Path:
    override = os.environ.get("AIRBNB_MARL_DATA_DIR")
    return Path(override) if override else repo_root() / "data"


def raw_snapshot_dir(snapshot_date: str) -> Path:
    return data_dir() / "raw" / snapshot_date


def processed_dir() -> Path:
    return data_dir() / "processed"


def results_dir() -> Path:
    override = os.environ.get("AIRBNB_MARL_RESULTS_DIR")
    return Path(override) if override else repo_root() / "results"


def configs_dir() -> Path:
    return repo_root() / "configs"

"""YAML config loading with attribute access and override merging."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml

from airbnb_marl.utils.paths import repo_root


class Config(dict):
    """Dict with recursive attribute access."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value

    def to_dict(self) -> dict:
        return copy.deepcopy(dict(self))


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else repo_root() / path


def deep_update(base: dict, override: Mapping) -> dict:
    """Merge override into a deep copy of base, recursively."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_yaml(path: str | Path) -> dict:
    with open(_resolve(path), "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(
    path: str | Path, overrides: str | Path | Mapping | None = None
) -> Config:
    """Load a YAML config, optionally merging an override file or mapping."""
    cfg = load_yaml(path)
    if overrides is not None:
        if isinstance(overrides, (str, Path)):
            overrides = load_yaml(overrides)
        cfg = deep_update(cfg, overrides)
    return Config(cfg)

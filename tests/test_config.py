"""Config system and repo-layout sanity tests."""

from pathlib import Path

import pytest

from airbnb_marl.config import Config, deep_update, load_config
from airbnb_marl.utils.paths import configs_dir, repo_root


def test_repo_root_found():
    root = repo_root()
    assert (root / "pyproject.toml").exists()
    assert (root / "src" / "airbnb_marl").is_dir()


def test_deep_update_merges_nested():
    base = {"a": 1, "nested": {"x": 1, "y": 2}}
    result = deep_update(base, {"nested": {"y": 99}, "b": 2})
    assert result == {"a": 1, "b": 2, "nested": {"x": 1, "y": 99}}
    assert base["nested"]["y"] == 2  # base untouched


def test_config_attribute_access():
    cfg = Config({"dqn": {"gamma": 0.95}})
    assert cfg.dqn.gamma == 0.95
    with pytest.raises(AttributeError):
        _ = cfg.missing


def test_data_config_loads():
    cfg = load_config("configs/data.yaml")
    assert cfg.primary_snapshot == "2026-06-19"
    pairs = cfg["snapshot_pairs"]
    assert all(len(pair) == 2 for pair in pairs)
    for snapshot in dict.fromkeys(d for pair in pairs for d in pair):
        assert snapshot in cfg["files"], f"snapshot {snapshot} missing file list"


def test_env_config_loads():
    cfg = load_config("configs/env.yaml")
    assert len(cfg["action_deltas"]) == 5
    assert 0.0 in cfg["action_deltas"]
    assert cfg.price_min_ratio < 1.0 < cfg.price_max_ratio


def test_all_experiment_configs_load_with_smoke_overrides():
    experiment_dir = configs_dir() / "experiments"
    paths = sorted(experiment_dir.glob("*.yaml"))
    assert len(paths) == 5
    for path in paths:
        cfg = load_config(path, overrides="configs/smoke/rl_overrides.yaml")
        assert cfg.algorithm in {"dqn", "tql", "ppo"}
        assert cfg.n_agents in {2, 4}
        assert cfg.n_seeds == 2  # smoke override applied
        assert cfg.total_steps == 8000
        block = cfg[cfg.algorithm] if cfg.algorithm != "tql" else cfg["tql"]
        assert "gamma" in block

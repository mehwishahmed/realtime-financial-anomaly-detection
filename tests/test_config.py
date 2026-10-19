"""The config loads, resolves paths, and rejects bad values."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from anomaly.config import ConfigError, load_config

from conftest import REPO_ROOT


def _write_variant(tmp_path: Path, edit) -> Path:
    raw = yaml.safe_load((REPO_ROOT / "configs" / "default.yaml").read_text())
    edit(raw)
    path = REPO_ROOT / "configs" / f"_test_{tmp_path.name}.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_default_config_loads():
    cfg = load_config("configs/default.yaml")
    assert cfg.root == REPO_ROOT
    assert cfg.paths.data_dir == REPO_ROOT / "data"
    assert cfg.universe.market_ticker == "SPY"
    assert set(cfg.frequencies) == {"daily", "intraday"}
    assert cfg.frequencies["daily"].splits.train_end == "2022-12-31"
    assert cfg.frequencies["intraday"].splits.train_frac == 0.6


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda r: r["frequencies"]["daily"]["splits"].update(train_end="2025-01-01"), "train_end must be before"),
        (lambda r: r["frequencies"]["intraday"]["splits"].update(train_frac=0.9), "sum to less than 1"),
        (lambda r: r["frequencies"]["daily"].update(volume_z_mode="weekly"), "volume_z_mode"),
        (lambda r: r["universe"].update(max_tickers=0), "max_tickers"),
        (lambda r: r["cleaning"].pop("max_missing_frac"), "max_missing_frac"),
    ],
)
def test_bad_values_are_rejected(tmp_path, edit, message):
    path = _write_variant(tmp_path, edit)
    try:
        with pytest.raises(ConfigError, match=message):
            load_config(path)
    finally:
        path.unlink()


def test_extends_merges_on_top_of_the_base(tmp_path):
    path = REPO_ROOT / "configs" / f"_test_extends_{tmp_path.name}.yaml"
    path.write_text(
        "extends: default.yaml\n"
        "paths: {data_dir: data/x, results_dir: results/x, models_dir: models/x, raw_data_dir: data/raw}\n"
        "universe: {max_tickers: null}\n"
        "models: {deep: {max_train_windows: 5000}}\n"
    )
    try:
        cfg = load_config(path)
    finally:
        path.unlink()
    base = load_config("configs/default.yaml")
    assert cfg.universe.max_tickers is None and cfg.universe.market_ticker == base.universe.market_ticker
    assert cfg.paths.data_dir == REPO_ROOT / "data" / "x" and cfg.raw_dir("daily") == base.raw_dir("daily")
    assert cfg.models.deep.max_train_windows == 5000 and cfg.models.deep.window == base.models.deep.window
    assert cfg.models.isolation_forest == base.models.isolation_forest

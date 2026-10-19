"""Shared test fixtures: a config that points at temp folders and tiny synthetic price series."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from anomaly.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """The real default config, with data/results/models redirected into a temp folder."""
    raw = yaml.safe_load((REPO_ROOT / "configs" / "default.yaml").read_text())
    raw["paths"] = {k: str(tmp_path / k) for k in ("data_dir", "results_dir", "models_dir")}
    path = REPO_ROOT / "configs" / f"_test_{tmp_path.name}.yaml"
    path.write_text(yaml.safe_dump(raw))
    try:
        return load_config(path)
    finally:
        path.unlink()


def make_bars(
    index: pd.DatetimeIndex,
    seed: int = 0,
    start_price: float = 100.0,
    daily_vol: float = 0.01,
    volume: float = 1e6,
) -> pd.DataFrame:
    """A random-walk OHLCV frame in the same format yfinance data is cached in."""
    rng = np.random.default_rng(seed)
    close = start_price * np.exp(np.cumsum(rng.normal(0, daily_vol, len(index))))
    open_ = close * np.exp(rng.normal(0, daily_vol / 4, len(index)))
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    vol = volume * np.exp(rng.normal(0, 0.2, len(index)))
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol.round(),
            "dividends": 0.0,
            "stock_splits": 0.0,
        },
        index=pd.DatetimeIndex(index, name="timestamp"),
    )


@pytest.fixture
def business_days() -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=900)

"""Time-based train/validation/test splits. No shuffling, so no future data leaks into training."""

from __future__ import annotations

import pandas as pd

from anomaly.config import SplitConfig

SPLITS = ("train", "val", "test")


def _session_date(ts: pd.Series) -> pd.Series:
    """Calendar date of each bar in exchange time (tz-naive midnight)."""
    if ts.dt.tz is not None:
        ts = ts.dt.tz_localize(None)
    return ts.dt.normalize()


def split_boundaries(timestamps: pd.Series, cfg: SplitConfig) -> dict[str, pd.Timestamp]:
    """Last date (inclusive) of the train and val sets.

    With fractions, the cut points fall between trading days, so a single
    day never ends up in two sets.
    """
    if cfg.train_end is not None and cfg.val_end is not None:
        return {"train_end": pd.Timestamp(cfg.train_end), "val_end": pd.Timestamp(cfg.val_end)}
    assert cfg.train_frac is not None and cfg.val_frac is not None
    days = pd.Series(_session_date(timestamps).unique()).sort_values(ignore_index=True)
    n = len(days)
    n_train = int(n * cfg.train_frac)
    n_val = int(n * cfg.val_frac)
    if n_train < 1 or n_val < 1 or n_train + n_val >= n:
        raise ValueError(f"Not enough trading days ({n}) for train_frac={cfg.train_frac}, val_frac={cfg.val_frac}")
    return {"train_end": days[n_train - 1], "val_end": days[n_train + n_val - 1]}


def apply_boundaries(df: pd.DataFrame, bounds: dict[str, pd.Timestamp]) -> pd.DataFrame:
    """Add a ``split`` column using boundaries that were already computed."""
    day = _session_date(df["timestamp"])
    split = pd.Series("test", index=df.index)
    split[day <= bounds["val_end"]] = "val"
    split[day <= bounds["train_end"]] = "train"
    out = df.copy()
    out["split"] = pd.Categorical(split, categories=list(SPLITS))
    return out


def assign_splits(df: pd.DataFrame, cfg: SplitConfig) -> tuple[pd.DataFrame, dict[str, pd.Timestamp]]:
    """Add a ``split`` column (train / val / test) based only on each row's timestamp."""
    bounds = split_boundaries(df["timestamp"], cfg)
    return apply_boundaries(df, bounds), bounds

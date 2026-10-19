"""Per-ticker features computed only from current and past bars (no lookahead)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from anomaly.config import FeatureConfig, FrequencyConfig


def feature_columns(fcfg: FeatureConfig) -> list[str]:
    """Names of the model-input feature columns, in a fixed order."""
    return ["log_ret", *[f"vol_{w}" for w in fcfg.vol_windows], "volume_z", "gap", "corr_market"]


def _rolling(grouped: pd.core.groupby.SeriesGroupBy, window: int, stat: str, min_periods: int) -> pd.Series:
    return grouped.transform(lambda s: getattr(s.rolling(window, min_periods=min_periods), stat)())


def volume_zscore(df: pd.DataFrame, freq: FrequencyConfig, fcfg: FeatureConfig) -> pd.Series:
    """How unusual this bar's volume is, compared with *earlier* bars only.

    ``rolling``: compared with the previous ``volume_z_window`` bars.
    ``time_of_day``: compared with the same intraday slot on the previous
    ``volume_z_days`` days. Intraday volume is always high at the open and
    close, so a plain rolling window would flag every open as an anomaly.

    The baseline is shifted by one bar, so a volume spike can't hide itself
    by inflating its own mean and standard deviation.
    """
    log_vol = np.log1p(df["volume"])  # NaN for filled bars, which have no real volume
    if freq.volume_z_mode == "rolling":
        keys = [df["ticker"]]
        window = fcfg.volume_z_window
    else:
        keys = [df["ticker"], df["timestamp"].dt.time]
        window = fcfg.volume_z_days
    min_periods = max(2, math.ceil(0.8 * window))
    past = log_vol.groupby(keys, sort=False).shift(1)
    grouped = past.groupby(keys, sort=False)
    mean = _rolling(grouped, window, "mean", min_periods)
    std = _rolling(grouped, window, "std", min_periods)
    return (log_vol - mean) / std.where(std > 0)


def rolling_corr(df: pd.DataFrame, x: str, y: str, window: int) -> pd.Series:
    """Per-ticker rolling Pearson correlation between columns ``x`` and ``y``."""
    both = df[x].notna() & df[y].notna()
    xs, ys = df[x].where(both), df[y].where(both)
    frame = pd.DataFrame({"x": xs, "y": ys, "xy": xs * ys, "xx": xs * xs, "yy": ys * ys})
    means = {c: _rolling(frame[c].groupby(df["ticker"], sort=False), window, "mean", window) for c in frame}
    cov = means["xy"] - means["x"] * means["y"]
    var_x = means["xx"] - means["x"] ** 2
    var_y = means["yy"] - means["y"] ** 2
    denom = np.sqrt(var_x.clip(lower=0) * var_y.clip(lower=0))
    return (cov / denom.where(denom > 1e-12)).clip(-1, 1)


def build_features(
    clean: pd.DataFrame,
    market_ticker: str,
    freq: FrequencyConfig,
    fcfg: FeatureConfig,
) -> pd.DataFrame:
    """Add feature columns to cleaned bars.

    Returns one row per (ticker, timestamp) for every ticker except the
    market ticker. Rows where any feature can't be computed yet (warm-up at
    the start of the history, or filled bars with no real volume) are dropped.
    The cleaned OHLCV columns are kept so later phases can inject synthetic
    anomalies into prices and recompute features.
    """
    df = clean.sort_values(["ticker", "timestamp"], ignore_index=True).copy()
    log_close = np.log(df["close"])
    prev_log_close = log_close.groupby(df["ticker"], sort=False).shift(1)
    df["log_ret"] = log_close - prev_log_close
    df["gap"] = np.log(df["open"]) - prev_log_close
    for w in fcfg.vol_windows:
        df[f"vol_{w}"] = _rolling(df["log_ret"].groupby(df["ticker"], sort=False), w, "std", w)
    df["volume_z"] = volume_zscore(df, freq, fcfg)

    market = df.loc[df["ticker"] == market_ticker, ["timestamp", "log_ret"]].rename(columns={"log_ret": "mkt_ret"})
    df = df.merge(market, on="timestamp", how="left")
    df["corr_market"] = rolling_corr(df, "log_ret", "mkt_ret", fcfg.corr_window)

    cols = feature_columns(fcfg)
    df = df[df["ticker"] != market_ticker].drop(columns="mkt_ret")
    df = df.dropna(subset=cols)
    df[cols] = df[cols].astype("float32")
    return df.reset_index(drop=True)

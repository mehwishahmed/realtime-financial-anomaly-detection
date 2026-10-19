"""Clean raw bars: align to the trading calendar, fill short gaps, fix splits, drop bad tickers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from anomaly.config import CleaningConfig, FrequencyConfig

CLEAN_COLUMNS = ["ticker", "timestamp", "open", "high", "low", "close", "volume", "is_filled"]
OHLC = ["open", "high", "low", "close"]
EXTREME_LOG_MOVE = np.log(1.4)  # only counted in the report; big moves are what we want to detect


@dataclass
class CleanResult:
    """Cleaned bars plus a per-ticker report of what happened to each ticker."""

    data: pd.DataFrame  # long format: one row per (ticker, timestamp), CLEAN_COLUMNS
    report: pd.DataFrame  # one row per ticker: status, reason, and data-quality counts


def trading_calendar(market: pd.DataFrame) -> pd.DatetimeIndex:
    """The market ticker's bar timestamps define which bars exist (this handles holidays and half days)."""
    m = market[~market.index.duplicated(keep="last")].dropna(subset=["close"])
    return pd.DatetimeIndex(m.index.sort_values())


def repair_splits(df: pd.DataFrame, tolerance: float, min_ratio: float) -> tuple[pd.DataFrame, int]:
    """Undo any split that Yahoo didn't adjust for.

    yfinance's ``auto_adjust`` normally back-adjusts prices for splits. If a
    split day's return still matches the split ratio (for example -90% on a
    10:1 split), earlier bars get divided by the ratio and their volume
    multiplied by it, so the series is continuous again.

    Only ratios of at least ``min_ratio`` (or at most ``1/min_ratio``) are
    checked. Small ratios like 1.02 are stock dividends, which move the price
    about as much as an ordinary day does, so the check can't tell them apart.
    """
    df = df.copy()
    repairs = 0
    ratios = df["stock_splits"].fillna(0)
    big = (ratios > 0) & (np.abs(np.log(ratios.where(ratios > 0, 1.0))) >= np.log(min_ratio))
    split_days = df.index[big]
    for day in split_days:
        pos = df.index.get_loc(day)
        if pos == 0:
            continue
        ratio = float(df["stock_splits"].iloc[pos])
        ret = np.log(df["close"].iloc[pos] / df["close"].iloc[pos - 1])
        if abs(ret + np.log(ratio)) < tolerance:
            before = df.index[:pos]
            df.loc[before, OHLC] = df.loc[before, OHLC] / ratio
            df.loc[before, "volume"] = df.loc[before, "volume"] * ratio
            repairs += 1
    return df, repairs


def repair_ohlc(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Make high the largest and low the smallest of OHLC. Adjustment rounding sometimes breaks this."""
    df = df.copy()
    hi = df[OHLC].max(axis=1)
    lo = df[OHLC].min(axis=1)
    bad = (df["high"] < hi) | (df["low"] > lo)
    df["high"], df["low"] = hi, lo
    return df, int(bad.sum())


def clean_ticker(
    ticker: str,
    raw: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    freq: FrequencyConfig,
    cfg: CleaningConfig,
) -> tuple[pd.DataFrame | None, dict[str, object]]:
    """Clean one ticker's bars. Returns (clean bars or None if rejected, report row)."""
    info: dict[str, object] = {"ticker": ticker, "raw_bars": len(raw)}
    df = raw[~raw.index.duplicated(keep="last")].sort_index()
    df = df.astype({c: "float64" for c in [*OHLC, "volume", "stock_splits"]})
    info["duplicate_bars"] = len(raw) - len(df)

    bad_price = df["close"].isna() | (df[OHLC] <= 0).any(axis=1)
    info["nonpositive_or_nan_rows"] = int(bad_price.sum())
    df = df[~bad_price]
    df = df[df.index.isin(calendar)]  # drop bars outside the trading calendar (e.g. odd timestamps)
    if df.empty:
        return None, {**info, "status": "rejected", "reason": "no_valid_bars"}

    df, info["split_repairs"] = repair_splits(df, cfg.split_tolerance, cfg.min_split_ratio)
    df, info["ohlc_repairs"] = repair_ohlc(df)

    span = calendar[(calendar >= df.index[0]) & (calendar <= df.index[-1])]
    missing = len(span) - len(df)
    info["first"], info["last"] = df.index[0], df.index[-1]
    info["missing_frac"] = round(missing / len(span), 6) if len(span) else 1.0
    zero_volume = df["volume"] == 0
    info["zero_volume_frac"] = round(float(zero_volume.mean()), 6)
    # A traded price with zero volume is a data glitch for liquid stocks; treat the volume as unknown.
    info["zero_volume_bars"] = int(zero_volume.sum())
    df.loc[zero_volume, "volume"] = np.nan

    full = df.reindex(span)
    full["is_filled"] = full["close"].isna()
    full[OHLC] = full[OHLC].ffill(limit=freq.max_ffill_bars)
    long_gap = full["close"].isna()
    info["dropped_long_gap_bars"] = int(long_gap.sum())
    full = full[~long_gap]
    # Filled bars carry the last price but had no trading, so volume is unknown.
    full.loc[full["is_filled"], "volume"] = np.nan
    for col in ("open", "high", "low"):
        full[col] = full[col].where(~full["is_filled"], full["close"])
    info["filled_bars"] = int(full["is_filled"].sum())
    info["clean_bars"] = len(full)
    log_ret = np.log(full["close"]).diff()
    info["extreme_moves"] = int((log_ret.abs() > EXTREME_LOG_MOVE).sum())

    if info["missing_frac"] > cfg.max_missing_frac:
        return None, {**info, "status": "rejected", "reason": "too_many_missing"}
    if info["zero_volume_frac"] > cfg.max_zero_volume_frac:
        return None, {**info, "status": "rejected", "reason": "too_many_zero_volume"}
    if len(full) < freq.min_history_bars:
        return None, {**info, "status": "rejected", "reason": "insufficient_history"}

    out = full.rename_axis("timestamp").reset_index()
    out["ticker"] = ticker
    return out[CLEAN_COLUMNS], {**info, "status": "ok", "reason": ""}


def clean_universe(
    raw: dict[str, pd.DataFrame],
    market_ticker: str,
    freq: FrequencyConfig,
    cfg: CleaningConfig,
) -> CleanResult:
    """Clean every ticker against the market ticker's trading calendar.

    Raises ``ValueError`` if the market ticker is missing or fails cleaning,
    because the calendar and the correlation features depend on it.
    """
    if market_ticker not in raw:
        raise ValueError(f"Market ticker {market_ticker} has no data; it is required for the calendar")
    calendar = trading_calendar(raw[market_ticker])
    frames, rows = [], []
    for ticker in sorted(raw):
        clean, info = clean_ticker(ticker, raw[ticker], calendar, freq, cfg)
        rows.append(info)
        if clean is not None:
            frames.append(clean)
    report = pd.DataFrame(rows)
    if report.loc[report["ticker"] == market_ticker, "status"].iloc[0] != "ok":
        reason = report.loc[report["ticker"] == market_ticker, "reason"].iloc[0]
        raise ValueError(f"Market ticker {market_ticker} failed cleaning: {reason}")
    data = pd.concat(frames, ignore_index=True).sort_values(["ticker", "timestamp"], ignore_index=True)
    lead = ["ticker", "status", "reason"]
    return CleanResult(data=data, report=report[lead + [c for c in report.columns if c not in lead]])

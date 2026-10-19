"""Tag real events: earnings-day shocks and large single-day moves."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from anomaly.config import DownloadConfig, RealEventConfig
from anomaly.labels import EVENT_COLUMNS
from anomaly.utils import get_logger

EarningsFetch = Callable[[str, int], pd.DataFrame | None]


def _default_earnings_fetch(ticker: str, limit: int) -> pd.DataFrame | None:
    import yfinance as yf

    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    return yf.Ticker(ticker).get_earnings_dates(limit=limit)


def download_earnings(
    tickers: list[str],
    out_dir: Path,
    limit: int,
    dl: DownloadConfig,
    fetch: EarningsFetch | None = None,
    sleep: Callable[[float], None] = time.sleep,
    logger: logging.Logger | None = None,
) -> dict[str, list[str]]:
    """Fetch and cache earnings announcement times, one parquet file per ticker.

    Returns {"cached": [...], "downloaded": [...], "failed": [...]}. A ticker
    with no earnings history (Yahoo returns nothing) counts as failed.
    """
    fetch = fetch or _default_earnings_fetch
    logger = logger or get_logger()
    out_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, list[str]] = {"cached": [], "downloaded": [], "failed": []}
    for i, ticker in enumerate(tickers):
        path = out_dir / f"{ticker}.parquet"
        if path.exists() and not dl.refresh:
            status["cached"].append(ticker)
            continue
        frame = None
        for attempt in range(dl.max_retries + 1):
            if attempt:
                sleep(dl.backoff_seconds * 2 ** (attempt - 1))
            try:
                frame = fetch(ticker, limit)
                break
            except Exception as exc:  # rate limits, network errors
                logger.warning("  earnings %s attempt %d failed: %s", ticker, attempt + 1, exc)
        if frame is None or frame.empty:
            status["failed"].append(ticker)
            continue
        dates = pd.DataFrame({"announced": pd.DatetimeIndex(frame.index)})
        dates.to_parquet(path, index=False)
        status["downloaded"].append(ticker)
        if (i + 1) % 25 == 0:
            logger.info("  earnings: %d/%d tickers", i + 1, len(tickers))
    return status


def load_earnings(tickers: list[str], out_dir: Path) -> pd.DataFrame:
    """Cached announcement times as a long frame with columns (ticker, announced)."""
    frames = [
        pd.read_parquet(p).assign(ticker=t) for t in tickers if (p := out_dir / f"{t}.parquet").exists()
    ]
    if not frames:
        return pd.DataFrame(columns=["ticker", "announced"])
    return pd.concat(frames, ignore_index=True)[["ticker", "announced"]]


def reaction_days(
    earnings: pd.DataFrame, calendar: pd.DatetimeIndex, after_hour: int
) -> pd.DataFrame:
    """Map each announcement to the first daily bar where the market can react.

    Announced before ``after_hour`` (Eastern time) means the same trading day;
    at or after it (for example 16:00, after the close) means the next trading day.
    Announcements outside the calendar's range are dropped.
    """
    if earnings.empty:
        return pd.DataFrame(columns=["ticker", "announced", "reaction_day"])
    ann = pd.DatetimeIndex(earnings["announced"])
    if ann.tz is not None:
        ann = ann.tz_convert("America/New_York")
    day = ann.tz_localize(None).normalize() if ann.tz is not None else ann.normalize()
    after = np.asarray(ann.hour >= after_hour)
    cal = pd.DatetimeIndex(calendar).sort_values()
    # First trading day >= announcement day, or > it if announced after the cutoff.
    pos = np.where(after, cal.searchsorted(day, side="right"), cal.searchsorted(day, side="left"))
    valid = (pos < len(cal)) & (day >= cal[0])
    out = earnings.loc[valid].copy()
    out["reaction_day"] = cal[pos[valid]]
    return out.drop_duplicates(["ticker", "reaction_day"]).reset_index(drop=True)


def tag_real_events(
    features: pd.DataFrame, reactions: pd.DataFrame, cfg: RealEventConfig, vol_col: str = "vol_20"
) -> pd.DataFrame:
    """Find real anomalies in the (uninjected) daily feature table.

    - earnings_shock: on an earnings reaction day, |log return| is at least
      ``earnings_sigma`` x the previous bar's rolling volatility.
    - large_move: any other day where |log return| is at least
      ``large_move_sigma`` x that volatility and at least ``large_move_min_abs``.

    A day can only have one type; earnings takes priority because it explains the move.
    """
    df = features[["ticker", "timestamp", "log_ret", vol_col]].sort_values(["ticker", "timestamp"]).copy()
    df["prior_vol"] = df.groupby("ticker", sort=False)[vol_col].shift(1)
    size = df["log_ret"].abs() / df["prior_vol"]
    keys = reactions[["ticker", "reaction_day"]].rename(columns={"reaction_day": "timestamp"})
    df = df.merge(keys.assign(is_earnings=True), on=["ticker", "timestamp"], how="left")
    df["is_earnings"] = df["is_earnings"].fillna(False).astype(bool)
    size = size.to_numpy()

    shock = df["is_earnings"].to_numpy() & (size >= cfg.earnings_sigma)
    large = (
        ~df["is_earnings"].to_numpy()
        & (size >= cfg.large_move_sigma)
        & (df["log_ret"].abs().to_numpy() >= cfg.large_move_min_abs)
    )
    rows = []
    for mask, kind in ((shock, "earnings_shock"), (large, "large_move")):
        hit = df.loc[mask]
        rows.append(
            pd.DataFrame(
                {
                    "type": kind,
                    "ticker": hit["ticker"],
                    "start": hit["timestamp"],
                    "end": hit["timestamp"],
                    "magnitude": (hit["log_ret"] / hit["prior_vol"]).round(4),
                }
            )
        )
    events = pd.concat(rows, ignore_index=True).sort_values(["ticker", "start"], ignore_index=True)
    events["event_id"] = [f"real-{i:05d}" for i in range(len(events))]
    events["track"] = "real"
    return events[EVENT_COLUMNS]

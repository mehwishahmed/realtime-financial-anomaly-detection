"""Download price/volume bars from yfinance in batches, with retries and a parquet cache."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from anomaly.config import DownloadConfig, FrequencyConfig
from anomaly.utils import get_logger

PRICE_COLUMNS = ["open", "high", "low", "close", "volume", "dividends", "stock_splits"]

FetchFn = Callable[..., pd.DataFrame]


@dataclass
class DownloadReport:
    """What happened to each requested ticker."""

    requested: list[str]
    cached: list[str] = field(default_factory=list)  # already on disk, skipped
    downloaded: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    requests_made: int = 0
    retries: int = 0

    def summary(self) -> dict[str, int]:
        return {
            "requested": len(self.requested),
            "cached": len(self.cached),
            "downloaded": len(self.downloaded),
            "failed": len(self.failed),
            "requests_made": self.requests_made,
            "retries": self.retries,
        }


def cache_path(raw_dir: Path, ticker: str) -> Path:
    return raw_dir / f"{ticker}.parquet"


def _default_fetch(**kwargs: object) -> pd.DataFrame:
    import yfinance as yf

    logging.getLogger("yfinance").setLevel(logging.CRITICAL)  # we report failures ourselves
    return yf.download(**kwargs)


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename yfinance columns to snake_case, keep the price columns, and drop empty rows."""
    out = frame.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    for col in PRICE_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0 if col in ("dividends", "stock_splits") else float("nan")
    out = out[PRICE_COLUMNS]
    out = out.dropna(how="all", subset=["open", "high", "low", "close"])
    out.index.name = "timestamp"
    return out.sort_index()


def extract_ticker(batch: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """Pull one ticker's bars out of a multi-ticker yfinance result, or None if it has none."""
    if batch is None or batch.empty:
        return None
    if isinstance(batch.columns, pd.MultiIndex):
        if ticker not in batch.columns.get_level_values(0):
            return None
        frame = batch[ticker]
    else:
        frame = batch
    frame = normalize_frame(frame)
    return frame if len(frame) else None


def download_bars(
    tickers: list[str],
    freq: FrequencyConfig,
    dl: DownloadConfig,
    raw_dir: Path,
    fetch: FetchFn | None = None,
    sleep: Callable[[float], None] = time.sleep,
    logger: logging.Logger | None = None,
) -> DownloadReport:
    """Download bars for ``tickers`` and cache one parquet file per ticker.

    Tickers already cached are skipped unless ``dl.refresh`` is set. Each batch
    is retried with exponential backoff (which covers rate limits and network
    errors). Tickers still empty after every retry are reported as failed:
    usually delisted, renamed, or not on Yahoo.
    """
    fetch = fetch or _default_fetch
    logger = logger or get_logger()
    raw_dir.mkdir(parents=True, exist_ok=True)
    report = DownloadReport(requested=list(tickers))

    todo = []
    for t in tickers:
        if not dl.refresh and cache_path(raw_dir, t).exists():
            report.cached.append(t)
        else:
            todo.append(t)
    logger.info("%s: %d requested, %d cached, %d to download", freq.name, len(tickers), len(report.cached), len(todo))

    window = {"period": freq.period} if freq.period else {"start": freq.start, "end": freq.end}
    batches = [todo[i : i + dl.batch_size] for i in range(0, len(todo), dl.batch_size)]
    for b, batch in enumerate(batches, start=1):
        pending = list(batch)
        for attempt in range(dl.max_retries + 1):
            if attempt > 0:
                wait = dl.backoff_seconds * 2 ** (attempt - 1)
                logger.info("  retry %d for %d tickers in %.0fs", attempt, len(pending), wait)
                report.retries += 1
                sleep(wait)
            report.requests_made += 1
            try:
                result = fetch(
                    tickers=pending,
                    interval=freq.interval,
                    group_by="ticker",
                    auto_adjust=True,
                    actions=True,
                    threads=True,
                    progress=False,
                    timeout=dl.timeout,
                    **window,
                )
            except Exception as exc:  # network errors, rate limits, Yahoo hiccups
                logger.warning("  batch %d/%d request failed: %s", b, len(batches), exc)
                continue
            still_pending = []
            for t in pending:
                frame = extract_ticker(result, t)
                if frame is None:
                    still_pending.append(t)
                else:
                    frame.to_parquet(cache_path(raw_dir, t))
                    report.downloaded.append(t)
            pending = still_pending
            if not pending:
                break
        report.failed.extend(pending)
        logger.info(
            "  batch %d/%d done: %d ok, %d failed", b, len(batches), len(batch) - len(pending), len(pending)
        )
        if b < len(batches):
            sleep(dl.pause_between_batches)

    _write_manifest(raw_dir, freq, report)
    return report


def _write_manifest(raw_dir: Path, freq: FrequencyConfig, report: DownloadReport) -> None:
    """Record when each ticker was fetched. This matters for intraday data, which Yahoo only keeps for ~60 days."""
    path = raw_dir / "_manifest.csv"
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = [{"ticker": t, "status": "downloaded", "fetched_at": now} for t in report.downloaded]
    rows += [{"ticker": t, "status": "failed", "fetched_at": now} for t in report.failed]
    new = pd.DataFrame(rows, columns=["ticker", "status", "fetched_at"])
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[~old["ticker"].isin(new["ticker"])], new], ignore_index=True)
    new.assign(interval=freq.interval).sort_values("ticker").to_csv(path, index=False)


def load_cached(tickers: list[str], raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Load cached bars for the tickers that have them."""
    return {t: pd.read_parquet(p) for t in tickers if (p := cache_path(raw_dir, t)).exists()}

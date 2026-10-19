"""Downloader batching, retries, caching, and failure reporting, using a fake yfinance."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from anomaly.data.download import download_bars, load_cached

from conftest import make_bars


class FakeYahoo:
    """Mimics yf.download(group_by='ticker'): a column MultiIndex of (ticker, field)."""

    def __init__(self, known: set[str], fail_first: int = 0):
        self.known = known
        self.fail_first = fail_first
        self.calls: list[list[str]] = []

    def __call__(self, tickers, **kwargs):
        self.calls.append(list(tickers))
        if len(self.calls) <= self.fail_first:
            raise RuntimeError("Too Many Requests. Rate limited.")
        idx = pd.bdate_range("2024-01-01", periods=5, name="Date")
        frames = {}
        for t in tickers:
            bars = make_bars(idx).rename(columns={"stock_splits": "Stock Splits"})
            bars.columns = [c.title() if c != "Stock Splits" else c for c in bars.columns]
            if t not in self.known:
                bars[:] = float("nan")  # yfinance returns all-NaN columns for unknown tickers
            frames[t] = bars
        return pd.concat(frames, axis=1)


def _setup(cfg, **dl):
    return cfg.frequencies["daily"], replace(cfg.download, backoff_seconds=0, pause_between_batches=0, **dl)


def test_batches_cache_and_failures(cfg, tmp_path):
    freq, dl = _setup(cfg, batch_size=2, max_retries=1)
    fake = FakeYahoo(known={"AAA", "BBB", "CCC"})
    report = download_bars(["AAA", "BBB", "CCC", "BAD"], freq, dl, tmp_path, fetch=fake, sleep=lambda s: None)

    assert sorted(report.downloaded) == ["AAA", "BBB", "CCC"]
    assert report.failed == ["BAD"]
    assert fake.calls[0] == ["AAA", "BBB"]
    assert ["BAD"] in fake.calls  # the failed ticker was retried on its own
    cached = load_cached(["AAA", "BAD"], tmp_path)
    assert list(cached) == ["AAA"]
    assert list(cached["AAA"].columns) == ["open", "high", "low", "close", "volume", "dividends", "stock_splits"]
    assert (tmp_path / "_manifest.csv").exists()

    # A second run finds everything cached and makes no requests.
    again = FakeYahoo(known={"AAA", "BBB", "CCC"})
    report2 = download_bars(["AAA", "BBB", "CCC"], freq, dl, tmp_path, fetch=again, sleep=lambda s: None)
    assert sorted(report2.cached) == ["AAA", "BBB", "CCC"] and again.calls == []


def test_retries_with_backoff_after_rate_limit(cfg, tmp_path):
    freq, dl = _setup(cfg, batch_size=10, max_retries=3)
    dl = replace(dl, backoff_seconds=1.0)
    waits: list[float] = []
    fake = FakeYahoo(known={"AAA"}, fail_first=2)
    report = download_bars(["AAA"], freq, dl, tmp_path, fetch=fake, sleep=waits.append)

    assert report.downloaded == ["AAA"]
    assert report.retries == 2
    assert waits == [1.0, 2.0]  # exponential backoff

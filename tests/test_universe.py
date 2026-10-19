"""Ticker list parsing and seeded ticker selection."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from anomaly.data.universe import parse_us_listed, select_tickers, to_yahoo_symbol

NASDAQ_TEXT = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N
QQQ|Invesco QQQ Trust, Series 1|G|N|N|100|Y|N
ZZZT|Test Stock|Q|Y|N|100|N|N
ABCDW|ABCD Corp - Warrants|S|N|N|100|N|N
ABCDU|ABCD Corp - Units|S|N|N|100|N|N
File Creation Time: 1019202600:00|||||||
"""

OTHER_TEXT = """ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
BRK.B|Berkshire Hathaway Inc. New Common Stock|N|BRK.B|N|100|N|BRK=B
ABR$D|Arbor Realty 6.375% Series D Preferred|N|ABRpD|N|100|N|ABR-D
SPY|SPDR S&P 500 ETF Trust|P|SPY|Y|100|N|SPY
TSM|Taiwan Semiconductor American Depositary Shares|N|TSM|N|100|N|TSM
File Creation Time: 1019202600:00|||||||
"""


def test_yahoo_symbols():
    assert to_yahoo_symbol("BRK.B") == "BRK-B"
    assert to_yahoo_symbol(" bf.b ") == "BF-B"


def test_us_listed_keeps_only_common_stock():
    out = parse_us_listed(NASDAQ_TEXT, OTHER_TEXT, snapshot="2026-10-19")
    assert out["ticker"].tolist() == ["AAPL", "BRK-B", "TSM"]  # ETFs, tests, warrants, units, preferreds dropped
    assert set(out["exchange"]) == {"NASDAQ", "NYSE"}


def test_selection_is_seeded_and_adds_market(cfg, tmp_path):
    universe = pd.DataFrame({"ticker": [f"T{i:03d}" for i in range(50)] + ["SPY"]})
    file = tmp_path / "u.csv"
    universe.to_csv(file, index=False)
    small = replace(cfg, universe=replace(cfg.universe, file=file, max_tickers=10))

    first = select_tickers(small)
    assert first == select_tickers(small)  # same seed, same tickers
    assert len(first) == 11 and first[-1] == "SPY" and first.count("SPY") == 1
    assert select_tickers(replace(small, seed=small.seed + 1)) != first

    everything = replace(small, universe=replace(small.universe, max_tickers=None))
    assert len(select_tickers(everything)) == 51

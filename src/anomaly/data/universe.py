"""Build ticker-universe files and pick which tickers a run uses."""

from __future__ import annotations

import io
import re
from datetime import date

import numpy as np
import pandas as pd
import requests

from anomaly.config import Config

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

UNIVERSE_COLUMNS = ["ticker", "name", "exchange", "sector", "source", "snapshot_date"]

# Security names for things that aren't common stock.
_NON_COMMON = re.compile(
    r"\b(?:warrants?|units?|rights?|preferred|preference|notes?|debentures?|bonds?|"
    r"trust preferred|depositary shares,? each representing a \d+/\d+)\b",
    re.IGNORECASE,
)


def to_yahoo_symbol(symbol: str) -> str:
    """Convert exchange-style symbols to Yahoo's format, e.g. ``BRK.B`` becomes ``BRK-B``."""
    return symbol.strip().upper().replace(".", "-").replace("/", "-")


def _http_get(url: str, timeout: int = 30) -> str:
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (research project)"})
    response.raise_for_status()
    return response.text


def parse_sp500_html(html: str, snapshot: str) -> pd.DataFrame:
    """Turn Wikipedia's S&P 500 constituents table into the universe format."""
    tables = pd.read_html(io.StringIO(html), attrs={"id": "constituents"})
    table = tables[0]
    out = pd.DataFrame(
        {
            "ticker": table["Symbol"].map(to_yahoo_symbol),
            "name": table["Security"],
            "exchange": "",
            "sector": table["GICS Sector"],
            "source": "wikipedia_sp500",
            "snapshot_date": snapshot,
        }
    )
    return out.drop_duplicates("ticker").sort_values("ticker").reset_index(drop=True)


def _read_pipe_file(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), sep="|", dtype=str, keep_default_na=False)
    # The last line is a footer like "File Creation Time: ...".
    first = df.columns[0]
    return df[~df[first].str.startswith("File Creation Time")]


def parse_us_listed(nasdaq_text: str, other_text: str, snapshot: str) -> pd.DataFrame:
    """Combine NASDAQ Trader's two symbol files into a list of US-listed common stocks.

    Drops ETFs, test issues, and non-common securities (warrants, units,
    rights, preferreds, notes). ADRs are kept because they trade as US stocks.
    """
    nasdaq = _read_pipe_file(nasdaq_text)
    nasdaq = nasdaq[(nasdaq["Test Issue"] == "N") & (nasdaq["ETF"] == "N")]
    nasdaq_df = pd.DataFrame({"symbol": nasdaq["Symbol"], "name": nasdaq["Security Name"], "exchange": "NASDAQ"})

    other = _read_pipe_file(other_text)
    other = other[(other["Test Issue"] == "N") & (other["ETF"] == "N")]
    exchange_names = {"A": "NYSE American", "N": "NYSE", "P": "NYSE Arca", "Z": "Cboe BZX", "V": "IEX"}
    other_df = pd.DataFrame(
        {
            "symbol": other["ACT Symbol"],
            "name": other["Security Name"],
            "exchange": other["Exchange"].map(exchange_names).fillna(other["Exchange"]),
        }
    )

    combined = pd.concat([nasdaq_df, other_df], ignore_index=True)
    combined = combined[~combined["symbol"].str.contains(r"[\$\^]", regex=True)]
    combined = combined[~combined["name"].str.contains(_NON_COMMON)]
    out = pd.DataFrame(
        {
            "ticker": combined["symbol"].map(to_yahoo_symbol),
            "name": combined["name"],
            "exchange": combined["exchange"],
            "sector": "",
            "source": "nasdaqtrader",
            "snapshot_date": snapshot,
        }
    )
    return out.drop_duplicates("ticker").sort_values("ticker").reset_index(drop=True)


def fetch_sp500(snapshot: str | None = None) -> pd.DataFrame:
    """Download the current S&P 500 constituents list."""
    return parse_sp500_html(_http_get(SP500_URL), snapshot or date.today().isoformat())


def fetch_us_listed(snapshot: str | None = None) -> pd.DataFrame:
    """Download the current list of US-listed common stocks."""
    return parse_us_listed(
        _http_get(NASDAQ_LISTED_URL), _http_get(OTHER_LISTED_URL), snapshot or date.today().isoformat()
    )


def select_tickers(cfg: Config) -> list[str]:
    """Pick the tickers for this run from the universe file.

    Returns the selected stocks followed by the market ticker (always
    included, never counted toward ``max_tickers``). Random selection is
    seeded, so the same config always gives the same tickers.
    """
    universe = pd.read_csv(cfg.universe.file, dtype=str)
    market = cfg.universe.market_ticker
    tickers = [t for t in universe["ticker"].dropna().unique().tolist() if t != market]
    n = cfg.universe.max_tickers
    if n is not None and n < len(tickers):
        if cfg.universe.selection == "random":
            rng = np.random.default_rng(cfg.seed)
            tickers = sorted(rng.choice(tickers, size=n, replace=False).tolist())
        else:
            tickers = tickers[:n]
    return [*tickers, market]

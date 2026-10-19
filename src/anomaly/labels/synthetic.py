"""Inject synthetic anomalies into cleaned prices at known times.

Anomalies go into the *prices and volumes*, not the features. Features are
then recomputed from the modified bars, so an injected spike affects returns,
volatility, gaps, and correlation exactly the way a real spike would.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from anomaly.config import SyntheticConfig
from anomaly.labels import EVENT_COLUMNS

TYPES = ("price_spike", "volume_burst", "level_shift")
OHLC = ["open", "high", "low", "close"]


def trailing_vol(clean: pd.DataFrame, window: int) -> pd.Series:
    """Std of log returns over the previous ``window`` bars (excluding the current bar)."""
    log_ret = np.log(clean["close"]).groupby(clean["ticker"], sort=False).diff()
    vol = log_ret.groupby(clean["ticker"], sort=False).transform(
        lambda s: s.rolling(window, min_periods=window).std()
    )
    return vol.groupby(clean["ticker"], sort=False).shift(1)


def plan_events(
    clean: pd.DataFrame,
    eligible: pd.Series,
    cfg: SyntheticConfig,
    rng: np.random.Generator,
    blocked: pd.Series | None = None,
) -> pd.DataFrame:
    """Choose where each synthetic anomaly goes and how big it is.

    Args:
        clean: cleaned bars sorted by (ticker, timestamp) with a default RangeIndex.
        eligible: boolean mask of rows that may start an event (e.g. val/test rows).
        cfg: synthetic-anomaly settings.
        rng: seeded random generator.
        blocked: optional mask of rows to keep events away from (real events).

    Returns:
        One row per event: EVENT_COLUMNS plus ``row`` (start position in ``clean``)
        and ``length`` (bars the event lasts).
    """
    vol = trailing_vol(clean, cfg.vol_window)
    ok = eligible & vol.notna() & (vol > 0) & ~clean["is_filled"]
    blocked = blocked if blocked is not None else pd.Series(False, index=clean.index)
    gap = cfg.min_gap_bars
    events = []
    for ticker, rows in clean.groupby("ticker", sort=True).indices.items():
        first, last = rows[0], rows[-1]
        tail_room = max(cfg.level_shift_label_bars, cfg.volume_burst_bars[1]) + gap
        candidates = [r for r in rows if ok.iat[r] and r <= last - tail_room]
        if not candidates:
            continue
        counts = rng.poisson(cfg.rate_per_1000_bars * len(candidates) / 1000, size=len(TYPES))
        kinds = rng.permutation(np.repeat(TYPES, counts))
        blocked_rows = [r for r in rows if blocked.iat[r]]
        taken: list[int] = []
        shuffled = iter(rng.permutation(candidates))
        for kind in kinds:
            # Walk the shuffled candidates until one is far enough from every other event.
            row = next(
                (r for r in shuffled if all(abs(r - t) >= gap for t in taken) and all(abs(r - b) >= gap for b in blocked_rows)),
                None,
            )
            if row is None:
                break  # this ticker has no room left
            taken.append(int(row))
            sigma = float(vol.iat[row])
            if kind == "price_spike":
                mult = rng.uniform(*cfg.price_spike_sigma) * rng.choice([-1, 1])
                magnitude, shock, length = mult, mult * sigma, 1
            elif kind == "volume_burst":
                magnitude = rng.uniform(*cfg.volume_burst_factor)
                shock = magnitude
                length = int(rng.integers(cfg.volume_burst_bars[0], cfg.volume_burst_bars[1] + 1))
            else:
                mult = rng.uniform(*cfg.level_shift_sigma) * rng.choice([-1, 1])
                magnitude, shock, length = mult, mult * sigma, cfg.level_shift_label_bars
            events.append(
                {
                    "ticker": ticker,
                    "type": kind,
                    "row": int(row),
                    "length": length,
                    "start": clean.at[row, "timestamp"],
                    "end": clean.at[min(row + length - 1, last), "timestamp"],
                    "magnitude": round(float(magnitude), 4),
                    "shock": float(shock),
                    "ticker_last_row": int(last),
                    "ticker_first_row": int(first),
                }
            )
    out = pd.DataFrame(events)
    if out.empty:
        return pd.DataFrame(columns=[*EVENT_COLUMNS, "row", "length", "shock", "ticker_last_row"])
    out = out.sort_values(["ticker", "start"], ignore_index=True)
    out["event_id"] = [f"syn-{i:05d}" for i in range(len(out))]
    out["track"] = "synthetic"
    return out[[*EVENT_COLUMNS, "row", "length", "shock", "ticker_last_row"]]


def inject(clean: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Apply planned events to a copy of the cleaned bars.

    - price_spike: the close jumps by ``shock`` (log units) for one bar; the
      next bar's open is unchanged, so the price snaps back.
    - volume_burst: volume is multiplied by ``shock`` for ``length`` bars.
    - level_shift: every price from the start bar to the end of the ticker's
      history moves by ``shock`` (log units), so the jump is permanent.
    """
    out = clean.copy()
    close = out.columns.get_loc("close")
    high = out.columns.get_loc("high")
    low = out.columns.get_loc("low")
    volume = out.columns.get_loc("volume")
    ohlc = [out.columns.get_loc(c) for c in OHLC]
    for ev in events.itertuples(index=False):
        if ev.type == "price_spike":
            out.iat[ev.row, close] *= np.exp(ev.shock)
            out.iat[ev.row, high] = max(out.iat[ev.row, high], out.iat[ev.row, close])
            out.iat[ev.row, low] = min(out.iat[ev.row, low], out.iat[ev.row, close])
        elif ev.type == "volume_burst":
            out.iloc[ev.row : ev.row + ev.length, volume] *= ev.shock
        elif ev.type == "level_shift":
            out.iloc[ev.row : ev.ticker_last_row + 1, ohlc] *= np.exp(ev.shock)
        else:
            raise ValueError(f"Unknown synthetic event type: {ev.type}")
    return out

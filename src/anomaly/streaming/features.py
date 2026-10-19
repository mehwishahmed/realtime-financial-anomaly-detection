"""Compute features one bar at a time from small per-ticker buffers.

The batch pipeline (``anomaly.data.features.build_features``) computes rolling
features over whole columns. A live system can't: it only has the bars that
have arrived. This engine keeps just the state each feature needs (the last
20 returns, the last 60 return pairs with the market, and so on) and must
produce the same numbers as the batch code. Tests check that it does.

Bars must arrive in time order, with the market ticker's bar first at each timestamp.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from anomaly.config import FeatureConfig, FrequencyConfig

NAN = float("nan")


@dataclass
class _TickerState:
    prev_log_close: float = NAN
    rets: deque = field(default_factory=deque)  # last max(vol_windows) log returns
    log_vols: deque = field(default_factory=deque)  # rolling mode: last N log volumes (before the current bar)
    slot_vols: dict = field(default_factory=dict)  # time_of_day mode: time -> last N log volumes in that slot
    pairs: deque = field(default_factory=deque)  # last corr_window (ret, market ret) pairs, NaN if either missing
    last_emitted_vol: float = NAN  # vol_<first window> of the previous emitted bar (for ret_z / gap_z / the rule)


def _sample_std(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return NAN
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


class StreamingFeatures:
    """Incremental version of the Phase 1 features plus the derived ``ret_z`` / ``gap_z``."""

    def __init__(self, freq: FrequencyConfig, fcfg: FeatureConfig, market_ticker: str):
        self.freq = freq
        self.fcfg = fcfg
        self.market = market_ticker
        self.vol_windows = tuple(fcfg.vol_windows)
        self.max_ret_window = max(self.vol_windows)
        self.vz_window = fcfg.volume_z_window if freq.volume_z_mode == "rolling" else fcfg.volume_z_days
        self.vz_min = max(2, math.ceil(0.8 * self.vz_window))
        self.states: dict[str, _TickerState] = {}
        self.market_ret: dict[Any, float] = {}  # timestamp -> market log return
        self.columns = ["log_ret", *[f"vol_{w}" for w in self.vol_windows], "volume_z", "gap", "corr_market", "ret_z", "gap_z"]

    def _state(self, ticker: str) -> _TickerState:
        st = self.states.get(ticker)
        if st is None:
            st = _TickerState(
                rets=deque(maxlen=self.max_ret_window),
                log_vols=deque(maxlen=self.vz_window),
                pairs=deque(maxlen=self.fcfg.corr_window),
            )
            self.states[ticker] = st
        return st

    def _volume_z(self, st: _TickerState, timestamp: Any, log_vol: float) -> float:
        if self.freq.volume_z_mode == "rolling":
            history = st.log_vols
        else:
            history = st.slot_vols.setdefault(timestamp.time(), deque(maxlen=self.vz_window))
        past = [v for v in history if not math.isnan(v)]
        history.append(log_vol)  # the current bar joins the baseline only after it has been scored
        if len(past) < self.vz_min:
            return NAN
        mean = sum(past) / len(past)
        std = _sample_std(past)
        if not std > 0 or math.isnan(log_vol):
            return NAN
        return (log_vol - mean) / std

    def _corr(self, st: _TickerState) -> float:
        if len(st.pairs) < self.fcfg.corr_window or any(math.isnan(x) for x, _ in st.pairs):
            return NAN
        n = len(st.pairs)
        mx = sum(x for x, _ in st.pairs) / n
        my = sum(y for _, y in st.pairs) / n
        mxy = sum(x * y for x, y in st.pairs) / n
        mxx = sum(x * x for x, _ in st.pairs) / n
        myy = sum(y * y for _, y in st.pairs) / n
        cov = mxy - mx * my
        denom = math.sqrt(max(mxx - mx * mx, 0.0) * max(myy - my * my, 0.0))
        if not denom > 1e-12:
            return NAN
        return min(1.0, max(-1.0, cov / denom))

    def update(self, ticker: str, timestamp: Any, open_: float, close: float, volume: float) -> dict[str, float] | None:
        """Feed one bar. Returns its features, or None if this bar wouldn't be in the batch feature table.

        None means it's the market ticker, a warm-up bar, or a bar with a missing
        feature (for example, unknown volume). Feature values are returned as
        float32-rounded floats, matching the batch table's storage type.
        """
        st = self._state(ticker)
        log_close = math.log(close)
        log_ret = log_close - st.prev_log_close
        gap = math.log(open_) - st.prev_log_close
        st.prev_log_close = log_close
        st.rets.append(log_ret)

        if ticker == self.market:
            self.market_ret[timestamp] = log_ret
            return None

        vols = {}
        for w in self.vol_windows:
            window = list(st.rets)[-w:]
            vols[f"vol_{w}"] = _sample_std(window) if len(window) == w and not any(math.isnan(r) for r in window) else NAN
        log_vol = math.log1p(volume) if not (volume is None or math.isnan(volume)) else NAN
        volume_z = self._volume_z(st, timestamp, log_vol)
        mkt = self.market_ret.get(timestamp, NAN)
        pair = (log_ret, mkt) if not (math.isnan(log_ret) or math.isnan(mkt)) else (NAN, NAN)
        st.pairs.append(pair)
        corr = self._corr(st)

        base = {"log_ret": log_ret, **vols, "volume_z": volume_z, "gap": gap, "corr_market": corr}
        if any(math.isnan(v) for v in base.values()):
            return None
        f32 = {k: float(np.float32(v)) for k, v in base.items()}
        # Derived features use the previous *emitted* bar's volatility, exactly as add_derived_features does.
        first_vol = f32[f"vol_{self.vol_windows[0]}"]
        prior = st.last_emitted_vol
        st.last_emitted_vol = first_vol
        prior32 = np.float32(prior) if prior > 0 else np.float32("nan")
        f32["ret_z"] = float(np.float32(np.float32(f32["log_ret"]) / prior32))
        f32["gap_z"] = float(np.float32(np.float32(f32["gap"]) / prior32))
        f32["prior_vol"] = float(prior32)
        return f32

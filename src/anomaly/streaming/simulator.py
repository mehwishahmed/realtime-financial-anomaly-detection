"""Score bars one at a time, as a live system would, and time every stage.

Per incoming bar:
    1. features   update the ticker's buffers and compute this bar's features
    2. members    score the bar with each ensemble member (rule, Isolation Forest,
                  LSTM autoencoder, Transformer); the deep models use the ticker's
                  last ``window`` feature vectors
    3. combine    normalize the member scores and take the ensemble's weighted average
    4. alarm      compare with the threshold tuned on val
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

import numpy as np
import pandas as pd
import torch

from anomaly.models.deep import WindowModel
from anomaly.models.ensemble import EnsembleModel
from anomaly.models.isolation_forest import IsolationForestModel
from anomaly.models.rules import ZScoreRuleModel
from anomaly.streaming.features import StreamingFeatures


class StreamingScorer:
    """Wraps a fitted ``EnsembleModel`` for one-bar-at-a-time scoring."""

    def __init__(self, ensemble: EnsembleModel, threshold: float, torch_threads: int = 1):
        self.ensemble = ensemble
        self.threshold = threshold
        self.names = ensemble.order
        self.weights = np.array([ensemble.weights[n] for n in self.names])
        torch.set_num_threads(torch_threads)
        self.windows: dict[tuple[str, str], deque] = {}
        self._scale_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        for name in self.names:
            member = ensemble.members[name]
            if isinstance(member, IsolationForestModel):
                member.forest.set_params(n_jobs=1)  # a parallel pool costs more than it saves for one row
            if isinstance(member, WindowModel):
                assert member.net is not None
                member.net.eval()

    def _scaled(self, name: str, member: Any, ticker: str, feats: dict[str, float]) -> np.ndarray:
        key = (name, ticker)
        if key not in self._scale_cache:
            self._scale_cache[key] = member.scaler.params_for(ticker)
        mean, std = self._scale_cache[key]
        x = np.array([feats[c] for c in member.features], dtype="float64")
        return ((x - mean) / std).astype("float32")

    def observe(self, ticker: str, feats: dict[str, float], score: bool = True) -> tuple[np.ndarray, dict[str, int]]:
        """Add one bar's features to the buffers and (optionally) score it.

        Returns raw member scores (NaN where a member can't score yet) and
        nanoseconds spent in each member.
        """
        raw = np.full(len(self.names), np.nan)
        timing: dict[str, int] = {}
        for i, name in enumerate(self.names):
            member = self.ensemble.members[name]
            start = time.perf_counter_ns()
            if isinstance(member, ZScoreRuleModel):
                if score:
                    vals = (abs(feats["ret_z"]), abs(feats["gap_z"]), abs(feats["volume_z"]))
                    raw[i] = np.nan if any(np.isnan(v) for v in vals) else float(np.float32(max(vals)))
            elif isinstance(member, IsolationForestModel):
                if score:
                    z = self._scaled(name, member, ticker, feats)
                    if not np.isnan(z).any():
                        raw[i] = -member.forest.score_samples(z[None, :])[0]
            elif isinstance(member, WindowModel):
                buf = self.windows.setdefault((name, ticker), deque(maxlen=member.window))
                buf.append(self._scaled(name, member, ticker, feats))
                if score and len(buf) == member.window:
                    x = np.stack(buf)
                    if not np.isnan(x).any():
                        with torch.no_grad():
                            _, s = member.errors(member.net, torch.from_numpy(x[None]))
                        raw[i] = float(s[0])
            else:
                raise TypeError(f"Streaming doesn't know how to score member {name} ({type(member).__name__})")
            timing[name] = time.perf_counter_ns() - start
        return raw, timing

    def combine(self, raw: np.ndarray) -> float:
        """Ensemble score from one bar's raw member scores."""
        return float(self.ensemble.combine_raw(raw[None, :])[0])


def stream_order(bars: pd.DataFrame, market: str) -> pd.DataFrame:
    """Time order, with the market ticker first at each timestamp (its return feeds the correlation)."""
    keyed = bars.assign(_not_market=bars["ticker"] != market)
    return keyed.sort_values(["timestamp", "_not_market", "ticker"], kind="stable").drop(columns="_not_market")


def replay(
    bars: pd.DataFrame,
    engine: StreamingFeatures,
    scorer: StreamingScorer,
    start: Any,
    market: str,
    max_timed_bars: int | None = None,
) -> pd.DataFrame:
    """Feed bars through the engine and scorer in stream order.

    Bars before ``start`` only warm up the buffers (untimed, unscored). From
    ``start`` on, every bar is timed and, if it produces features, scored.
    Returns one row per timed stock bar with member scores, the ensemble
    score, the alarm flag, and per-stage nanoseconds.
    """
    records = []
    timed = 0
    for bar in stream_order(bars, market).itertuples(index=False):
        live = bar.timestamp >= start
        if live and max_timed_bars is not None and timed >= max_timed_bars:
            break
        t0 = time.perf_counter_ns()
        feats = engine.update(bar.ticker, bar.timestamp, bar.open, bar.close, bar.volume)
        t_features = time.perf_counter_ns() - t0
        if bar.ticker == market:
            continue
        if not live:
            if feats is not None:
                scorer.observe(bar.ticker, feats, score=False)
            continue
        timed += 1
        rec: dict[str, Any] = {"ticker": bar.ticker, "timestamp": bar.timestamp, "t_features": t_features}
        if feats is None:
            rec.update({"scored": False, "ensemble": np.nan, "alarm": False})
        else:
            raw, timing = scorer.observe(bar.ticker, feats, score=True)
            t1 = time.perf_counter_ns()
            ens = scorer.combine(raw)
            rec["t_combine"] = time.perf_counter_ns() - t1
            rec.update({f"t_{n}": v for n, v in timing.items()})
            rec.update({n: raw[i] for i, n in enumerate(scorer.names)})
            rec.update({"scored": True, "ensemble": ens, "alarm": bool(np.isfinite(ens) and ens >= scorer.threshold)})
        records.append(rec)
    return pd.DataFrame(records)


def latency_summary(records: pd.DataFrame, names: list[str], weights: dict[str, float], run: str) -> pd.DataFrame:
    """p50/p95/p99 per stage in microseconds, plus the deployed-ensemble and all-members totals."""
    scored = records[records["scored"]].copy()
    deployed = [n for n in names if weights[n] > 0]
    scored["t_deployed_total"] = scored["t_features"] + scored[[f"t_{n}" for n in deployed]].sum(axis=1) + scored["t_combine"]
    scored["t_all_members_total"] = scored["t_features"] + scored[[f"t_{n}" for n in names]].sum(axis=1) + scored["t_combine"]
    stages = ["t_features", *[f"t_{n}" for n in names], "t_combine", "t_deployed_total", "t_all_members_total"]
    rows = []
    for stage in stages:
        us = scored[stage].to_numpy(dtype=float) / 1000
        rows.append(
            {
                "run": run,
                "stage": stage.removeprefix("t_"),
                "bars": len(us),
                "mean_us": us.mean(),
                "p50_us": np.percentile(us, 50),
                "p95_us": np.percentile(us, 95),
                "p99_us": np.percentile(us, 99),
                "max_us": us.max(),
            }
        )
    return pd.DataFrame(rows)

"""The streaming path must reproduce the batch pipeline exactly, using only past bars."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from anomaly.data.clean import clean_universe
from anomaly.data.features import build_features
from anomaly.models.deep import LSTMAutoencoderModel
from anomaly.models.ensemble import EnsembleModel
from anomaly.models.isolation_forest import IsolationForestModel
from anomaly.models.preprocess import add_derived_features
from anomaly.models.rules import ZScoreRuleModel
from anomaly.streaming.features import StreamingFeatures
from anomaly.streaming.simulator import StreamingScorer, latency_summary, replay, stream_order

from conftest import make_bars

FEATURES = ["log_ret", "vol_20", "volume_z", "gap", "corr_market", "ret_z", "gap_z"]


def _clean_daily(cfg, n=320):
    days = pd.bdate_range("2021-01-01", periods=n)
    stock = make_bars(days, seed=1)
    stock = stock.drop(days[150])  # a one-day gap: filled, with unknown volume
    stock.loc[days[200], "volume"] *= 15  # a volume spike
    other = make_bars(days, seed=2)
    market = make_bars(days, seed=0).drop(days[250])  # the market misses a day: a holiday for everyone
    freq = replace(cfg.frequencies["daily"], min_history_bars=100)
    return clean_universe({"SPY": market, "AAA": stock, "BBB": other}, "SPY", freq, replace(cfg.cleaning, max_missing_frac=0.5)).data


def _stream(clean, freq, fcfg):
    eng = StreamingFeatures(freq, fcfg, "SPY")
    rows = []
    for bar in stream_order(clean, "SPY").itertuples(index=False):
        f = eng.update(bar.ticker, bar.timestamp, bar.open, bar.close, bar.volume)
        if f is not None:
            rows.append({"ticker": bar.ticker, "timestamp": bar.timestamp, **f})
    return pd.DataFrame(rows)


def _compare(batch: pd.DataFrame, streamed: pd.DataFrame):
    merged = batch.merge(streamed, on=["ticker", "timestamp"], how="outer", suffixes=("_b", "_s"), indicator=True)
    assert (merged["_merge"] == "both").all(), merged["_merge"].value_counts()
    for col in FEATURES:
        np.testing.assert_allclose(merged[f"{col}_s"], merged[f"{col}_b"].astype(float), rtol=1e-5, atol=1e-6, equal_nan=True, err_msg=col)


def test_streamed_daily_features_equal_batch(cfg):
    clean = _clean_daily(cfg)
    freq = replace(cfg.frequencies["daily"], min_history_bars=100)
    batch = add_derived_features(build_features(clean, "SPY", freq, cfg.features))
    _compare(batch, _stream(clean, freq, cfg.features))


def test_streamed_time_of_day_features_equal_batch(cfg):
    sessions = pd.bdate_range("2026-08-03", periods=25)
    slots = pd.timedelta_range("09:30:00", "15:55:00", freq="5min")
    index = pd.DatetimeIndex([d + s for d in sessions for s in slots]).tz_localize("America/New_York")
    stock = make_bars(index, seed=1, daily_vol=0.001)
    stock["volume"] = (stock["volume"] * np.tile(np.where(np.arange(78) < 6, 8.0, 1.0), len(sessions))).round()
    freq = replace(cfg.frequencies["intraday"], min_history_bars=100)
    clean = clean_universe({"SPY": make_bars(index, seed=0, daily_vol=0.001), "AAA": stock}, "SPY", freq, cfg.cleaning).data
    fcfg = replace(cfg.features, corr_window=20)
    batch = add_derived_features(build_features(clean, "SPY", freq, fcfg))
    _compare(batch, _stream(clean, freq, fcfg))


def test_streaming_features_never_change_when_the_future_changes(cfg):
    clean = _clean_daily(cfg)
    freq = replace(cfg.frequencies["daily"], min_history_bars=100)
    cut = clean["timestamp"].sort_values().unique()[260]
    future = clean["timestamp"] > cut
    wild = clean.copy()
    wild.loc[future, ["open", "high", "low", "close"]] *= 5
    a = _stream(clean, freq, cfg.features)
    b = _stream(wild, freq, cfg.features)
    pd.testing.assert_frame_equal(a[a["timestamp"] <= cut].reset_index(drop=True), b[b["timestamp"] <= cut].reset_index(drop=True))


@pytest.fixture
def small_ensemble(cfg):
    clean = _clean_daily(cfg)
    freq = replace(cfg.frequencies["daily"], min_history_bars=100)
    feats = add_derived_features(build_features(clean, "SPY", freq, cfg.features))
    cut = feats["timestamp"].sort_values().unique()[200]
    train = feats[feats["timestamp"] <= cut]
    members = {
        "baseline_zscore_rule": ZScoreRuleModel(),
        "isolation_forest": IsolationForestModel(FEATURES, n_estimators=20, max_samples=64, min_train_rows_per_ticker=50, seed=0, n_jobs=1).fit(train),
        "lstm_autoencoder": LSTMAutoencoderModel(FEATURES, hidden_size=4, window=8, batch_size=64, max_epochs=2, patience=2,
                                                 holdout_frac=0.2, torch_threads=1, min_train_rows_per_ticker=50, seed=0).fit(train),
    }
    ens = EnsembleModel(members, {"baseline_zscore_rule": 0.6, "isolation_forest": 0.2, "lstm_autoencoder": 0.2}, "robust_z").fit(train)
    return clean, freq, feats, cut, ens


def test_replayed_scores_equal_batch_scores(cfg, small_ensemble):
    clean, freq, feats, cut, ens = small_ensemble
    batch = feats[["ticker", "timestamp"]].assign(batch=ens.score(feats))
    threshold = float(np.nanquantile(batch["batch"], 0.95))
    records = replay(clean, StreamingFeatures(freq, cfg.features, "SPY"), StreamingScorer(ens, threshold), cut, "SPY")

    live = batch[batch["timestamp"] >= cut]
    got = live.merge(records[records["scored"]][["ticker", "timestamp", "ensemble", "alarm"]], on=["ticker", "timestamp"], how="left")
    assert got["ensemble"].notna().sum() == got["batch"].notna().sum()
    np.testing.assert_allclose(got["ensemble"], got["batch"], rtol=1e-4, atol=1e-5, equal_nan=True)
    assert (got["alarm"] == (got["batch"] >= threshold)).all()
    assert records["timestamp"].min() >= cut and "SPY" not in set(records["ticker"])  # warm-up and market bars aren't recorded


def test_latency_summary_has_every_stage(cfg, small_ensemble):
    clean, freq, feats, cut, ens = small_ensemble
    records = replay(clean, StreamingFeatures(freq, cfg.features, "SPY"), StreamingScorer(ens, 1.0), cut, "SPY", max_timed_bars=50)
    assert len(records) == 50
    summary = latency_summary(records, ens.order, ens.weights, "toy").set_index("stage")
    assert {"features", "combine", "deployed_total", "all_members_total", *ens.order} <= set(summary.index)
    assert (summary["p50_us"] <= summary["p95_us"]).all() and (summary["p95_us"] <= summary["p99_us"]).all()
    assert summary.loc["deployed_total", "p50_us"] <= summary.loc["all_members_total", "p50_us"] + 1e-9

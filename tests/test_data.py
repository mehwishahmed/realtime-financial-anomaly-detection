"""Cleaning, features, and splits: holidays, gaps, splits, bad tickers, and no lookahead."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from anomaly.data.clean import clean_universe
from anomaly.data.features import build_features, feature_columns
from anomaly.data.splits import assign_splits
from anomaly.config import SplitConfig

from conftest import make_bars


def _clean(cfg, raw, freq="daily"):
    return clean_universe(raw, "SPY", cfg.frequencies[freq], cfg.cleaning)


# ---------- cleaning ----------


def test_short_gap_is_filled_and_flagged(cfg, business_days):
    stock = make_bars(business_days, seed=1).drop(business_days[100:102])
    result = _clean(cfg, {"SPY": make_bars(business_days), "AAA": stock})
    aaa = result.data[result.data["ticker"] == "AAA"].set_index("timestamp")
    assert len(aaa) == len(business_days)
    assert aaa.loc[business_days[100:102], "is_filled"].all()
    assert (aaa.loc[business_days[100:102], "close"] == aaa.loc[business_days[99], "close"]).all()
    assert aaa.loc[business_days[100:102], "volume"].isna().all()  # no fake volume on filled bars
    assert aaa["is_filled"].sum() == 2


def test_market_holidays_come_from_the_market_calendar(cfg, business_days):
    holiday = business_days[50]
    market = make_bars(business_days).drop(holiday)  # SPY didn't trade that day
    stock = make_bars(business_days, seed=1)  # a stray bar on the holiday
    result = _clean(cfg, {"SPY": market, "AAA": stock})
    assert holiday not in set(result.data["timestamp"])


def test_bad_tickers_are_rejected_with_a_reason(cfg, business_days):
    gappy = make_bars(business_days, seed=2).iloc[::2]  # half the bars missing
    short = make_bars(business_days[-100:], seed=3)
    dead = make_bars(business_days, seed=4)
    dead.loc[dead.index[:200], "volume"] = 0
    result = _clean(cfg, {"SPY": make_bars(business_days), "GAP": gappy, "NEW": short, "ZERO": dead})
    reasons = result.report.set_index("ticker")["reason"].to_dict()
    assert reasons == {"GAP": "too_many_missing", "NEW": "insufficient_history", "SPY": "", "ZERO": "too_many_zero_volume"}
    assert set(result.data["ticker"]) == {"SPY"}


def test_nonpositive_prices_are_dropped(cfg, business_days):
    stock = make_bars(business_days, seed=1)
    stock.iloc[300, stock.columns.get_loc("close")] = -1.0
    result = _clean(cfg, {"SPY": make_bars(business_days), "AAA": stock})
    row = result.report.set_index("ticker").loc["AAA"]
    assert row["nonpositive_or_nan_rows"] == 1 and row["status"] == "ok"
    assert (result.data["close"] > 0).all()


def test_unadjusted_split_is_repaired(cfg, business_days):
    stock = make_bars(business_days, seed=1, daily_vol=0.005)
    day = business_days[400]
    stock.loc[stock.index >= day, ["open", "high", "low", "close"]] /= 4  # 4:1 split, never adjusted
    stock.loc[day, "stock_splits"] = 4.0
    result = _clean(cfg, {"SPY": make_bars(business_days), "AAA": stock})
    aaa = result.data[result.data["ticker"] == "AAA"].set_index("timestamp")
    jump = np.log(aaa["close"]).diff().loc[day]
    assert abs(jump) < 0.05
    assert result.report.set_index("ticker").loc["AAA", "split_repairs"] == 1


def test_small_stock_dividend_is_not_mistaken_for_a_split(cfg, business_days):
    stock = make_bars(business_days, seed=1)
    stock.loc[business_days[400], "stock_splits"] = 1.03  # already adjusted by Yahoo
    result = _clean(cfg, {"SPY": make_bars(business_days), "AAA": stock})
    aaa = result.data[result.data["ticker"] == "AAA"].set_index("timestamp")
    assert result.report.set_index("ticker").loc["AAA", "split_repairs"] == 0
    np.testing.assert_allclose(aaa["close"].to_numpy(), stock["close"].to_numpy())


def test_missing_market_ticker_raises(cfg, business_days):
    with pytest.raises(ValueError, match="SPY"):
        _clean(cfg, {"AAA": make_bars(business_days)})


# ---------- features ----------


def _features(cfg, raw, freq="daily"):
    clean = _clean(cfg, raw, freq).data
    return build_features(clean, "SPY", cfg.frequencies[freq], cfg.features)


def test_feature_values(cfg, business_days):
    market = make_bars(business_days, seed=0)
    feats = _features(cfg, {"SPY": market, "AAA": make_bars(business_days, seed=1)})
    assert set(feats["ticker"]) == {"AAA"}  # the market ticker is not a modeled asset
    assert feats[feature_columns(cfg.features)].notna().all().all()

    aaa = feats.set_index("timestamp")
    raw = make_bars(business_days, seed=1)
    t = aaa.index[10]
    prev = raw.index[raw.index.get_loc(t) - 1]
    assert aaa.loc[t, "log_ret"] == pytest.approx(np.log(raw.loc[t, "close"] / raw.loc[prev, "close"]), rel=1e-5)
    assert aaa.loc[t, "gap"] == pytest.approx(np.log(raw.loc[t, "open"] / raw.loc[prev, "close"]), rel=1e-5)


def test_correlation_to_market(cfg, business_days):
    market = make_bars(business_days, seed=0)
    twin = market.copy()  # moves exactly with the market
    mirror = market.copy()
    mirror[["open", "high", "low", "close"]] = 1e4 / market[["open", "low", "high", "close"]].to_numpy()
    feats = _features(cfg, {"SPY": market, "TWIN": twin, "MIRR": mirror}).groupby("ticker")["corr_market"]
    assert feats.min()["TWIN"] > 0.999
    assert feats.max()["MIRR"] < -0.999


def test_volume_spike_stands_out(cfg, business_days):
    stock = make_bars(business_days, seed=1)
    spike_day = business_days[500]
    stock.loc[spike_day, "volume"] *= 20
    feats = _features(cfg, {"SPY": make_bars(business_days), "AAA": stock}).set_index("timestamp")
    assert feats.loc[spike_day, "volume_z"] > 8
    assert feats["volume_z"].drop(spike_day).abs().max() < 6


def test_time_of_day_volume_ignores_the_daily_pattern(cfg):
    sessions = pd.bdate_range("2026-08-03", periods=30)
    bars_per_day = pd.timedelta_range("09:30:00", "15:55:00", freq="5min")
    index = pd.DatetimeIndex([d + b for d in sessions for b in bars_per_day]).tz_localize("America/New_York")
    stock = make_bars(index, seed=1, daily_vol=0.001)
    u_shape = np.tile(np.where((np.arange(78) < 6) | (np.arange(78) > 71), 10.0, 1.0), len(sessions))
    stock["volume"] = (stock["volume"] * u_shape).round()  # heavy volume at the open and close, every day
    freq = replace(cfg.frequencies["intraday"], min_history_bars=100)
    features = replace(cfg.features, corr_window=20)
    clean = clean_universe({"SPY": make_bars(index), "AAA": stock}, "SPY", freq, cfg.cleaning).data

    by_slot = build_features(clean, "SPY", freq, features)
    rolling = build_features(clean, "SPY", replace(freq, volume_z_mode="rolling"), features)
    assert by_slot["volume_z"].abs().quantile(0.99) < 4  # the open isn't unusual compared with earlier opens
    assert rolling["volume_z"].abs().quantile(0.99) > 6  # a plain rolling window flags every open


def test_no_lookahead(cfg, business_days):
    """Features at time t must not change when future data changes."""
    raw = {"SPY": make_bars(business_days, seed=0), "AAA": make_bars(business_days, seed=1)}
    full = _features(cfg, raw).set_index("timestamp")

    cut = business_days[600]
    shocked = {t: df.copy() for t, df in raw.items()}
    for df in shocked.values():
        after = df.index > cut
        df.loc[after, ["open", "high", "low", "close"]] *= 3.0  # wild future
        df.loc[after, "volume"] *= 50
    changed = _features(cfg, shocked).set_index("timestamp")

    cols = feature_columns(cfg.features)
    pd.testing.assert_frame_equal(full.loc[:cut, cols], changed.loc[:cut, cols])
    assert not np.allclose(full.loc[cut:, "volume_z"].iloc[1:5], changed.loc[cut:, "volume_z"].iloc[1:5])


# ---------- splits ----------


def test_date_splits_are_ordered_and_disjoint(business_days):
    df = pd.DataFrame({"timestamp": business_days, "x": 0})
    out, bounds = assign_splits(df, SplitConfig(train_end="2021-06-30", val_end="2022-01-31"))
    ts = {s: out.loc[out["split"] == s, "timestamp"] for s in ("train", "val", "test")}
    assert ts["train"].max() < ts["val"].min() and ts["val"].max() < ts["test"].min()
    assert ts["train"].max() <= bounds["train_end"] and ts["val"].max() <= bounds["val_end"]
    assert len(out) == sum(len(v) for v in ts.values())


def test_fraction_splits_never_cut_a_day():
    sessions = pd.bdate_range("2026-08-03", periods=10)
    index = pd.DatetimeIndex(
        [d + pd.Timedelta(minutes=570 + 5 * k) for d in sessions for k in range(78)]
    ).tz_localize("America/New_York")
    out, _ = assign_splits(pd.DataFrame({"timestamp": index}), SplitConfig(train_frac=0.6, val_frac=0.2))
    per_day = out.groupby(out["timestamp"].dt.date)["split"].nunique()
    assert (per_day == 1).all()
    days = out.groupby("split", observed=True)["timestamp"].apply(lambda s: s.dt.date.nunique()).to_dict()
    assert days == {"train": 6, "val": 2, "test": 2}

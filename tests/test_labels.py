"""Synthetic injection, real-event tagging, and per-row label columns."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from anomaly.labels import EVENT_COLUMNS
from anomaly.labels.attach import attach_labels
from anomaly.labels.events import download_earnings, reaction_days, tag_real_events
from anomaly.labels.synthetic import inject, plan_events

from conftest import make_bars


def _clean_frame(index: pd.DatetimeIndex, tickers=("AAA", "BBB")) -> pd.DataFrame:
    frames = []
    for i, t in enumerate(tickers):
        bars = make_bars(index, seed=i).drop(columns=["dividends", "stock_splits"])
        frames.append(bars.rename_axis("timestamp").reset_index().assign(ticker=t, is_filled=False))
    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "timestamp"], ignore_index=True)


# ---------- synthetic ----------


@pytest.fixture
def planned(cfg, business_days):
    clean = _clean_frame(business_days)
    eligible = clean["timestamp"] >= business_days[300]
    blocked = (clean["ticker"] == "AAA") & (clean["timestamp"] == business_days[500])
    syn = replace(cfg.labels.synthetic, rate_per_1000_bars=8.0)
    events = plan_events(clean, eligible, syn, np.random.default_rng(0), blocked=blocked)
    return clean, eligible, blocked, syn, events


def test_events_respect_eligibility_spacing_and_blocks(planned):
    clean, eligible, blocked, syn, events = planned
    assert len(events) > 10
    assert set(events["type"]) == {"price_spike", "volume_burst", "level_shift"}
    assert eligible.iloc[events["row"]].all()
    for _, g in events.groupby("ticker"):
        assert np.diff(np.sort(g["row"].to_numpy())).min() >= syn.min_gap_bars
    blocked_row = int(np.flatnonzero(blocked)[0])
    assert (np.abs(events["row"] - blocked_row) >= syn.min_gap_bars).all()


def test_event_count_matches_the_configured_rate(cfg, business_days):
    """Collisions must move an event elsewhere, not silently drop it."""
    clean = _clean_frame(business_days, tickers=tuple(f"T{i}" for i in range(30)))
    eligible = clean["timestamp"] >= business_days[300]
    syn = replace(cfg.labels.synthetic, rate_per_1000_bars=6.0)  # ~10 events per ticker, so collisions are common
    events = plan_events(clean, eligible, syn, np.random.default_rng(0))
    candidates = 30 * (600 - (syn.level_shift_label_bars + syn.min_gap_bars))
    expected = 3 * syn.rate_per_1000_bars * candidates / 1000
    assert len(events) == pytest.approx(expected, rel=0.15)


def test_planning_is_seeded(planned):
    clean, eligible, blocked, syn, events = planned
    again = plan_events(clean, eligible, syn, np.random.default_rng(0), blocked=blocked)
    pd.testing.assert_frame_equal(events, again)
    other = plan_events(clean, eligible, syn, np.random.default_rng(1), blocked=blocked)
    assert not events["row"].equals(other["row"])


def _single(clean: pd.DataFrame, kind: str, row: int, shock: float, length: int = 1) -> pd.DataFrame:
    last = int(clean.index[clean["ticker"] == clean.at[row, "ticker"]][-1])
    return pd.DataFrame(
        [{"type": kind, "row": row, "shock": shock, "length": length, "ticker_last_row": last}]
    )


def test_price_spike_moves_one_close_and_snaps_back(business_days):
    clean = _clean_frame(business_days)
    out = inject(clean, _single(clean, "price_spike", 400, 0.2))
    changed = (out[["open", "high", "low", "close"]] != clean[["open", "high", "low", "close"]]).any(axis=1)
    assert changed[changed].index.tolist() == [400]
    assert np.log(out.at[400, "close"] / clean.at[400, "close"]) == pytest.approx(0.2)
    assert out.at[400, "high"] >= out.at[400, "close"]


def test_volume_burst_changes_only_volume(business_days):
    clean = _clean_frame(business_days)
    out = inject(clean, _single(clean, "volume_burst", 400, 8.0, length=3))
    np.testing.assert_allclose(out.loc[400:402, "volume"], clean.loc[400:402, "volume"] * 8)
    assert out.drop(index=[400, 401, 402])["volume"].equals(clean.drop(index=[400, 401, 402])["volume"])
    assert out["close"].equals(clean["close"])


def test_level_shift_is_permanent_within_one_ticker(business_days):
    clean = _clean_frame(business_days)
    out = inject(clean, _single(clean, "level_shift", 400, -0.1, length=5))
    aaa_end = int(clean.index[clean["ticker"] == "AAA"][-1])
    ratio = np.log(out["close"] / clean["close"])
    assert ratio.loc[:399].abs().max() == 0
    np.testing.assert_allclose(ratio.loc[400:aaa_end], -0.1)
    assert ratio.loc[aaa_end + 1 :].abs().max() == 0  # the next ticker is untouched


# ---------- real events ----------


def test_reaction_day_timing():
    cal = pd.bdate_range("2024-01-01", "2024-01-31")
    earnings = pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "D"],
            "announced": pd.to_datetime(
                [
                    "2024-01-09 16:00",  # Tuesday after close: Wednesday
                    "2024-01-09 07:00",  # Tuesday before open: same day
                    "2024-01-12 16:05",  # Friday after close: Monday
                    "2024-01-13 10:00",  # Saturday: Monday
                ]
            ).tz_localize("America/New_York"),
        }
    )
    out = reaction_days(earnings, cal, after_hour=12).set_index("ticker")["reaction_day"]
    assert out.to_dict() == {
        "A": pd.Timestamp("2024-01-10"),
        "B": pd.Timestamp("2024-01-09"),
        "C": pd.Timestamp("2024-01-15"),
        "D": pd.Timestamp("2024-01-15"),
    }


def test_real_event_tagging(cfg):
    days = pd.bdate_range("2024-01-01", periods=10)
    rets = [0.0, 0.01, 0.06, 0.0, 0.10, 0.0, 0.02, 0.0, -0.12, 0.0]
    feats = pd.DataFrame({"ticker": "A", "timestamp": days, "log_ret": rets, "vol_20": 0.01})
    reactions = pd.DataFrame({"ticker": ["A", "A"], "reaction_day": [days[2], days[6]]})
    events = tag_real_events(feats, reactions, cfg.labels.real)
    got = events.set_index("start")["type"].to_dict()
    # day 2: earnings, 6 sigma -> shock. day 6: earnings, only 2 sigma -> nothing.
    # day 4: no earnings, 10 sigma and >= 8% -> large move. day 8: -12 sigma -> large move.
    assert got == {days[2]: "earnings_shock", days[4]: "large_move", days[8]: "large_move"}
    assert list(events.columns) == EVENT_COLUMNS


def test_earnings_download_retries_and_reports_failures(cfg, tmp_path):
    calls = []

    def fake(ticker, limit):
        calls.append(ticker)
        if ticker == "FLAKY" and calls.count("FLAKY") == 1:
            raise RuntimeError("rate limited")
        if ticker == "NONE":
            return None
        idx = pd.DatetimeIndex(["2024-01-09 16:00"]).tz_localize("America/New_York")
        return pd.DataFrame({"EPS Estimate": [1.0]}, index=idx)

    dl = replace(cfg.download, backoff_seconds=0)
    status = download_earnings(["OK", "FLAKY", "NONE"], tmp_path, 10, dl, fetch=fake, sleep=lambda s: None)
    assert status == {"cached": [], "downloaded": ["OK", "FLAKY"], "failed": ["NONE"]}
    assert download_earnings(["OK"], tmp_path, 10, dl, fetch=fake)["cached"] == ["OK"]


# ---------- label columns ----------


def test_attach_labels_marks_events_and_grace(business_days):
    days = business_days[:20]
    feats = pd.concat([pd.DataFrame({"ticker": t, "timestamp": days, "x": 0.0}) for t in ("A", "B")], ignore_index=True)
    events = pd.DataFrame(
        [
            {"event_id": "syn-1", "track": "synthetic", "type": "volume_burst", "ticker": "A", "start": days[5], "end": days[6], "magnitude": 5.0},
            {"event_id": "syn-2", "track": "synthetic", "type": "price_spike", "ticker": "A", "start": days[18], "end": days[18], "magnitude": 5.0},
            {"event_id": "real-1", "track": "real", "type": "large_move", "ticker": "B", "start": days[10], "end": days[10], "magnitude": 9.0},
            {"event_id": "real-2", "track": "real", "type": "large_move", "ticker": "B", "start": pd.Timestamp("1999-01-01"), "end": pd.Timestamp("1999-01-01"), "magnitude": 9.0},
        ]
    )
    labeled, kept = attach_labels(feats, events, grace_bars=3)
    a = labeled[labeled["ticker"] == "A"].set_index("bar_idx")
    assert a.index[a["synthetic_label"]].tolist() == [5, 6, 18]
    assert a.index[a["synthetic_window"] == "syn-1"].tolist() == [5, 6, 7, 8, 9]
    assert a.index[a["synthetic_window"] == "syn-2"].tolist() == [18, 19]  # grace stops at the ticker's last bar
    b = labeled[labeled["ticker"] == "B"].set_index("bar_idx")
    assert b.index[b["real_label"]].tolist() == [10] and not b["synthetic_label"].any()
    assert sorted(kept["event_id"]) == ["real-1", "syn-1", "syn-2"]  # the unscorable event is dropped
    assert kept.set_index("event_id").loc["syn-1", "start_idx"] == 5

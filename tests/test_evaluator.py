"""The evaluator scores every model the same way, including the edge cases."""

from __future__ import annotations

import pandas as pd
import pytest

from anomaly.evaluation.evaluator import Evaluator, save_report
from anomaly.labels.attach import attach_labels


@pytest.fixture
def setup(cfg):
    """Two tickers, 40 days. The test split holds one synthetic event per ticker and one real event."""
    days = pd.bdate_range("2024-01-01", periods=40)
    feats = pd.concat([pd.DataFrame({"ticker": t, "timestamp": days}) for t in ("A", "B")], ignore_index=True)
    feats["split"] = ["val" if d < days[20] else "test" for d in feats["timestamp"]]
    ev = lambda i, track, t, s, e: {"event_id": i, "track": track, "type": "x", "ticker": t, "start": days[s], "end": days[e], "magnitude": 1.0}
    events = pd.DataFrame(
        [
            ev("syn-v", "synthetic", "A", 5, 5),
            ev("syn-a", "synthetic", "A", 25, 26),
            ev("syn-b", "synthetic", "B", 30, 30),
            ev("real-b", "real", "B", 22, 22),
        ]
    )
    labeled, kept = attach_labels(feats, events, grace_bars=cfg.evaluation.grace_bars)
    return Evaluator(labeled, kept, cfg.evaluation), labeled, days


def _scores(labeled, alarms: set[tuple[str, int]], value: float = 1.0) -> pd.DataFrame:
    s = labeled[["ticker", "timestamp", "bar_idx"]].copy()
    s["score"] = [value if (t, i) in alarms else 0.0 for t, i in zip(s["ticker"], s["bar_idx"])]
    return s


def test_perfect_scorer(setup):
    ev, labeled, _ = setup
    scores = _scores(labeled, {("A", 5), ("A", 25), ("A", 26), ("B", 30)})
    r = ev.evaluate(scores, threshold=0.5, split="test", track="synthetic")
    assert (r["precision"], r["recall"], r["event_recall"]) == (1.0, 1.0, 1.0)
    assert r["false_alarms"] == 0 and r["mean_delay_bars"] == 0 and r["pr_auc"] == pytest.approx(1.0)
    assert r["events"] == 2  # syn-v is in val, not test


def test_late_alarm_counts_as_detection_not_false_alarm(setup):
    ev, labeled, _ = setup
    scores = _scores(labeled, {("A", 28), ("B", 31)})  # 3 and 1 bars after the starts, inside grace
    r = ev.evaluate(scores, threshold=0.5, split="test", track="synthetic")
    assert r["event_recall"] == 1.0
    assert r["mean_delay_bars"] == pytest.approx(2.0)
    assert r["false_alarms"] == 0
    assert r["recall"] == 0.0  # point recall: the labeled bars themselves were missed


def test_other_tracks_events_are_not_false_alarms(setup):
    ev, labeled, _ = setup
    scores = _scores(labeled, {("B", 22), ("A", 35)})  # B@22 is a real event; A@35 is a true false alarm
    syn = ev.evaluate(scores, threshold=0.5, split="test", track="synthetic")
    assert syn["false_alarms"] == 1
    real = ev.evaluate(scores, threshold=0.5, split="test", track="real")
    assert real["event_recall"] == 1.0 and real["false_alarms"] == 1
    assert syn["false_alarms_per_day_all_tickers"] == pytest.approx(1 / 20)  # 1 false alarm over 20 test days
    assert syn["false_alarms_per_ticker_day"] == pytest.approx(1 / 40)  # 2 tickers x 20 days
    assert syn["false_alarms_per_ticker_month"] == pytest.approx(21 / 40)


def test_unscored_rows_count_as_no_alarm(setup):
    ev, labeled, _ = setup
    scores = _scores(labeled, {("A", 25), ("A", 26), ("B", 30)})
    partial = scores[~((scores["ticker"] == "B") & (scores["bar_idx"] >= 20))]  # model skipped ticker B's test rows
    r = ev.evaluate(partial, threshold=0.5, split="test", track="synthetic")
    assert r["coverage"] == pytest.approx(0.5)
    assert r["event_recall"] == 0.5


def test_duplicate_scores_are_rejected(setup):
    ev, labeled, _ = setup
    scores = _scores(labeled, set())
    with pytest.raises(ValueError, match="duplicate"):
        ev.evaluate(pd.concat([scores, scores.head(1)]), 0.5, "test", "synthetic")


def test_run_tunes_on_val_and_saves(setup, tmp_path):
    ev, labeled, _ = setup
    scores = _scores(labeled, {("A", 5), ("A", 25), ("A", 26), ("B", 30)}, value=0.9)
    report = ev.run("toy", scores)
    assert report["threshold"] == pytest.approx(0.9)  # best F1 on val alarms only on the val event
    assert report["results"]["test"]["synthetic"]["f1"] == 1.0
    save_report(report, tmp_path)
    save_report({**report, "model": "toy2"}, tmp_path)
    save_report(report, tmp_path)  # re-saving replaces rows rather than duplicating them
    summary = pd.read_csv(tmp_path / "metrics" / "summary.csv")
    assert sorted(summary["model"].unique()) == ["toy", "toy2"]
    assert len(summary) == 8  # 2 models x 2 splits x 2 tracks

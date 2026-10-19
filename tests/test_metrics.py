"""Metric functions checked against hand-calculated answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from anomaly.evaluation import metrics as m


def test_point_metrics_by_hand():
    y = np.array([1, 1, 0, 0, 0, 1], dtype=bool)
    alarm = np.array([1, 0, 1, 0, 0, 1], dtype=bool)
    out = m.point_metrics(y, alarm)
    assert (out["tp"], out["fp"], out["fn"]) == (2, 1, 1)
    assert out["precision"] == pytest.approx(2 / 3)
    assert out["recall"] == pytest.approx(2 / 3)
    assert out["f1"] == pytest.approx(2 / 3)


def test_point_metrics_with_no_alarms():
    out = m.point_metrics(np.array([1, 0], dtype=bool), np.array([0, 0], dtype=bool))
    assert out["precision"] == 0 and out["recall"] == 0 and out["f1"] == 0


def test_pr_auc_perfect_and_random():
    rng = np.random.default_rng(0)
    y = rng.random(50_000) < 0.02
    assert m.pr_auc(y, y.astype(float)) == pytest.approx(1.0)
    assert m.pr_auc(y, rng.random(len(y))) == pytest.approx(y.mean(), abs=0.005)
    assert np.isnan(m.pr_auc(np.zeros(5, bool), np.arange(5.0)))


def test_false_alarm_rates():
    out = m.false_alarm_rates(false_alarms=42, n_days=21, n_ticker_days=2100)
    assert out["false_alarms_per_day"] == 2.0
    assert out["false_alarms_per_ticker_month"] == pytest.approx(0.42)


def test_event_detection_and_delay():
    events = pd.DataFrame({"event_id": ["e1", "e2", "e3"], "start_idx": [10, 50, 90]})
    windows = pd.Series(["e1", "e1", pd.NA, "e3"], dtype="string")
    bar_idx = pd.Series([12, 11, 70, 90])
    out = m.event_metrics(events, windows, bar_idx).set_index("event_id")
    assert out["detected"].to_dict() == {"e1": True, "e2": False, "e3": True}
    assert out.loc["e1", "delay_bars"] == 1  # first alarm, not the later one
    assert out.loc["e3", "delay_bars"] == 0
    assert np.isnan(out.loc["e2", "delay_bars"])


def test_best_f1_threshold():
    y = np.array([0, 0, 0, 1, 1, 0, 1])
    score = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.85, 0.7])
    threshold, f1 = m.best_f1_threshold(y, score)
    alarm = score >= threshold
    assert m.point_metrics(y.astype(bool), alarm)["f1"] == pytest.approx(f1)
    assert f1 == pytest.approx(m.point_metrics(y.astype(bool), score >= 0.7)["f1"])


def test_fa_budget_threshold_respects_budget():
    rng = np.random.default_rng(0)
    negatives = rng.random(10_000)
    for per_day in (0.5, 2.0, 10.0):
        threshold = m.fa_budget_threshold(negatives, n_days=100, max_per_day=per_day)
        assert (negatives >= threshold).sum() <= per_day * 100
        assert (negatives >= np.nextafter(threshold, -np.inf) - 1e-9).sum() >= per_day * 100  # as low as allowed

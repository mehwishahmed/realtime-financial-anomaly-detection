"""Metric functions on plain arrays. The evaluator decides which rows to pass in."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve


def point_metrics(y: np.ndarray, alarm: np.ndarray) -> dict[str, float]:
    """Precision, recall, and F1, treating each bar as one prediction."""
    y = np.asarray(y, dtype=bool)
    alarm = np.asarray(alarm, dtype=bool)
    tp = int((y & alarm).sum())
    fp = int((~y & alarm).sum())
    fn = int((y & ~alarm).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Area under the precision-recall curve (average precision).

    A random scorer gets about the positive rate, not 0.5. That's why it suits
    rare anomalies better than ROC-AUC does.
    """
    y = np.asarray(y, dtype=bool)
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, np.asarray(score, dtype=float)))


def false_alarm_rates(false_alarms: int, n_days: int, n_ticker_days: int, days_per_month: float = 21.0) -> dict[str, float]:
    """False-alarm rates, both universe-wide and per ticker.

    - ``false_alarms_per_day_all_tickers``: total false alarms / trading days.
      This is the alert volume a team watching the whole universe would see,
      so it grows with the number of tickers.
    - ``false_alarms_per_ticker_day``: total false alarms / (tickers x trading days),
      counted as ticker-days (bars) actually in the split.
    - ``false_alarms_per_ticker_month``: per ticker-day x ``days_per_month`` (21 trading days).
      Doesn't depend on universe size, so it's the one to compare across runs.
    """
    per_ticker_day = false_alarms / n_ticker_days if n_ticker_days else float("nan")
    return {
        "false_alarms": int(false_alarms),
        "false_alarms_per_day_all_tickers": false_alarms / n_days if n_days else float("nan"),
        "false_alarms_per_ticker_day": per_ticker_day,
        "false_alarms_per_ticker_month": per_ticker_day * days_per_month,
    }


def event_metrics(
    events: pd.DataFrame, alarm_windows: pd.Series, alarm_bar_idx: pd.Series
) -> pd.DataFrame:
    """Was each event caught, and how many bars after it started?

    Args:
        events: one row per event with ``event_id`` and ``start_idx``.
        alarm_windows: event_id of the window each alarmed bar falls in (NA if none).
        alarm_bar_idx: ``bar_idx`` of each alarmed bar (same index as ``alarm_windows``).

    Returns:
        ``events`` plus ``detected`` (bool) and ``delay_bars`` (first alarm's
        bar_idx minus start_idx; NaN if missed).
    """
    hits = pd.DataFrame({"event_id": alarm_windows, "bar_idx": alarm_bar_idx}).dropna(subset=["event_id"])
    first = hits.groupby("event_id")["bar_idx"].min()
    out = events.copy()
    first_alarm = out["event_id"].map(first)
    out["detected"] = first_alarm.notna()
    out["delay_bars"] = first_alarm - out["start_idx"]
    return out


def best_f1_threshold(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """The score threshold (alarm when score >= threshold) that maximizes F1. Returns (threshold, f1)."""
    precision, recall, thresholds = precision_recall_curve(np.asarray(y, dtype=bool), np.asarray(score, dtype=float))
    f1 = 2 * precision * recall / np.where(precision + recall > 0, precision + recall, 1)
    best = int(np.argmax(f1[:-1])) if len(thresholds) else 0
    return float(thresholds[best]), float(f1[best])


def fa_budget_threshold(
    score_negatives: np.ndarray, n_days: int, max_per_day: float
) -> float:
    """The lowest threshold whose false alarms stay within ``max_per_day`` per trading day.

    ``score_negatives`` are the scores of bars that are not anomalies. The
    lowest allowed threshold catches as many anomalies as the budget allows.
    """
    allowed = int(np.floor(max_per_day * n_days))
    s = np.sort(np.asarray(score_negatives, dtype=float))[::-1]
    if allowed >= len(s):
        return float(s[-1]) if len(s) else 0.0
    # Alarm on score > s[allowed]: at most `allowed` negatives exceed it.
    return float(np.nextafter(s[allowed], np.inf))

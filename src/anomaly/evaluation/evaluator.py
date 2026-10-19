"""The single evaluation path: tune a threshold on validation, report on test, save results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from anomaly.config import EvaluationConfig
from anomaly.evaluation import metrics as m
from anomaly.labels.attach import TRACKS

KEY = ["ticker", "timestamp"]


class Evaluator:
    """Scores any model the same way.

    A model hands over a DataFrame with ``ticker``, ``timestamp``, and
    ``score`` (higher = more anomalous). Bars the model didn't score count as
    "no alarm", so skipping hard bars can't make a model look better.

    For each label track, these bars are excluded from point metrics:
      - bars in the *other* track's event windows (a real crash isn't a false
        alarm on the synthetic track, and vice versa)
      - grace bars just after an event, where an alarm is late but not false.
    """

    def __init__(self, labeled: pd.DataFrame, events: pd.DataFrame, cfg: EvaluationConfig):
        needed = {*KEY, "split", "bar_idx", *(f"{t}_{c}" for t in TRACKS for c in ("label", "window"))}
        missing = needed - set(labeled.columns)
        if missing:
            raise ValueError(f"Labeled table is missing columns: {sorted(missing)}")
        self.labels = labeled[sorted(needed)].reset_index(drop=True)
        self.events = events
        self.cfg = cfg

    # ----- helpers -----

    def _aligned_scores(self, scores: pd.DataFrame) -> np.ndarray:
        if scores.duplicated(KEY).any():
            raise ValueError("scores has duplicate (ticker, timestamp) rows")
        merged = self.labels[KEY].merge(scores[[*KEY, "score"]], on=KEY, how="left")
        return merged["score"].to_numpy(dtype=float)

    def _mask(self, split: str, track: str) -> tuple[np.ndarray, np.ndarray]:
        """(rows in split, rows that count for point metrics on this track)."""
        lab = self.labels
        other = "real" if track == "synthetic" else "synthetic"
        in_split = (lab["split"] == split).to_numpy()
        own_window = lab[f"{track}_window"].notna().to_numpy()
        other_window = lab[f"{other}_window"].notna().to_numpy()
        grace = own_window & ~lab[f"{track}_label"].to_numpy()
        counted = in_split & ~grace & ~(other_window & ~lab[f"{track}_label"].to_numpy())
        return in_split, counted

    # ----- public API -----

    def align(self, scores: pd.DataFrame) -> np.ndarray:
        """Scores as an array in the labeled table's row order (NaN where a row has no score)."""
        return self._aligned_scores(scores)

    def pr_auc_aligned(self, aligned: np.ndarray, split: str, track: str) -> float:
        """PR-AUC from an already aligned score array. A fast path for searching many score combinations."""
        _, counted = self._mask(split, track)
        y = self.labels[f"{track}_label"].to_numpy()[counted]
        s = aligned[counted]
        finite = np.isfinite(s)
        filled = np.where(finite, s, np.nanmin(s[finite]) - 1 if finite.any() else 0.0)
        return m.pr_auc(y, filled)

    def tune_threshold(self, scores: pd.DataFrame, split: str = "val", track: str | None = None) -> float:
        """Pick the alarm threshold on ``split`` using the configured objective."""
        track = track or self.cfg.tune_track
        s = self._aligned_scores(scores)
        in_split, counted = self._mask(split, track)
        rows = counted & ~np.isnan(s)
        y = self.labels[f"{track}_label"].to_numpy()[rows]
        if self.cfg.threshold_objective == "f1":
            threshold, _ = m.best_f1_threshold(y, s[rows])
            return threshold
        n_days = int(self.labels.loc[in_split, "timestamp"].nunique())
        return m.fa_budget_threshold(s[rows][~y], n_days, self.cfg.max_false_alarms_per_day_all_tickers)

    def evaluate(self, scores: pd.DataFrame, threshold: float, split: str, track: str) -> dict[str, Any]:
        """All metrics for one split and label track at a fixed threshold."""
        s = self._aligned_scores(scores)
        in_split, counted = self._mask(split, track)
        lab = self.labels
        scored = ~np.isnan(s)
        alarm = scored & (s >= threshold)
        y = lab[f"{track}_label"].to_numpy()

        filled = np.where(scored, s, np.nanmin(s[scored]) - 1 if scored.any() else 0.0)
        point = m.point_metrics(y[counted], alarm[counted])
        n_days = int(lab.loc[in_split, "timestamp"].nunique())
        fa = m.false_alarm_rates(point["fp"], n_days, n_ticker_days=int(in_split.sum()))

        events = self.events[self.events["track"] == track]
        split_of_event = (
            events[["ticker", "start"]]
            .rename(columns={"start": "timestamp"})
            .merge(lab[KEY + ["split"]], on=KEY, how="left")["split"]
            .to_numpy()
        )
        events = events[split_of_event == split]
        alarmed = in_split & alarm
        ev = m.event_metrics(events, lab.loc[alarmed, f"{track}_window"], lab.loc[alarmed, "bar_idx"])
        by_type = {
            kind: {
                "events": int(len(g)),
                "recall": float(g["detected"].mean()) if len(g) else float("nan"),
                "mean_delay_bars": float(g["delay_bars"].mean()) if g["detected"].any() else float("nan"),
            }
            for kind, g in ev.groupby("type")
        }
        detected = ev[ev["detected"]]
        return {
            "rows": int(in_split.sum()),
            "rows_counted": int(counted.sum()),
            "coverage": float(scored[in_split].mean()) if in_split.any() else float("nan"),
            "anomaly_rate": float(y[counted].mean()) if counted.any() else float("nan"),
            "pr_auc": m.pr_auc(y[counted], filled[counted]),
            **point,
            **fa,
            "events": int(len(ev)),
            "event_recall": float(ev["detected"].mean()) if len(ev) else float("nan"),
            "mean_delay_bars": float(detected["delay_bars"].mean()) if len(detected) else float("nan"),
            "median_delay_bars": float(detected["delay_bars"].median()) if len(detected) else float("nan"),
            "by_type": by_type,
        }

    def run(self, model: str, scores: pd.DataFrame) -> dict[str, Any]:
        """Tune on validation, then evaluate val and test on both tracks."""
        threshold = self.tune_threshold(scores)
        results = {
            split: {track: self.evaluate(scores, threshold, split, track) for track in TRACKS}
            for split in ("val", "test")
        }
        return {
            "model": model,
            "threshold": threshold,
            "threshold_objective": self.cfg.threshold_objective,
            "tuned_on": f"val/{self.cfg.tune_track}",
            "grace_bars": self.cfg.grace_bars,
            "results": results,
        }


SUMMARY_COLUMNS = [
    "model", "split", "track", "pr_auc", "precision", "recall", "f1", "event_recall",
    "mean_delay_bars", "false_alarms", "false_alarms_per_day_all_tickers", "false_alarms_per_ticker_day",
    "false_alarms_per_ticker_month", "events", "coverage", "threshold",
]


def save_report(report: dict[str, Any], results_dir: Path) -> Path:
    """Write ``<model>.json`` and upsert this model's rows in ``summary.csv``."""
    out_dir = results_dir / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{report['model']}.json"
    path.write_text(json.dumps(report, indent=2, default=float) + "\n")

    rows = [
        {"model": report["model"], "split": split, "track": track, "threshold": report["threshold"], **vals}
        for split, tracks in report["results"].items()
        for track, vals in tracks.items()
    ]
    new = pd.DataFrame(rows)[SUMMARY_COLUMNS]
    summary_path = out_dir / "summary.csv"
    if summary_path.exists():
        old = pd.read_csv(summary_path)
        new = pd.concat([old[old["model"] != report["model"]], new], ignore_index=True)
    new.round(6).to_csv(summary_path, index=False)
    return path

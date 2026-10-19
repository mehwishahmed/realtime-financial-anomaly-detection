"""Load the labeled dataset and save model scores. Every model script uses these."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from anomaly.config import Config
from anomaly.labels.attach import TRACKS


def load_labeled(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The labeled feature table and the events table the evaluator needs."""
    assert cfg.labels is not None
    labeled = pd.read_parquet(cfg.processed_dir() / f"{cfg.labels.frequency}_labeled.parquet")
    events = pd.concat(
        [pd.read_csv(cfg.paths.results_dir / "labels" / f"{t}_events.csv", parse_dates=["start", "end"]) for t in TRACKS],
        ignore_index=True,
    )
    return labeled, events


def scores_path(cfg: Config, model: str) -> Path:
    """Where a model's per-row scores live (used by the Phase 5 ensemble)."""
    return cfg.processed_dir() / "scores" / f"{model}.parquet"


def save_scores(cfg: Config, model: str, scores: pd.DataFrame) -> Path:
    path = scores_path(cfg, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    scores[["ticker", "timestamp", "score"]].to_parquet(path, index=False)
    return path

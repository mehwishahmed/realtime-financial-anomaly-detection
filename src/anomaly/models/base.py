"""The interface every model implements, so the evaluator, ensemble, and streaming code treat them all alike."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class AnomalyModel(ABC):
    """Fit on normal-looking history, then give every bar an anomaly score.

    Higher scores mean more anomalous. Scores only need to be comparable
    within a model; the evaluator picks the alarm threshold. A bar the
    model can't score (for example, missing inputs) gets NaN, which the
    evaluator treats as "no alarm".
    """

    name: str = "model"

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> "AnomalyModel":
        """Learn from the train split only."""

    @abstractmethod
    def score(self, df: pd.DataFrame) -> np.ndarray:
        """One score per row of ``df``, in the same order."""

    def score_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scores in the evaluator's format: ticker, timestamp, score."""
        return df[["ticker", "timestamp"]].assign(score=self.score(df)).reset_index(drop=True)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "AnomalyModel":
        return joblib.load(path)

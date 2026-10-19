"""Phase 3 baseline: Isolation Forest on per-bar features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from anomaly.models.base import AnomalyModel
from anomaly.models.preprocess import FeatureScaler


class IsolationForestModel(AnomalyModel):
    """Isolation Forest: anomalies are points that random splits isolate quickly.

    Each tree keeps cutting the data at random values of random features. A
    bar that ends up alone after only a few cuts is unusual. The anomaly
    score is the negated ``score_samples``, so shorter average paths give
    higher scores.

    It looks at one bar at a time (no sequence context), which makes it a
    good baseline for the sequence models in Phase 4.
    """

    name = "isolation_forest"

    def __init__(
        self,
        features: list[str],
        n_estimators: int = 100,
        max_samples: int | str = 256,
        max_features: float = 1.0,
        scaling: str = "per_ticker",
        min_train_rows_per_ticker: int = 250,
        seed: int = 42,
        n_jobs: int = -1,
    ):
        self.features = list(features)
        self.scaler = FeatureScaler(scaling, min_train_rows_per_ticker)  # type: ignore[arg-type]
        self.forest = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            random_state=seed,
            n_jobs=n_jobs,
        )
        self.train_rows = 0

    def fit(self, train: pd.DataFrame) -> "IsolationForestModel":
        """Fit the scaler and the forest on train rows that have every feature."""
        rows = train.dropna(subset=self.features)
        self.scaler.fit(rows, self.features)
        x = self.scaler.transform(rows)
        self.forest.fit(x)
        self.train_rows = len(rows)
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Negated ``score_samples`` (higher = more anomalous); NaN where an input is missing."""
        out = np.full(len(df), np.nan)
        ok = df[self.features].notna().all(axis=1).to_numpy()
        if ok.any():
            x = self.scaler.transform(df.loc[ok])
            out[ok] = -self.forest.score_samples(x)
        return out

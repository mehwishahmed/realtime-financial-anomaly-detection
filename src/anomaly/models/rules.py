"""The z-score rule as a model, so it can join the ensemble and the streaming replay."""

from __future__ import annotations

import numpy as np
import pandas as pd

from anomaly.evaluation.baselines import zscore_rule_scores
from anomaly.models.base import AnomalyModel


class ZScoreRuleModel(AnomalyModel):
    """Largest of |return| and |gap| (in units of yesterday's volatility) and |volume z-score|.

    It has nothing to learn, so ``fit`` does nothing. It needs the previous
    bar of each ticker, so pass enough history when scoring.
    """

    name = "baseline_zscore_rule"

    def __init__(self, vol_col: str = "vol_20"):
        self.vol_col = vol_col

    def fit(self, train: pd.DataFrame) -> "ZScoreRuleModel":
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        flat = df.reset_index(drop=True)
        order = flat.sort_values(["ticker", "timestamp"], kind="stable").index
        scores = zscore_rule_scores(flat.loc[order], vol_col=self.vol_col)["score"].to_numpy()
        out = np.empty(len(flat))
        out[order.to_numpy()] = scores
        return out

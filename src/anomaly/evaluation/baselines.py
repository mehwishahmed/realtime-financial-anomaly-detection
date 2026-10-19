"""Trivial scorers used to sanity-check the evaluator, not real models.

If the evaluator works, ``random`` should get a PR-AUC near the anomaly rate,
and a simple "how unusual is this bar" rule should do clearly better.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def random_scores(labeled: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Uniform random scores: the floor any real model has to beat."""
    rng = np.random.default_rng(seed)
    return labeled[["ticker", "timestamp"]].assign(score=rng.random(len(labeled)))


def zscore_rule_scores(labeled: pd.DataFrame, vol_col: str = "vol_20") -> pd.DataFrame:
    """Largest of |return|, |gap| (both in units of yesterday's volatility) and |volume z-score|.

    It uses yesterday's volatility, so a spike can't shrink its own score by
    inflating today's volatility. The first bar of each ticker gets no score.
    """
    prior_vol = labeled.groupby("ticker", sort=False)[vol_col].shift(1)
    score = pd.concat(
        [labeled["log_ret"].abs() / prior_vol, labeled["gap"].abs() / prior_vol, labeled["volume_z"].abs()], axis=1
    ).max(axis=1, skipna=False)
    return labeled[["ticker", "timestamp"]].assign(score=score.to_numpy())

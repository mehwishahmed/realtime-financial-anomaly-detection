"""Phase 5: combine member models' scores into one ensemble score.

Each member's raw scores live on its own scale (Isolation Forest path
lengths, reconstruction errors, sigma multiples), so they are first put on a
common scale using statistics from the **train** split only, then averaged with
weights tuned on val.
"""

from __future__ import annotations

import itertools
from typing import Literal

import numpy as np
import pandas as pd

from anomaly.models.base import AnomalyModel

Normalization = Literal["robust_z", "rank"]


class ScoreNormalizer:
    """Map one member's raw scores onto a common scale, using train-split scores only.

    ``robust_z``: (score - median) / (1.4826 x MAD). Unbounded, so a huge
        anomaly keeps a huge normalized score. The median and MAD barely move
        when a few training bars are extreme.
    ``rank``: the share of training scores below this score (0 to 1). Bounded,
        so no member can dominate, but every extreme score ends up near 1.0
        (it saturates, like Isolation Forest does).
    """

    def __init__(self, method: Normalization = "robust_z"):
        if method not in ("robust_z", "rank"):
            raise ValueError(f"Unknown normalization: {method}")
        self.method = method
        self.center = 0.0
        self.scale = 1.0
        self.reference: np.ndarray | None = None

    def fit(self, train_scores: np.ndarray) -> "ScoreNormalizer":
        s = np.asarray(train_scores, dtype=float)
        s = s[np.isfinite(s)]
        if len(s) == 0:
            raise ValueError("No finite training scores to fit the normalizer on")
        if self.method == "robust_z":
            self.center = float(np.median(s))
            mad = float(np.median(np.abs(s - self.center))) * 1.4826
            self.scale = mad if mad > 1e-12 else (float(s.std()) or 1.0)
        else:
            self.reference = np.sort(s)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        s = np.asarray(scores, dtype=float)
        if self.method == "robust_z":
            return (s - self.center) / self.scale
        assert self.reference is not None, "transform called before fit"
        out = np.searchsorted(self.reference, s, side="right") / len(self.reference)
        return np.where(np.isfinite(s), out, np.nan)


def weight_grid(n_members: int, step: float) -> list[tuple[float, ...]]:
    """Every weight vector with entries in multiples of ``step`` that sums to 1."""
    units = int(round(1 / step))
    out = []
    for combo in itertools.product(range(units + 1), repeat=n_members):
        if sum(combo) == units:
            out.append(tuple(round(c / units, 10) for c in combo))
    return out


def combine(normalized: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted average across members (columns), ignoring members that couldn't score a row.

    Weights are rescaled over the members that have a score, so a bar the deep
    models can't score yet (not enough history) still gets an ensemble score.
    A row where no positively weighted member has a score gets NaN.
    """
    w = np.asarray(weights, dtype=float)
    have = np.isfinite(normalized) & (w > 0)
    total = (have * w).sum(axis=1)
    weighted = np.where(have, normalized * w, 0.0).sum(axis=1)
    return np.where(total > 0, weighted / np.where(total > 0, total, 1.0), np.nan)


class EnsembleModel(AnomalyModel):
    """A weighted average of already-trained member models' normalized scores.

    ``fit`` doesn't retrain the members. It only fits each member's
    normalizer on that member's scores for the training rows.
    """

    name = "ensemble"

    def __init__(self, members: dict[str, AnomalyModel], weights: dict[str, float], normalization: Normalization = "robust_z"):
        if set(weights) != set(members):
            raise ValueError("weights must name exactly the ensemble members")
        self.members = members
        self.weights = weights
        self.normalization = normalization
        self.normalizers: dict[str, ScoreNormalizer] = {}

    @property
    def order(self) -> list[str]:
        return list(self.members)

    def fit(self, train: pd.DataFrame) -> "EnsembleModel":
        return self.fit_from_scores({n: self.members[n].score(train) for n in self.order})

    def fit_from_scores(self, train_scores: dict[str, np.ndarray]) -> "EnsembleModel":
        """Fit normalizers from members' train-split scores that were computed elsewhere."""
        self.normalizers = {n: ScoreNormalizer(self.normalization).fit(train_scores[n]) for n in self.order}
        return self

    def member_scores(self, df: pd.DataFrame) -> np.ndarray:
        """Raw scores, one column per member (in ``order``)."""
        return np.column_stack([self.members[n].score(df) for n in self.order])

    def combine_raw(self, raw: np.ndarray) -> np.ndarray:
        """Normalize raw member scores (columns in ``order``) and take the weighted average."""
        if not self.normalizers:
            raise RuntimeError("EnsembleModel used before fit")
        normalized = np.column_stack([self.normalizers[n].transform(raw[:, i]) for i, n in enumerate(self.order)])
        return combine(normalized, np.array([self.weights[n] for n in self.order]))

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Ensemble score per row. Pass enough history for every member (20 bars for the deep models)."""
        return self.combine_raw(self.member_scores(df))

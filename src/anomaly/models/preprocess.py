"""Model inputs: derived features and a scaler whose statistics come from the train split only."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def add_derived_features(df: pd.DataFrame, vol_col: str = "vol_20") -> pd.DataFrame:
    """Add ``ret_z`` and ``gap_z``: return and gap divided by the *previous* bar's volatility.

    These say "how many sigmas" a move is, relative to the stock's own recent
    behaviour. Using the previous bar's volatility stops a spike from shrinking
    its own z-score. The first bar of each ticker has no previous volatility, so it gets NaN.
    """
    out = df.sort_values(["ticker", "timestamp"]).copy()
    prior_vol = out.groupby("ticker", sort=False)[vol_col].shift(1)
    prior_vol = prior_vol.where(prior_vol > 0)
    out["ret_z"] = (out["log_ret"] / prior_vol).astype("float32")
    out["gap_z"] = (out["gap"] / prior_vol).astype("float32")
    return out.loc[df.index]  # back to the caller's row order


class FeatureScaler:
    """Standardize features to mean 0 and std 1, using train-split statistics.

    ``per_ticker`` uses each ticker's own mean and std, so a volatile stock's
    ordinary day doesn't look extreme next to a calm stock's. Tickers with
    fewer than ``min_rows`` training rows (recent listings) fall back to the
    statistics of all tickers combined.
    """

    def __init__(self, mode: Literal["per_ticker", "global"] = "per_ticker", min_rows: int = 250):
        self.mode = mode
        self.min_rows = min_rows
        self.columns: list[str] = []
        self.global_mean: pd.Series | None = None
        self.global_std: pd.Series | None = None
        self.ticker_mean: pd.DataFrame | None = None
        self.ticker_std: pd.DataFrame | None = None

    @staticmethod
    def _safe_std(std: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
        return std.where(std > 1e-12, 1.0)

    def fit(self, train: pd.DataFrame, columns: list[str]) -> "FeatureScaler":
        self.columns = list(columns)
        data = train[["ticker", *self.columns]].dropna()
        self.global_mean = data[self.columns].mean()
        self.global_std = self._safe_std(data[self.columns].std())
        if self.mode == "per_ticker":
            grouped = data.groupby("ticker")[self.columns]
            counts = grouped.size()
            enough = counts[counts >= self.min_rows].index
            self.ticker_mean = grouped.mean().loc[enough]
            self.ticker_std = self._safe_std(grouped.std().loc[enough])
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Scaled feature matrix (float32), one row per row of ``df``. NaN inputs stay NaN."""
        if self.global_mean is None:
            raise RuntimeError("FeatureScaler.transform called before fit")
        x = df[self.columns].to_numpy(dtype="float64")
        mean = np.broadcast_to(self.global_mean.to_numpy(), x.shape).copy()
        std = np.broadcast_to(self.global_std.to_numpy(), x.shape).copy()
        if self.mode == "per_ticker" and self.ticker_mean is not None and len(self.ticker_mean):
            t_mean = self.ticker_mean.reindex(df["ticker"]).to_numpy()  # NaN rows = tickers without own stats
            t_std = self.ticker_std.reindex(df["ticker"]).to_numpy()
            known = ~np.isnan(t_mean).any(axis=1)
            mean[known], std[known] = t_mean[known], t_std[known]
        return ((x - mean) / std).astype("float32")

    def fallback_tickers(self, df: pd.DataFrame) -> list[str]:
        """Tickers in ``df`` that are scaled with global rather than their own statistics."""
        if self.mode != "per_ticker" or self.ticker_mean is None:
            return sorted(df["ticker"].unique())
        return sorted(set(df["ticker"]) - set(self.ticker_mean.index))

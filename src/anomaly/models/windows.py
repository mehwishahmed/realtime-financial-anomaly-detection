"""Turn the per-bar feature table into sliding windows for the sequence models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from anomaly.models.preprocess import FeatureScaler


def make_windows(
    df: pd.DataFrame,
    features: list[str],
    scaler: FeatureScaler,
    length: int,
    end_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one window per bar that has ``length - 1`` earlier bars of the same ticker.

    Each window holds the scaled features of bars ``t - length + 1 .. t``, so
    the model only ever sees the present and the past of the bar it scores.

    Returns:
        X: float32 array of shape (n_windows, length, n_features).
        rows: for each window, the position (``iloc``) in ``df`` of its last bar,
            which is the bar that window scores.

    Windows that contain a missing feature value are skipped, so those bars get no score.

    ``end_mask`` (one bool per row of ``df``) limits which bars get a window,
    so memory grows with the number of windows you need rather than the size of ``df``.
    """
    if not {"ticker", "timestamp", *features} <= set(df.columns):
        raise ValueError("df needs ticker, timestamp, and the feature columns")
    flat = df.reset_index(drop=True)
    order = flat.sort_values(["ticker", "timestamp"], kind="stable").index.to_numpy()
    z = scaler.transform(flat.iloc[order])
    tickers = flat["ticker"].to_numpy()[order]
    starts = np.flatnonzero(np.r_[True, tickers[1:] != tickers[:-1]])
    stops = np.r_[starts[1:], len(order)]

    xs, rows = [], []
    for a, b in zip(starts, stops):
        if b - a < length:
            continue
        end_rows = order[a + length - 1 : b]
        if end_mask is None:
            win = sliding_window_view(z[a:b], (length, z.shape[1]))[:, 0]
        else:
            keep = np.asarray(end_mask)[end_rows]
            if not keep.any():
                continue
            local_ends = np.flatnonzero(keep) + length - 1
            win = z[a:b][local_ends[:, None] + np.arange(-length + 1, 1)]
            end_rows = end_rows[keep]
        ok = ~np.isnan(win).any(axis=(1, 2))
        xs.append(win[ok])
        rows.append(end_rows[ok])
    if not xs:
        return np.empty((0, length, len(features)), dtype="float32"), np.empty(0, dtype=int)
    return np.ascontiguousarray(np.concatenate(xs), dtype="float32"), np.concatenate(rows)


def window_end_candidates(df: pd.DataFrame, length: int) -> np.ndarray:
    """Positions (iloc) of rows that have at least ``length - 1`` earlier rows of the same ticker."""
    flat = df.reset_index(drop=True)
    rank = flat.sort_values(["ticker", "timestamp"], kind="stable").groupby("ticker", sort=False).cumcount()
    return np.sort(rank[rank >= length - 1].index.to_numpy())

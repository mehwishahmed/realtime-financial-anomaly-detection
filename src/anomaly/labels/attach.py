"""Turn event tables into per-row label columns that the evaluator reads."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRACKS = ("synthetic", "real")


def attach_labels(features: pd.DataFrame, events: pd.DataFrame, grace_bars: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add label columns for each track to the feature table.

    For each track ``T`` in (synthetic, real):
        ``T_label``  True on bars inside an event (start..end).
        ``T_window`` event_id on bars from start to ``grace_bars`` after end.
                     Alarms here count toward catching the event and are never false alarms.

    Also adds ``bar_idx`` (bar number within each ticker), which detection
    delay is measured in.

    Returns (labeled table, events that landed on a scorable row, with
    ``start_idx`` added). An event whose start bar isn't in the feature table
    (dropped during feature warm-up) is left out of the returned events.
    """
    df = features.sort_values(["ticker", "timestamp"], ignore_index=True).copy()
    df["bar_idx"] = df.groupby("ticker", sort=False).cumcount().astype("int32")
    key = pd.MultiIndex.from_frame(df[["ticker", "timestamp"]])
    first_row = df.groupby("ticker", sort=False).indices
    kept = []
    for track in TRACKS:
        label = np.zeros(len(df), dtype=bool)
        window = np.full(len(df), None, dtype=object)
        track_events = events[events["track"] == track]
        for ev in track_events.itertuples(index=False):
            start = key.get_indexer([(ev.ticker, ev.start)])[0]
            end = key.get_indexer([(ev.ticker, ev.end)])[0]
            if start < 0:
                continue
            rows = first_row[ev.ticker]
            last = rows[-1]
            end = end if end >= 0 else min(start, last)
            label[start : end + 1] = True
            window[start : min(end + grace_bars, last) + 1] = ev.event_id
            kept.append({**ev._asdict(), "start_idx": int(df.at[start, "bar_idx"])})
        df[f"{track}_label"] = label
        df[f"{track}_window"] = pd.array(window, dtype="string")
    kept_events = pd.DataFrame(kept, columns=[*events.columns, "start_idx"])
    return df, kept_events

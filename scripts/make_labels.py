"""Build the labeled dataset: tag real events, inject synthetic anomalies, recompute features.

    python scripts/make_labels.py --config configs/default.yaml

Outputs:
    data/processed/<freq>_labeled.parquet     features + split + label columns (not committed)
    data/processed/<freq>_bars_injected.parquet   cleaned bars (incl. market ticker) after injection;
                                                  the streaming replay starts from these (not committed)
    results/labels/synthetic_events.csv       every injected anomaly: where, what type, how big
    results/labels/real_events.csv            every tagged real event
    results/labels/summary.json               event counts and anomaly rates per split
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from anomaly.config import load_config
from anomaly.data.clean import clean_universe
from anomaly.data.download import load_cached
from anomaly.data.features import build_features
from anomaly.data.splits import apply_boundaries, assign_splits
from anomaly.data.universe import select_tickers
from anomaly.labels import EVENT_COLUMNS
from anomaly.labels.attach import TRACKS, attach_labels
from anomaly.labels.events import download_earnings, load_earnings, reaction_days, tag_real_events
from anomaly.labels.synthetic import inject, plan_events
from anomaly.utils import get_logger, set_seed, timer, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    assert cfg.labels and cfg.evaluation and cfg.cleaning and cfg.features
    set_seed(cfg.seed)
    log = get_logger()
    freq = cfg.frequencies[cfg.labels.frequency]
    market = cfg.universe.market_ticker
    tickers = select_tickers(cfg)
    stocks = [t for t in tickers if t != market]
    vol_col = f"vol_{cfg.features.vol_windows[0]}"

    with timer() as t:
        clean = clean_universe(load_cached(tickers, cfg.raw_dir(freq.name)), market, freq, cfg.cleaning).data
        original = build_features(clean, market, freq, cfg.features)
        original, bounds = assign_splits(original, freq.splits)

        # Real events, found on the untouched data. Earnings dates are only needed for tickers that passed cleaning.
        stocks = sorted(set(clean["ticker"]) - {market})
        log.info("Fetching earnings dates for %d tickers", len(stocks))
        earnings_dir = cfg.earnings_dir()
        earn_status = download_earnings(stocks, earnings_dir, cfg.labels.real.earnings_limit, cfg.download, logger=log)
        calendar = pd.DatetimeIndex(clean.loc[clean["ticker"] == market, "timestamp"])
        reactions = reaction_days(load_earnings(stocks, earnings_dir), calendar, cfg.labels.real.earnings_after_hour)
        real = tag_real_events(original, reactions, cfg.labels.real, vol_col=vol_col)

        # Synthetic events, injected into val/test prices away from real events.
        clean = apply_boundaries(clean.reset_index(drop=True), bounds)
        eligible = clean["split"].isin(cfg.labels.inject_splits) & (clean["ticker"] != market)
        real_keys = pd.MultiIndex.from_frame(real[["ticker", "start"]])
        blocked = pd.Series(pd.MultiIndex.from_frame(clean[["ticker", "timestamp"]]).isin(real_keys), index=clean.index)
        rng = np.random.default_rng(cfg.seed)
        planned = plan_events(clean, eligible, cfg.labels.synthetic, rng, blocked=blocked)
        injected = inject(clean.drop(columns="split"), planned)
        injected.to_parquet(cfg.processed_dir() / f"{freq.name}_bars_injected.parquet", index=False)

        features = build_features(injected, market, freq, cfg.features)
        features = apply_boundaries(features, bounds)
        all_events = pd.concat([planned[EVENT_COLUMNS], real], ignore_index=True)
        labeled, kept = attach_labels(features, all_events, cfg.evaluation.grace_bars)

        out_path = cfg.processed_dir() / f"{freq.name}_labeled.parquet"
        labeled.to_parquet(out_path, index=False)

    out_dir = cfg.paths.results_dir / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    kept = kept.merge(labeled[["ticker", "timestamp", "split"]], left_on=["ticker", "start"], right_on=["ticker", "timestamp"])
    kept = kept.drop(columns="timestamp")
    for track in TRACKS:
        kept[kept["track"] == track].to_csv(out_dir / f"{track}_events.csv", index=False)

    counts = {
        track: {
            split: kept[(kept["track"] == track) & (kept["split"] == split)]["type"].value_counts().sort_index().to_dict()
            for split in ("train", "val", "test")
        }
        for track in TRACKS
    }
    rates = {
        track: {
            split: round(float(labeled.loc[labeled["split"] == split, f"{track}_label"].mean()), 6)
            for split in ("train", "val", "test")
        }
        for track in TRACKS
    }
    summary = {
        "frequency": freq.name,
        "seed": cfg.seed,
        "tickers": len(stocks),
        "earnings_dates": {
            "tickers_with_dates": len(earn_status["cached"]) + len(earn_status["downloaded"]),
            "tickers_without_dates": sorted(earn_status["failed"]),
            "reaction_days_in_range": len(reactions),
        },
        "events_planned": {"synthetic": len(planned), "real": len(real)},
        "events_kept": {"synthetic": int((kept["track"] == "synthetic").sum()), "real": int((kept["track"] == "real").sum())},
        "events_by_split": counts,
        "anomalous_bar_rate": rates,
        "rows": len(labeled),
        "seconds": round(t["seconds"], 1),
    }
    write_json(summary, out_dir / "summary.json")
    log.info("Events kept: %s (planned %s)", summary["events_kept"], summary["events_planned"])
    for track in TRACKS:
        log.info("  %s by split: %s", track, counts[track])
    log.info("  anomalous bar rate: %s", rates)
    log.info("  earnings: %s, %.1fs", summary["earnings_dates"]["tickers_with_dates"], t["seconds"])
    log.info("Saved %s", out_path.relative_to(cfg.root))


if __name__ == "__main__":
    main()

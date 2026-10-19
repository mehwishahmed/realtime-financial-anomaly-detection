"""Clean cached bars, build features, assign time-based splits, and save everything.

    python scripts/build_features.py --config configs/default.yaml --freq all

Outputs:
    data/processed/<freq>_features.parquet   model-ready rows (not committed)
    results/data/<freq>_quality.csv          per-ticker cleaning report
    results/data/<freq>_summary.json         asset counts, row counts, split dates, runtime
"""

from __future__ import annotations

import argparse

from anomaly.config import load_config
from anomaly.data.clean import clean_universe
from anomaly.data.download import load_cached
from anomaly.data.features import build_features, feature_columns
from anomaly.data.splits import SPLITS, assign_splits
from anomaly.data.universe import select_tickers
from anomaly.utils import get_logger, set_seed, timer, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--freq", default="all", help="daily, intraday, or all enabled frequencies")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    log = get_logger()
    tickers = select_tickers(cfg)
    market = cfg.universe.market_ticker
    assert cfg.cleaning is not None and cfg.features is not None
    cols = feature_columns(cfg.features)

    freqs = [f for f in cfg.frequencies.values() if f.enabled and args.freq in ("all", f.name)]
    for freq in freqs:
        with timer() as t:
            raw = load_cached(tickers, cfg.raw_dir(freq.name))
            cleaned = clean_universe(raw, market, freq, cfg.cleaning)
            feats = build_features(cleaned.data, market, freq, cfg.features)
            feats, bounds = assign_splits(feats, freq.splits)
            out_path = cfg.processed_dir() / f"{freq.name}_features.parquet"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            feats.to_parquet(out_path, index=False)

        report = cleaned.report
        stocks = report[report["ticker"] != market]
        rows_by_split = feats["split"].value_counts().reindex(SPLITS).fillna(0).astype(int).to_dict()
        ranges = {
            s: [str(g["timestamp"].min()), str(g["timestamp"].max())]
            for s, g in feats.groupby("split", observed=True)
        }
        summary = {
            "frequency": freq.name,
            "interval": freq.interval,
            "universe_file": cfg.universe.file.name,
            "seed": cfg.seed,
            "assets": {
                "selected": len(tickers) - 1,
                "with_cached_data": len([t for t in raw if t != market]),
                "passed_cleaning": int((stocks["status"] == "ok").sum()),
                "in_feature_table": int(feats["ticker"].nunique()),
            },
            "rejected_by_reason": stocks.loc[stocks["status"] != "ok", "reason"].value_counts().to_dict(),
            "rows": {"total": len(feats), **rows_by_split},
            "filled_bars_in_clean_data": int(cleaned.data["is_filled"].sum()),
            "split_boundaries": {k: str(v.date()) for k, v in bounds.items()},
            "split_ranges": ranges,
            "features": cols,
            "seconds": round(t["seconds"], 1),
        }
        results_dir = cfg.paths.results_dir / "data"
        results_dir.mkdir(parents=True, exist_ok=True)
        report.to_csv(results_dir / f"{freq.name}_quality.csv", index=False)
        write_json(summary, results_dir / f"{freq.name}_summary.json")
        log.info("%s: %s", freq.name, summary["assets"])
        log.info("  rows %s, rejected %s, %.1fs", summary["rows"], summary["rejected_by_reason"], t["seconds"])
        log.info("  saved %s", out_path.relative_to(cfg.root))


if __name__ == "__main__":
    main()

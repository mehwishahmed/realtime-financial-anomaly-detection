"""Download and cache daily and/or intraday bars for the configured universe.

    python scripts/download_data.py --config configs/default.yaml --freq all
"""

from __future__ import annotations

import argparse

from anomaly.config import load_config
from anomaly.data.download import download_bars
from anomaly.data.universe import select_tickers
from anomaly.utils import get_logger, timer, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--freq", default="all", help="daily, intraday, or all enabled frequencies")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = get_logger()
    tickers = select_tickers(cfg)
    log.info("Universe %s: %d stocks + %s", cfg.universe.file.name, len(tickers) - 1, cfg.universe.market_ticker)

    freqs = [f for f in cfg.frequencies.values() if f.enabled and args.freq in ("all", f.name)]
    for freq in freqs:
        with timer() as t:
            report = download_bars(tickers, freq, cfg.download, cfg.raw_dir(freq.name), logger=log)
        summary = {
            "frequency": freq.name,
            "interval": freq.interval,
            "universe_file": cfg.universe.file.name,
            **report.summary(),
            "failed_tickers": sorted(report.failed),
            "seconds": round(t["seconds"], 1),
        }
        write_json(summary, cfg.paths.results_dir / "data" / f"{freq.name}_download.json")
        log.info("%s download: %s in %.1fs", freq.name, report.summary(), t["seconds"])
        if report.failed:
            log.info("  failed: %s", ", ".join(sorted(report.failed)))


if __name__ == "__main__":
    main()

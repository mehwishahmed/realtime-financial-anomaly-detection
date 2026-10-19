"""Fetch current ticker lists and save them to configs/tickers/.

    python scripts/build_universe.py --which both
"""

from __future__ import annotations

import argparse

from anomaly.config import load_config
from anomaly.data.universe import fetch_sp500, fetch_us_listed
from anomaly.utils import get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--which", choices=["sp500", "us_all", "both"], default="both")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = get_logger()
    out_dir = cfg.root / "configs" / "tickers"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.which in ("sp500", "both"):
        sp500 = fetch_sp500()
        sp500.to_csv(out_dir / "sp500.csv", index=False)
        log.info("sp500.csv: %d tickers", len(sp500))
    if args.which in ("us_all", "both"):
        us_all = fetch_us_listed()
        us_all.to_csv(out_dir / "us_all.csv", index=False)
        log.info("us_all.csv: %d tickers (%s)", len(us_all), us_all["exchange"].value_counts().to_dict())


if __name__ == "__main__":
    main()

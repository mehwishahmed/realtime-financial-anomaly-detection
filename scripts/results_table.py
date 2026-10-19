"""Regenerate the README results table from results/metrics/summary.csv.

    python scripts/results_table.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse

from anomaly.config import load_config
from anomaly.reporting import update_readme


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    print(update_readme(cfg.root / "README.md", cfg.paths.results_dir / "metrics" / "summary.csv"))


if __name__ == "__main__":
    main()

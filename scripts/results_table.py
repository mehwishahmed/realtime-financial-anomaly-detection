"""Regenerate the README results table (results/metrics/summary.csv) and scaling table (results/scaling/scaling_table.csv).

    python scripts/results_table.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse

from anomaly.config import load_config
from anomaly.reporting import update_readme, update_readme_scaling


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    print(update_readme(cfg.root / "README.md", cfg.paths.results_dir / "metrics" / "summary.csv"))
    scaling = update_readme_scaling(cfg.root / "README.md", cfg.root / "results" / "scaling" / "scaling_table.csv")
    if scaling:
        print(scaling)


if __name__ == "__main__":
    main()

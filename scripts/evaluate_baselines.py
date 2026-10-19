"""Run the sanity-check scorers (random, z-score rule) through the evaluator.

    python scripts/evaluate_baselines.py --config configs/default.yaml

Writes results/metrics/<model>.json and updates results/metrics/summary.csv.
"""

from __future__ import annotations

import argparse

from anomaly.config import load_config
from anomaly.evaluation.baselines import random_scores, zscore_rule_scores
from anomaly.evaluation.evaluator import Evaluator, save_report
from anomaly.evaluation.io import load_labeled
from anomaly.labels.attach import TRACKS
from anomaly.utils import get_logger, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    assert cfg.evaluation and cfg.features
    set_seed(cfg.seed)
    log = get_logger()
    labeled, events = load_labeled(cfg)
    evaluator = Evaluator(labeled, events, cfg.evaluation)

    scorers = {
        "baseline_random": random_scores(labeled, cfg.seed),
        "baseline_zscore_rule": zscore_rule_scores(labeled, vol_col=f"vol_{cfg.features.vol_windows[0]}"),
    }
    for name, scores in scorers.items():
        report = evaluator.run(name, scores)
        path = save_report(report, cfg.paths.results_dir)
        log.info("%s (threshold %.4f) -> %s", name, report["threshold"], path.relative_to(cfg.root))
        for split in ("val", "test"):
            for track in TRACKS:
                r = report["results"][split][track]
                log.info(
                    "  %-4s %-9s PR-AUC %.3f  P %.3f  R %.3f  F1 %.3f  event recall %.3f  delay %.2f  "
                    "FA/day (all tickers) %.2f  FA/ticker/month %.3f",
                    split, track, r["pr_auc"], r["precision"], r["recall"], r["f1"],
                    r["event_recall"], r["mean_delay_bars"], r["false_alarms_per_day_all_tickers"],
                    r["false_alarms_per_ticker_month"],
                )


if __name__ == "__main__":
    main()

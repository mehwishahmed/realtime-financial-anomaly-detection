"""Phase 3: train the Isolation Forest baseline, tune it on val, report it on test.

    python scripts/train_iforest.py --config configs/default.yaml

Steps:
    1. Fit one model per grid combination on the train split.
    2. Pick the combination with the best val PR-AUC (test is never used for choosing).
    3. Score every row with the chosen model and run it through the shared evaluator.

Outputs:
    results/models/isolation_forest_grid.csv     val PR-AUC and timings for every combination
    results/models/isolation_forest.json         chosen settings, training size, timings
    results/metrics/isolation_forest.json         evaluator report (also added to summary.csv)
    data/processed/scores/isolation_forest.parquet   per-row scores for the ensemble (not committed)
    models/isolation_forest.joblib                fitted model (not committed)
"""

from __future__ import annotations

import argparse
import itertools

import pandas as pd

from anomaly.config import load_config
from anomaly.evaluation.evaluator import Evaluator, save_report
from anomaly.evaluation.io import load_labeled, save_scores
from anomaly.models.isolation_forest import IsolationForestModel
from anomaly.models.preprocess import add_derived_features
from anomaly.utils import get_logger, set_seed, timer, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    assert cfg.models and cfg.evaluation and cfg.features
    set_seed(cfg.seed)
    log = get_logger()
    mcfg, icfg = cfg.models, cfg.models.isolation_forest
    track = cfg.evaluation.tune_track

    labeled, events = load_labeled(cfg)
    labeled = add_derived_features(labeled, vol_col=f"vol_{cfg.features.vol_windows[0]}")
    evaluator = Evaluator(labeled, events, cfg.evaluation)
    train = labeled[labeled["split"] == "train"]
    val = labeled[labeled["split"] == "val"]
    log.info("Train rows %d, val rows %d", len(train), len(val))

    def build(feature_set: str, n_estimators: int, max_samples: int) -> IsolationForestModel:
        """An unfitted Isolation Forest for one grid combination."""
        return IsolationForestModel(
            features=list(mcfg.feature_sets[feature_set]),
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=icfg.max_features,
            scaling=mcfg.scaling,
            min_train_rows_per_ticker=mcfg.min_train_rows_per_ticker,
            seed=cfg.seed,
            n_jobs=icfg.n_jobs,
        )

    rows = []
    keys = list(icfg.grid)
    for combo in itertools.product(*icfg.grid.values()):
        params = dict(zip(keys, combo))
        model = build(**params)
        with timer() as fit_t:
            model.fit(train)
        with timer() as score_t:
            val_scores = model.score_frame(val)
        # PR-AUC doesn't depend on the threshold, so any value works here.
        val_syn = evaluator.evaluate(val_scores, threshold=float("inf"), split="val", track="synthetic")
        val_real = evaluator.evaluate(val_scores, threshold=float("inf"), split="val", track="real")
        rows.append(
            {
                **params,
                "val_pr_auc_synthetic": round(val_syn["pr_auc"], 6),
                "val_pr_auc_real": round(val_real["pr_auc"], 6),
                "fit_seconds": round(fit_t["seconds"], 2),
                "score_seconds_val": round(score_t["seconds"], 2),
            }
        )
        log.info("  %s  val PR-AUC synthetic %.4f  real %.4f  (fit %.1fs)", params, val_syn["pr_auc"], val_real["pr_auc"], fit_t["seconds"])

    grid = pd.DataFrame(rows).sort_values(f"val_pr_auc_{track}", ascending=False, ignore_index=True)
    out_dir = cfg.paths.results_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(out_dir / "isolation_forest_grid.csv", index=False)
    best = {k: grid.loc[0, k] for k in keys}
    best = {k: (int(v) if k != "feature_set" else str(v)) for k, v in best.items()}
    log.info("Best on val/%s: %s", track, best)

    model = build(**best)
    with timer() as fit_t:
        model.fit(train)
    with timer() as score_t:
        scores = model.score_frame(labeled)
    save_scores(cfg, model.name, scores)
    model.save(cfg.paths.models_dir / f"{model.name}.joblib")

    report = evaluator.run(model.name, scores)
    save_report(report, cfg.paths.results_dir)
    write_json(
        {
            "model": model.name,
            "chosen_by": f"best val PR-AUC on the {track} track",
            "params": best,
            "features": model.features,
            "scaling": mcfg.scaling,
            "train_rows": model.train_rows,
            "tickers_using_global_scaling": model.scaler.fallback_tickers(labeled),
            "fit_seconds": round(fit_t["seconds"], 2),
            "score_seconds_all_rows": round(score_t["seconds"], 2),
            "rows_scored": int(scores["score"].notna().sum()),
            "seed": cfg.seed,
        },
        out_dir / f"{model.name}.json",
    )

    summary = pd.read_csv(cfg.paths.results_dir / "metrics" / "summary.csv")
    test = summary[summary["split"] == "test"]
    cols = ["model", "track", "pr_auc", "precision", "recall", "f1", "event_recall", "mean_delay_bars",
            "false_alarms_per_day_all_tickers", "false_alarms_per_ticker_month"]
    log.info("Test results, all models so far:\n%s", test[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

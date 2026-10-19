"""Phase 4: train the LSTM autoencoder and the Transformer forecaster, then evaluate them.

    python scripts/train_deep.py --config configs/default.yaml --model all

Each model trains on the train split only, early-stops on the last 10% of the
train dates, and goes through the same evaluator as every other model
(threshold tuned on val, results reported on test).

Outputs (per model):
    results/models/<model>_history.csv        loss per epoch
    results/charts/<model>_training.png       training curves
    results/models/<model>.json               settings, size, timings, best epoch
    results/metrics/<model>.json              evaluator report (also in summary.csv)
    data/processed/scores/<model>.parquet     per-row scores for the ensemble (not committed)
    models/<model>.joblib                     fitted model (not committed)
"""

from __future__ import annotations

import argparse

import pandas as pd

from anomaly.config import load_config
from anomaly.evaluation.evaluator import Evaluator, save_report
from anomaly.evaluation.io import load_labeled, save_scores
from anomaly.models.deep import LSTMAutoencoderModel, TransformerForecastModel, WindowModel
from anomaly.models.preprocess import add_derived_features
from anomaly.plots import plot_training_curves
from anomaly.utils import get_logger, set_seed, timer, write_json

TITLES = {"lstm_autoencoder": "LSTM autoencoder: training curves", "transformer": "Transformer forecaster: training curves"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", choices=["lstm_autoencoder", "transformer", "all"], default="all")
    args = parser.parse_args()

    cfg = load_config(args.config)
    assert cfg.models and cfg.models.deep and cfg.evaluation and cfg.features
    set_seed(cfg.seed)
    log = get_logger()
    d, m = cfg.models.deep, cfg.models

    labeled, events = load_labeled(cfg)
    labeled = add_derived_features(labeled, vol_col=f"vol_{cfg.features.vol_windows[0]}")
    evaluator = Evaluator(labeled, events, cfg.evaluation)
    train = labeled[labeled["split"] == "train"]
    features = list(m.feature_sets[d.feature_set])
    common = dict(
        window=d.window,
        scaling=m.scaling,
        min_train_rows_per_ticker=m.min_train_rows_per_ticker,
        batch_size=d.batch_size,
        max_epochs=d.max_epochs,
        patience=d.patience,
        learning_rate=d.learning_rate,
        holdout_frac=d.holdout_frac,
        seed=cfg.seed,
        torch_threads=d.torch_threads,
    )
    builders = {
        "lstm_autoencoder": lambda: LSTMAutoencoderModel(features, **d.lstm_autoencoder, **common),
        "transformer": lambda: TransformerForecastModel(features, **d.transformer, **common),
    }
    names = list(builders) if args.model == "all" else [args.model]

    for name in names:
        model: WindowModel = builders[name]()
        log.info("Training %s on %d train rows", name, len(train))
        with timer() as fit_t:
            model.fit(train)
        history = pd.DataFrame(model.history)
        for row in model.history:
            log.info("  epoch %2d  train %.4f  holdout %.4f  (%.1fs)", row["epoch"], row["train_loss"], row["holdout_loss"], row["seconds"])
        with timer() as score_t:
            scores = model.score_frame(labeled)  # the full history, so val/test bars keep their earlier context
        save_scores(cfg, name, scores)
        model.save(cfg.paths.models_dir / f"{name}.joblib")

        out_dir = cfg.paths.results_dir / "models"
        out_dir.mkdir(parents=True, exist_ok=True)
        history.round(6).to_csv(out_dir / f"{name}_history.csv", index=False)
        chart = plot_training_curves(history, model.best_epoch, TITLES[name], cfg.paths.results_dir / "charts" / f"{name}_training.png")

        report = evaluator.run(name, scores)
        save_report(report, cfg.paths.results_dir)
        write_json(
            {
                "model": name,
                "features": features,
                "window": d.window,
                "architecture": d.lstm_autoencoder if name == "lstm_autoencoder" else d.transformer,
                "parameters": model.n_parameters(),
                "train_windows": model.train_windows,
                "holdout_windows": model.holdout_windows,
                "epochs_run": len(history),
                "best_epoch": model.best_epoch,
                "best_holdout_loss": round(float(history.loc[history["epoch"] == model.best_epoch, "holdout_loss"].iloc[0]), 6),
                "fit_seconds": round(fit_t["seconds"], 1),
                "score_seconds_all_rows": round(score_t["seconds"], 1),
                "rows_scored": int(scores["score"].notna().sum()),
                "torch_threads": d.torch_threads,
                "seed": cfg.seed,
            },
            out_dir / f"{name}.json",
        )
        log.info("%s: %d params, best epoch %d of %d, fit %.0fs -> %s", name, model.n_parameters(), model.best_epoch, len(history), fit_t["seconds"], chart.relative_to(cfg.root))

    summary = pd.read_csv(cfg.paths.results_dir / "metrics" / "summary.csv")
    cols = ["model", "track", "pr_auc", "precision", "recall", "f1", "event_recall", "mean_delay_bars",
            "false_alarms_per_day_all_tickers", "false_alarms_per_ticker_month"]
    log.info("Test results, all models so far:\n%s", summary[summary["split"] == "test"][cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

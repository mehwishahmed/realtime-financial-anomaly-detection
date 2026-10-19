"""Phase 4: train the LSTM autoencoder and the Transformer forecaster, then evaluate them.

    python scripts/train_deep.py --config configs/default.yaml --model all

Each model trains on the train split only, early-stops on the last 10% of the
train dates, and goes through the same evaluator as every other model
(threshold tuned on val, results reported on test).

The LSTM autoencoder also searches over tighter bottlenecks (``hidden_size_grid``),
because the original (``hidden_size``) was found to copy large moves through:
  - every size, including the original, is scored on **val only**: PR-AUC plus a copying diagnostic
  - the best val PR-AUC on the tune track *among the grid sizes* is reported as ``lstm_autoencoder``
  - the original is also reported, as ``lstm_autoencoder_original``, so the change can be judged in the open

Outputs (per reported model):
    results/models/<model>_history.csv        loss per epoch
    results/charts/<model>_training.png       training curves
    results/models/<model>.json               settings, size, timings, best epoch
    results/metrics/<model>.json              evaluator report (also in summary.csv)
    results/models/lstm_autoencoder_bottleneck_search.csv   val-only search table
    data/processed/scores/<model>.parquet     per-row scores for the ensemble (not committed)
    models/<model>.joblib                     fitted model (not committed)
"""

from __future__ import annotations

import argparse
from typing import Any

import pandas as pd

from anomaly.config import Config, load_config
from anomaly.evaluation.evaluator import Evaluator, save_report
from anomaly.evaluation.io import load_labeled, save_scores
from anomaly.models.deep import LSTMAutoencoderModel, TransformerForecastModel, WindowModel
from anomaly.models.preprocess import add_derived_features
from anomaly.plots import plot_training_curves
from anomaly.utils import get_logger, set_seed, timer, write_json

log = get_logger()


def report_model(
    cfg: Config,
    name: str,
    title: str,
    model: WindowModel,
    fit_seconds: float,
    labeled: pd.DataFrame,
    evaluator: Evaluator,
    extra: dict[str, Any],
) -> None:
    """Score every row, save curves and scores, and run the shared evaluator under ``name``."""
    with timer() as score_t:
        scores = model.score_frame(labeled)  # the full history, so val/test bars keep their earlier context
    save_scores(cfg, name, scores)
    model.save(cfg.paths.models_dir / f"{name}.joblib")
    history = pd.DataFrame(model.history)
    out_dir = cfg.paths.results_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    history.round(6).to_csv(out_dir / f"{name}_history.csv", index=False)
    chart = plot_training_curves(history, model.best_epoch, title, cfg.paths.results_dir / "charts" / f"{name}_training.png")
    save_report(evaluator.run(name, scores), cfg.paths.results_dir)
    assert cfg.models and cfg.models.deep
    write_json(
        {
            "model": name,
            "features": model.features,
            "window": model.window,
            **extra,
            "parameters": model.n_parameters(),
            "train_windows": model.train_windows,
            "holdout_windows": model.holdout_windows,
            "epochs_run": len(history),
            "best_epoch": model.best_epoch,
            "best_holdout_loss": round(float(history.loc[history["epoch"] == model.best_epoch, "holdout_loss"].iloc[0]), 6),
            "fit_seconds": round(fit_seconds, 1),
            "score_seconds_all_rows": round(score_t["seconds"], 1),
            "rows_scored": int(scores["score"].notna().sum()),
            "torch_threads": cfg.models.deep.torch_threads,
            "seed": cfg.seed,
        },
        out_dir / f"{name}.json",
    )
    log.info("%s: %d params, best epoch %d of %d, fit %.0fs -> %s", name, model.n_parameters(), model.best_epoch, len(history), fit_seconds, chart.relative_to(cfg.root))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", choices=["lstm_autoencoder", "transformer", "all"], default="all")
    args = parser.parse_args()

    cfg = load_config(args.config)
    assert cfg.models and cfg.models.deep and cfg.evaluation and cfg.features
    set_seed(cfg.seed)
    d, m = cfg.models.deep, cfg.models
    track = cfg.evaluation.tune_track

    (cfg.paths.results_dir / "models").mkdir(parents=True, exist_ok=True)
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
        max_train_windows=d.max_train_windows,
    )

    if args.model in ("lstm_autoencoder", "all"):
        lcfg = d.lstm_autoencoder
        original = int(lcfg["hidden_size"])
        candidates = sorted(set(map(int, lcfg.get("hidden_size_grid", [original]))))
        grid = sorted({*candidates, original})  # the original is measured too, for the comparison table
        fitted: dict[int, tuple[LSTMAutoencoderModel, float]] = {}
        rows = []
        for hidden in grid:
            log.info("LSTM autoencoder, bottleneck %d", hidden)
            model = LSTMAutoencoderModel(features, hidden_size=hidden, num_layers=int(lcfg["num_layers"]), **common)
            with timer() as t:
                model.fit(train)
            fitted[hidden] = (model, t["seconds"])
            val_scores = model.score_frame(labeled)
            # PR-AUC doesn't depend on the threshold; only the val split is evaluated here.
            syn = evaluator.evaluate(val_scores, float("inf"), "val", "synthetic")["pr_auc"]
            real = evaluator.evaluate(val_scores, float("inf"), "val", "real")["pr_auc"]
            slope = model.copy_slope(labeled, split="val")
            rows.append(
                {
                    "hidden_size": hidden,
                    "role": "candidate" if hidden in candidates else "original (not a candidate)",
                    "parameters": model.n_parameters(),
                    "val_pr_auc_synthetic": round(syn, 6),
                    "val_pr_auc_real": round(real, 6),
                    "val_copy_slope_ret_z": round(slope, 4),
                    "best_epoch": model.best_epoch,
                    "epochs_run": len(model.history),
                    "fit_seconds": round(t["seconds"], 1),
                }
            )
            log.info("  val PR-AUC synthetic %.4f  real %.4f  copy slope %.2f  (best epoch %d, %.0fs)", syn, real, slope, model.best_epoch, t["seconds"])
        search = pd.DataFrame(rows)
        search.to_csv(cfg.paths.results_dir / "models" / "lstm_autoencoder_bottleneck_search.csv", index=False)
        pool = search[search["hidden_size"].isin(candidates)]
        best = int(pool.sort_values([f"val_pr_auc_{track}", "hidden_size"], ascending=[False, True]).iloc[0]["hidden_size"])
        log.info("Selected bottleneck %d by val PR-AUC (%s); original was %d", best, track, original)

        if original != best or original not in candidates:  # don't report the same model twice
            report_model(
                cfg, "lstm_autoencoder_original", f"LSTM autoencoder, original (bottleneck {original}): training curves",
                *fitted[original], labeled, evaluator,
                {"hidden_size": original, "role": "original first-pass model, kept for comparison"},
            )
        report_model(
            cfg, "lstm_autoencoder", f"LSTM autoencoder, selected (bottleneck {best}): training curves",
            *fitted[best], labeled, evaluator,
            {"hidden_size": best, "role": f"tighter bottleneck chosen by best val PR-AUC on the {track} track", "candidates": candidates},
        )

    if args.model in ("transformer", "all"):
        model = TransformerForecastModel(features, **d.transformer, **common)
        log.info("Transformer forecaster")
        with timer() as t:
            model.fit(train)
        report_model(cfg, "transformer", "Transformer forecaster: training curves", model, t["seconds"], labeled, evaluator, {"architecture": d.transformer})

    summary = pd.read_csv(cfg.paths.results_dir / "metrics" / "summary.csv")
    cols = ["model", "track", "pr_auc", "precision", "recall", "f1", "event_recall", "mean_delay_bars",
            "false_alarms_per_day_all_tickers", "false_alarms_per_ticker_month"]
    log.info("Test results, all models so far:\n%s", summary[summary["split"] == "test"][cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

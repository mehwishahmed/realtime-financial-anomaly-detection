"""Phase 5: combine the z-score rule, Isolation Forest, LSTM autoencoder, and Transformer.

    python scripts/run_ensemble.py --config configs/default.yaml

Steps:
    1. Put each member's scores on a common scale (normalizers fit on TRAIN-split scores only).
    2. Try every weight vector (steps of ``weight_step``, summing to 1) for each normalization,
       scoring each on the **val** split only.
    3. Report three ensembles through the shared evaluator:
         ensemble           best val PR-AUC on the tune track (synthetic)
         ensemble_balanced  best mean of val PR-AUC on the synthetic and real tracks
         ensemble_equal     equal weights, robust_z (untuned reference)

Outputs:
    results/models/ensemble_search.csv     val PR-AUC for every weight vector and normalization
    results/models/ensembles.json          chosen weights and normalization for each ensemble
    results/metrics/<ensemble>.json        evaluator reports (also in summary.csv)
    data/processed/scores/<ensemble>.parquet, models/<ensemble>.joblib   (not committed)
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from anomaly.config import load_config
from anomaly.evaluation.evaluator import Evaluator, save_report
from anomaly.evaluation.io import load_labeled, save_scores, scores_path
from anomaly.models.base import AnomalyModel
from anomaly.models.ensemble import EnsembleModel, ScoreNormalizer, combine, weight_grid
from anomaly.models.preprocess import add_derived_features
from anomaly.models.rules import ZScoreRuleModel
from anomaly.utils import get_logger, set_seed, timer, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    assert cfg.models and cfg.models.ensemble and cfg.evaluation and cfg.features
    set_seed(cfg.seed)
    log = get_logger()
    ens_cfg = cfg.models.ensemble
    track = cfg.evaluation.tune_track
    names = list(ens_cfg.members)

    (cfg.paths.results_dir / "models").mkdir(parents=True, exist_ok=True)
    labeled, events = load_labeled(cfg)
    labeled = add_derived_features(labeled, vol_col=f"vol_{cfg.features.vol_windows[0]}")
    evaluator = Evaluator(labeled, events, cfg.evaluation)
    split = labeled["split"].to_numpy()
    train = split == "train"

    # Member models and their raw scores, aligned to the labeled table's rows.
    rule = ZScoreRuleModel(vol_col=f"vol_{cfg.features.vol_windows[0]}")
    members: dict[str, AnomalyModel] = {}
    raw_cols = []
    for name in names:
        if name == rule.name:
            members[name] = rule
            raw_cols.append(rule.score(labeled))
        else:
            members[name] = AnomalyModel.load(cfg.paths.models_dir / f"{name}.joblib")
            raw_cols.append(evaluator.align(pd.read_parquet(scores_path(cfg, name))))
    raw = np.column_stack(raw_cols)
    log.info("Members: %s; scored share per member: %s", names, np.isfinite(raw).mean(axis=0).round(4).tolist())

    # Search weights on val.
    grid = weight_grid(len(names), ens_cfg.weight_step)
    rows = []
    normalized = {}
    with timer() as search_t:
        for norm in ens_cfg.normalizations:
            normalizers = [ScoreNormalizer(norm).fit(raw[train, i]) for i in range(len(names))]
            normalized[norm] = np.column_stack([nz.transform(raw[:, i]) for i, nz in enumerate(normalizers)])
            for w in grid:
                s = combine(normalized[norm], np.array(w))
                syn = evaluator.pr_auc_aligned(s, "val", "synthetic")
                real = evaluator.pr_auc_aligned(s, "val", "real")
                rows.append({"normalization": norm, **dict(zip(names, w)), "val_pr_auc_synthetic": syn, "val_pr_auc_real": real})
    search = pd.DataFrame(rows)
    search["val_pr_auc_mean"] = (search["val_pr_auc_synthetic"] + search["val_pr_auc_real"]) / 2
    out_dir = cfg.paths.results_dir / "models"
    search.round(6).to_csv(out_dir / "ensemble_search.csv", index=False)
    log.info("Searched %d weight vectors x %d normalizations on val in %.1fs", len(grid), len(ens_cfg.normalizations), search_t["seconds"])

    def pick(metric: str) -> tuple[str, dict[str, float]]:
        """Normalization and weights of the best val row by ``metric``."""
        best = search.sort_values(metric, ascending=False, kind="stable").iloc[0]
        return str(best["normalization"]), {n: float(best[n]) for n in names}

    choices = {
        "ensemble": (*pick(f"val_pr_auc_{track}"), f"best val PR-AUC on the {track} track"),
        "ensemble_balanced": (*pick("val_pr_auc_mean"), "best mean of val PR-AUC on the synthetic and real tracks"),
        "ensemble_equal": ("robust_z", {n: 1 / len(names) for n in names}, "equal weights, robust_z (untuned reference)"),
    }

    summary_info = {}
    train_scores = {n: raw[train, i] for i, n in enumerate(names)}
    for ens_name, (norm, weights, why) in choices.items():
        model = EnsembleModel(members, weights, norm).fit_from_scores(train_scores)
        model.name = ens_name
        scores = labeled[["ticker", "timestamp"]].assign(score=model.combine_raw(raw)).reset_index(drop=True)
        save_scores(cfg, ens_name, scores)
        model.save(cfg.paths.models_dir / f"{ens_name}.joblib")
        report = evaluator.run(ens_name, scores)
        save_report(report, cfg.paths.results_dir)
        val_aligned = evaluator.align(scores)
        summary_info[ens_name] = {
            "chosen_by": why,
            "normalization": norm,
            "weights": weights,
            "val_pr_auc_synthetic": round(evaluator.pr_auc_aligned(val_aligned, "val", "synthetic"), 6),
            "val_pr_auc_real": round(evaluator.pr_auc_aligned(val_aligned, "val", "real"), 6),
            "threshold": report["threshold"],
        }
        log.info("%s: %s %s  val PR-AUC syn %.4f real %.4f", ens_name, norm, weights,
                 summary_info[ens_name]["val_pr_auc_synthetic"], summary_info[ens_name]["val_pr_auc_real"])

    # The saved model must reproduce these scores end to end (members re-scoring the raw data), as streaming will.
    check = AnomalyModel.load(cfg.paths.models_dir / "ensemble.joblib")
    with timer() as e2e_t:
        end_to_end = check.score(labeled)
    stored = evaluator.align(pd.read_parquet(scores_path(cfg, "ensemble")))
    max_diff = float(np.nanmax(np.abs(end_to_end - stored)))
    same_nan = bool((np.isnan(end_to_end) == np.isnan(stored)).all())
    log.info("End-to-end check: max |difference| %.2e, same unscored rows %s, %.1fs for %d rows", max_diff, same_nan, e2e_t["seconds"], len(labeled))
    if max_diff > 1e-4 or not same_nan:
        raise RuntimeError("Saved ensemble does not reproduce its scores")

    write_json(
        {
            "members": names,
            "weight_step": ens_cfg.weight_step,
            "normalizations_tried": list(ens_cfg.normalizations),
            "weight_vectors_tried": len(grid),
            "normalizers_fit_on": "train split scores of each member",
            "ensembles": summary_info,
            "end_to_end_check": {"max_abs_difference": max_diff, "same_unscored_rows": same_nan, "seconds": round(e2e_t["seconds"], 1)},
        },
        out_dir / "ensembles.json",
    )

    summary = pd.read_csv(cfg.paths.results_dir / "metrics" / "summary.csv")
    cols = ["model", "track", "pr_auc", "precision", "recall", "f1", "event_recall", "mean_delay_bars",
            "false_alarms_per_day_all_tickers", "false_alarms_per_ticker_month"]
    log.info("Test results, all models:\n%s", summary[summary["split"] == "test"][cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

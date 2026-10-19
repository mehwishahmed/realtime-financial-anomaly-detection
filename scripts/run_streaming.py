"""Phase 6: replay data bar by bar through features -> members -> ensemble -> alarm, and time it.

    python scripts/run_streaming.py --config configs/default.yaml --run all

Runs:
    daily     The daily TEST period, replayed one bar at a time from the raw (injected) prices.
              Earlier bars only warm up the buffers. Produces latency AND detection /
              false-alarm metrics, and checks that streamed scores equal the batch scores.
    intraday  The 5-minute TEST days. LATENCY-ONLY: the models were trained on daily
              features, so their alarms on 5-minute bars aren't meaningful and aren't reported.

Outputs:
    results/streaming/latency_summary.csv          p50/p95/p99 per stage, both runs
    results/streaming/daily_replay.json            detection, false alarms, batch-vs-stream check
    results/streaming/daily_alarms_by_month.csv    alarms and false alarms per month
    results/streaming/intraday_latency_only.json   intraday run facts (no detection metrics)
    results/charts/streaming_latency.png, results/charts/streaming_false_alarms_by_month.png
    data/processed/streaming/<run>_records.parquet per-bar records (not committed)
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from anomaly.config import load_config
from anomaly.data.clean import clean_universe
from anomaly.data.download import load_cached
from anomaly.data.splits import assign_splits
from anomaly.data.universe import select_tickers
from anomaly.evaluation.evaluator import Evaluator
from anomaly.evaluation.io import load_labeled, scores_path
from anomaly.labels.attach import TRACKS
from anomaly.models.base import AnomalyModel
from anomaly.models.preprocess import add_derived_features
from anomaly.plots import plot_latency, plot_monthly_false_alarms
from anomaly.streaming.features import StreamingFeatures
from anomaly.streaming.simulator import StreamingScorer, latency_summary, replay
from anomaly.utils import get_logger, set_seed, timer, write_json

log = get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run", choices=["daily", "intraday", "all"], default="all")
    args = parser.parse_args()

    cfg = load_config(args.config)
    assert cfg.streaming and cfg.features and cfg.evaluation and cfg.labels
    set_seed(cfg.seed)
    sc = cfg.streaming
    market = cfg.universe.market_ticker
    out_dir = cfg.paths.results_dir / "streaming"
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_dir = cfg.processed_dir() / "streaming"
    rec_dir.mkdir(parents=True, exist_ok=True)
    threshold = json.loads((cfg.paths.results_dir / "metrics" / f"{sc.model}.json").read_text())["threshold"]
    summaries = []

    def new_scorer() -> StreamingScorer:
        ensemble = AnomalyModel.load(cfg.paths.models_dir / f"{sc.model}.joblib")
        return StreamingScorer(ensemble, threshold, sc.torch_threads)

    if args.run in ("daily", "all"):
        freq = cfg.frequencies[cfg.labels.frequency]
        bars = pd.read_parquet(cfg.processed_dir() / f"{freq.name}_bars_injected.parquet")
        labeled, events = load_labeled(cfg)
        labeled = add_derived_features(labeled, vol_col=f"vol_{cfg.features.vol_windows[0]}")
        evaluator = Evaluator(labeled, events, cfg.evaluation)
        split_rows = labeled[labeled["split"] == sc.daily_split]
        start = split_rows["timestamp"].min()
        scorer = new_scorer()
        log.info("Daily replay: %d bars total, timing from %s (%s split), threshold %.4f", len(bars), start.date(), sc.daily_split, threshold)
        with timer() as wall:
            records = replay(bars, StreamingFeatures(freq, cfg.features, market), scorer, start, market)
        records.to_parquet(rec_dir / "daily_records.parquet", index=False)
        summaries.append(latency_summary(records, scorer.names, scorer.ensemble.weights, "daily"))

        # Same numbers as batch? (Streaming can only see the past, so a match also rules out lookahead in the batch pipeline.)
        stream_scores = records.loc[records["scored"], ["ticker", "timestamp", "ensemble"]].rename(columns={"ensemble": "score"})
        batch = pd.read_parquet(scores_path(cfg, sc.model))
        cmp = split_rows[["ticker", "timestamp"]].merge(batch, on=["ticker", "timestamp"]).merge(
            stream_scores, on=["ticker", "timestamp"], how="left", suffixes=("_batch", "_stream")
        )
        diff = (cmp["score_batch"] - cmp["score_stream"]).abs()
        alarm_mismatch = int(((cmp["score_batch"] >= threshold) != (cmp["score_stream"] >= threshold)).sum())

        metrics = {t: evaluator.evaluate(stream_scores, threshold, sc.daily_split, t) for t in TRACKS}
        keep = ["pr_auc", "precision", "recall", "f1", "event_recall", "mean_delay_bars", "median_delay_bars", "events",
                "false_alarms", "false_alarms_per_day_all_tickers", "false_alarms_per_ticker_day", "false_alarms_per_ticker_month"]
        merged = records.merge(labeled[["ticker", "timestamp", "synthetic_window", "real_window"]], on=["ticker", "timestamp"], how="left")
        merged["false_alarm"] = merged["alarm"] & merged["synthetic_window"].isna() & merged["real_window"].isna()
        merged["month"] = merged["timestamp"].dt.to_period("M")
        monthly = merged.groupby("month").agg(alarms=("alarm", "sum"), false_alarms=("false_alarm", "sum"), trading_days=("timestamp", "nunique")).reset_index()
        monthly.to_csv(out_dir / "daily_alarms_by_month.csv", index=False)
        plot_monthly_false_alarms(monthly, cfg.paths.results_dir / "charts" / "streaming_false_alarms_by_month.png",
                                  "False alarms per month, daily replay of the test period (100 tickers)")

        deployed = latency_summary(records, scorer.names, scorer.ensemble.weights, "daily").set_index("stage").loc["deployed_total"]
        write_json(
            {
                "run": "daily",
                "model": sc.model,
                "weights": scorer.ensemble.weights,
                "threshold": threshold,
                "split": sc.daily_split,
                "period": [str(start.date()), str(split_rows["timestamp"].max().date())],
                "trading_days": int(split_rows["timestamp"].nunique()),
                "tickers": int(records["ticker"].nunique()),
                "bars_timed": len(records),
                "bars_scored": int(records["scored"].sum()),
                "alarms": int(records["alarm"].sum()),
                "replay_wall_seconds_incl_warmup": round(wall["seconds"], 1),
                "deployed_latency_us": {k: round(float(deployed[k]), 1) for k in ("p50_us", "p95_us", "p99_us", "mean_us")},
                "throughput_bars_per_second_deployed": round(1e6 / float(deployed["mean_us"]), 0),
                "batch_vs_stream": {
                    "rows_compared": len(cmp),
                    "rows_missing_in_stream": int(cmp["score_stream"].isna().sum() - cmp["score_batch"].isna().sum()),
                    "max_abs_score_difference": float(diff.max()),
                    "alarm_disagreements": alarm_mismatch,
                },
                "metrics": {t: {k: metrics[t][k] for k in keep} for t in TRACKS},
            },
            out_dir / "daily_replay.json",
        )
        log.info("Daily: %d bars timed, deployed p50 %.0f us, p99 %.0f us; batch-vs-stream max diff %.2e, alarm disagreements %d",
                 len(records), deployed["p50_us"], deployed["p99_us"], diff.max(), alarm_mismatch)
        for t in TRACKS:
            log.info("  %s: PR-AUC %.3f  event recall %.3f  FA/day (all tickers) %.2f  FA/ticker/month %.3f", t,
                     metrics[t]["pr_auc"], metrics[t]["event_recall"], metrics[t]["false_alarms_per_day_all_tickers"], metrics[t]["false_alarms_per_ticker_month"])

    if args.run in ("intraday", "all") and sc.intraday_latency_only:
        freq = cfg.frequencies["intraday"]
        tickers = select_tickers(cfg)
        assert cfg.cleaning is not None
        clean = clean_universe(load_cached(tickers, cfg.raw_dir(freq.name)), market, freq, cfg.cleaning).data
        _, bounds = assign_splits(clean, freq.splits)
        ts = clean["timestamp"]
        day = ts.dt.tz_localize(None).dt.normalize()
        start = ts[day > bounds["val_end"]].min()
        scorer = new_scorer()
        log.info("Intraday LATENCY-ONLY replay: %d bars, timing from %s", len(clean), start)
        with timer() as wall:
            records = replay(clean, StreamingFeatures(freq, cfg.features, market), scorer, start, market)
        records.to_parquet(rec_dir / "intraday_records.parquet", index=False)
        lat = latency_summary(records, scorer.names, scorer.ensemble.weights, "intraday (latency only)")
        summaries.append(lat)
        deployed = lat.set_index("stage").loc["deployed_total"]
        write_json(
            {
                "run": "intraday",
                "latency_only": True,
                "why": "The models were trained on daily features. Their scores on 5-minute bars are not meaningful "
                       "detections, so no detection or false-alarm metrics are reported for this run.",
                "interval": freq.interval,
                "period": [str(start), str(ts.max())],
                "trading_days": int(day[day > bounds["val_end"]].nunique()),
                "tickers": int(records["ticker"].nunique()),
                "bars_timed": len(records),
                "bars_scored": int(records["scored"].sum()),
                "replay_wall_seconds_incl_warmup": round(wall["seconds"], 1),
                "deployed_latency_us": {k: round(float(deployed[k]), 1) for k in ("p50_us", "p95_us", "p99_us", "mean_us")},
                "throughput_bars_per_second_deployed": round(1e6 / float(deployed["mean_us"]), 0),
            },
            out_dir / "intraday_latency_only.json",
        )
        log.info("Intraday (latency only): %d bars timed, deployed p50 %.0f us, p99 %.0f us", len(records), deployed["p50_us"], deployed["p99_us"])

    if summaries:
        path = out_dir / "latency_summary.csv"
        new = pd.concat(summaries, ignore_index=True)
        if path.exists():
            old = pd.read_csv(path)
            new = pd.concat([old[~old["run"].isin(new["run"])], new], ignore_index=True)
        new.round(2).to_csv(path, index=False)
        plot_latency(new, cfg.paths.results_dir / "charts" / "streaming_latency.png", "Per-bar latency, tick-by-tick replay on a laptop CPU")
        log.info("Latency summary:\n%s", new.round(1).to_string(index=False))


if __name__ == "__main__":
    main()

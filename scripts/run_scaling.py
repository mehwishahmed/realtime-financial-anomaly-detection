"""Run the deployed pipeline at one scale and record what it handled, how long it took, and how it performed.

    python scripts/run_scaling.py --config configs/scale_sp500.yaml

Steps (each in its own process, timed, with its peak memory measured by /usr/bin/time -l):
    download   daily bars for the universe (shared cache: only new tickers are downloaded)
    features   clean, build features, assign splits
    labels     real events (incl. earnings dates) and synthetic anomalies
    baselines  z-score rule through the evaluator
    train      LSTM autoencoder (bottleneck 4)
    ensemble   rule + LSTM, weights tuned on this scale's val split
    stream     tick-by-tick replay of the daily test period

The run stops at the first step that fails or exceeds ``scaling.step_timeout_minutes``.
Whatever completed is still summarized, with the reason it stopped.

Outputs:
    <results_dir>/scaling_summary.json      everything below for this scale
    results/scaling/scaling_table.csv       one row per scale (updated in place)
    <data_dir>/logs/<step>.log              full step output (not committed)
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from anomaly.config import load_config
from anomaly.reporting import scaling_row
from anomaly.utils import get_logger, write_json

STEPS = [
    ("download", ["scripts/download_data.py", "--freq", "daily"]),
    ("features", ["scripts/build_features.py", "--freq", "daily"]),
    ("labels", ["scripts/make_labels.py"]),
    ("baselines", ["scripts/evaluate_baselines.py"]),
    ("train", ["scripts/train_deep.py", "--model", "lstm_autoencoder"]),
    ("ensemble", ["scripts/run_ensemble.py"]),
    ("stream", ["scripts/run_streaming.py", "--run", "daily"]),
]


def peak_rss_mb(time_output: str) -> float | None:
    """Parse '/usr/bin/time -l' output (macOS reports maximum resident set size in bytes)."""
    match = re.search(r"(\d+)\s+maximum resident set size", time_output)
    return round(int(match.group(1)) / 1e6, 1) if match else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    assert cfg.scaling is not None, "scaling configs need a 'scaling' section"
    log = get_logger()
    scale = Path(args.config).stem.removeprefix("scale_")
    log_dir = cfg.paths.data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(cfg.root / "src")}
    timeout = cfg.scaling.step_timeout_minutes * 60

    steps, stopped = [], None
    for name, cmd in STEPS:
        log.info("[%s] %s ...", scale, name)
        full = ["/usr/bin/time", "-l", sys.executable, *cmd, "--config", args.config]
        start = time.perf_counter()
        try:
            proc = subprocess.run(full, cwd=cfg.root, env=env, capture_output=True, text=True, timeout=timeout)
            seconds = time.perf_counter() - start
            (log_dir / f"{name}.log").write_text(proc.stdout + proc.stderr)
            status = "ok" if proc.returncode == 0 else f"failed (exit {proc.returncode})"
            rss = peak_rss_mb(proc.stderr)
        except subprocess.TimeoutExpired as exc:
            seconds = time.perf_counter() - start
            out = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            err = (exc.stderr or b"").decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            (log_dir / f"{name}.log").write_text(out + err)
            status, rss = f"timed out after {cfg.scaling.step_timeout_minutes} min", None
        steps.append({"step": name, "status": status, "seconds": round(seconds, 1), "peak_rss_mb": rss})
        log.info("[%s] %s: %s in %.0fs, peak memory %s MB", scale, name, status, seconds, rss)
        if status != "ok":
            stopped = f"{name}: {status} (see {(log_dir / f'{name}.log').relative_to(cfg.root)})"
            break

    row = scaling_row(cfg, scale, steps, stopped)
    write_json({"scale": scale, "config": args.config, "steps": steps, "stopped": stopped, "summary": row}, cfg.paths.results_dir / "scaling_summary.json")
    table_path = cfg.root / "results" / "scaling" / "scaling_table.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([row])
    if table_path.exists():
        old = pd.read_csv(table_path, dtype={"scale": str})
        table = pd.concat([old[old["scale"] != scale], table], ignore_index=True)
    order = {"100": 0, "sp500": 1, "us_all": 2}
    table = table.sort_values("scale", key=lambda s: s.map(order).fillna(9)).reset_index(drop=True)
    table.to_csv(table_path, index=False)
    log.info("[%s] done%s. Summary row:\n%s", scale, f" (stopped at {stopped})" if stopped else "", pd.Series(row).to_string())


if __name__ == "__main__":
    main()

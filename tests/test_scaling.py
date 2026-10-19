"""Scaling-run helpers: memory parsing and the generated README scaling table."""

from __future__ import annotations

import importlib.util

import pandas as pd
import pytest

from anomaly.reporting import SCALING_END, SCALING_START, scaling_markdown, update_readme_scaling

from conftest import REPO_ROOT


def _run_scaling_module():
    spec = importlib.util.spec_from_file_location("run_scaling", REPO_ROOT / "scripts" / "run_scaling.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_peak_memory_is_parsed_from_time_output():
    rs = _run_scaling_module()
    out = "        1.23 real         1.00 user\n  1312800768  maximum resident set size\n  0  swaps\n"
    assert rs.peak_rss_mb(out) == pytest.approx(1312.8)
    assert rs.peak_rss_mb("no memory line") is None


def _row(scale, completed=True, stopped=""):
    return {
        "scale": scale, "completed": completed, "stopped_at": stopped, "tickers_requested": 500, "tickers_newly_downloaded": 400,
        "tickers_failed_download": 2, "tickers_passed_cleaning": 480, "feature_rows": 1_400_000, "download_seconds": 60.0,
        "features_seconds": 8.0, "labels_seconds": 200.0, "train_seconds": 600.0, "ensemble_seconds": 20.0, "stream_seconds": 120.0,
        "total_seconds": 1008.0, "peak_rss_mb": 4000.0, "ensemble_weights": "rule 0.9, lstm 0.1", "stream_bars": 270_000,
        "latency_p50_us": 210.0, "latency_p95_us": 230.0, "latency_p99_us": 260.0, "false_alarms_per_day_all_tickers": 2.3,
        "false_alarms_per_ticker_month": 0.1, "test_pr_auc_synthetic": 0.31, "test_pr_auc_real": 0.7,
    }


def test_scaling_table_shows_each_scale_and_stops_honestly():
    table = pd.DataFrame([_row("100"), _row("us_all", completed=False, stopped="labels: timed out after 120 min")])
    md = scaling_markdown(table)
    assert "| | 100 stocks | All US-listed |" in md
    assert "**Passed cleaning (assets handled)** | 480 | 480 |" in md
    assert "210 / 230 / 260" in md and "16.8" in md  # latency triple and total minutes
    assert "stopped: labels: timed out after 120 min" in md


def test_missing_values_render_as_dashes(tmp_path):
    row = _row("us_all", completed=False, stopped="train: failed")
    for key in ("stream_bars", "latency_p50_us", "latency_p95_us", "latency_p99_us", "test_pr_auc_synthetic", "test_pr_auc_real", "train_seconds"):
        row[key] = None
    csv = tmp_path / "t.csv"
    pd.DataFrame([row]).to_csv(csv, index=False)
    readme = tmp_path / "README.md"
    readme.write_text(f"top\n{SCALING_START}\nold\n{SCALING_END}\nbottom\n")
    md = update_readme_scaling(readme, csv)
    assert "– / – / –" in md and "| LSTM training time (s) | – |" in md
    assert readme.read_text().startswith("top\n") and readme.read_text().endswith("bottom\n")

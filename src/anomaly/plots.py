"""Static charts saved to results/. Colors come from a validated colorblind-safe palette."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834"]  # slots 1-2; validated together (CVD dE 24.7, normal dE 33.6, both >= 3:1 contrast)


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_training_curves(history: pd.DataFrame, best_epoch: int, title: str, path: Path) -> Path:
    """Train vs holdout loss per epoch, with the kept (best) epoch marked."""
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150, facecolor=SURFACE)
    _style(ax)
    lines = [("train_loss", "Train", "-"), ("holdout_loss", "Holdout (last 10% of train dates)", "--")]
    for (col, label, dash), color in zip(lines, SERIES):
        ax.plot(history["epoch"], history[col], color=color, linewidth=2, linestyle=dash, marker="o", markersize=4, label=label)
        last = history.iloc[-1]
        ax.annotate(
            f"{last[col]:.3f}", (last["epoch"], last[col]), xytext=(6, 0), textcoords="offset points",
            va="center", fontsize=9, color=TEXT_MUTED,
        )
    best = history.loc[history["epoch"] == best_epoch, "holdout_loss"].iloc[0]
    ax.axvline(best_epoch, color=TEXT_MUTED, linewidth=1, linestyle=":")
    ax.annotate(
        f"kept: epoch {best_epoch}", (best_epoch, best), xytext=(0, 14), textcoords="offset points",
        ha="center", fontsize=9, color=TEXT,
    )
    ax.set_title(title, loc="left", fontsize=12, color=TEXT, pad=28)
    ax.set_xlabel("Epoch", color=TEXT_MUTED, fontsize=10)
    ax.set_ylabel("Mean squared error (scaled features)", color=TEXT_MUTED, fontsize=10)
    ax.xaxis.get_major_locator().set_params(integer=True)
    # Legend in its own row above the plot, so it can never cover a line or a label.
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


PERCENTILE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # slots 1-3, validated together; aqua < 3:1 contrast, so bars carry value labels
STAGE_LABELS = {
    "features": "Features (incremental)",
    "baseline_zscore_rule": "Z-score rule",
    "isolation_forest": "Isolation Forest",
    "lstm_autoencoder": "LSTM autoencoder",
    "transformer": "Transformer",
    "combine": "Ensemble combine",
    "deployed_total": "Total, deployed ensemble",
    "all_members_total": "Total, all four members",
}


def plot_latency(summary: pd.DataFrame, path: Path, title: str) -> Path:
    """Per-stage p50/p95/p99 latency, one panel per run, log-scale microseconds, every bar value-labeled."""
    runs = list(dict.fromkeys(summary["run"]))
    stages = [s for s in STAGE_LABELS if s in set(summary["stage"])]
    fig, axes = plt.subplots(1, len(runs), figsize=(6.2 * len(runs), 5.2), dpi=150, facecolor=SURFACE, sharey=True, squeeze=False)
    xmax = summary[["p99_us"]].to_numpy().max() * 6
    for ax, run in zip(axes[0], runs):
        _style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        s = summary[summary["run"] == run].set_index("stage")
        y = np.arange(len(stages))
        height = 0.26
        for k, (col, label) in enumerate((("p50_us", "p50"), ("p95_us", "p95"), ("p99_us", "p99"))):
            vals = [s.loc[st, col] for st in stages]
            pos = y + (k - 1) * height
            ax.barh(pos, vals, height=height - 0.04, color=PERCENTILE_COLORS[k], label=label)
            for p_, v in zip(pos, vals):
                ax.text(v * 1.12, p_, f"{v:,.0f}" if v >= 10 else f"{v:.1f}", va="center", fontsize=7, color=TEXT_MUTED)
        ax.set_xscale("log")
        ax.set_xlim(left=max(summary["p50_us"].min() / 3, 0.1), right=xmax)
        ax.set_yticks(y, [STAGE_LABELS[st] for st in stages], fontsize=9, color=TEXT)
        ax.invert_yaxis()
        n = int(s["bars"].iloc[0])
        ax.set_title(f"{run} ({n:,} bars scored)", loc="left", fontsize=10, color=TEXT)
        ax.set_xlabel("Microseconds per bar (log scale)", color=TEXT_MUTED, fontsize=9)
    axes[0][0].legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="lower left", bbox_to_anchor=(0, 1.07), ncol=3, borderaxespad=0.2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=TEXT)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_monthly_false_alarms(monthly: pd.DataFrame, path: Path, title: str) -> Path:
    """False alarms per month across the replay (one series, so the title names it and there's no legend)."""
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=150, facecolor=SURFACE)
    _style(ax)
    labels = monthly["month"].astype(str).tolist()
    ax.bar(range(len(labels)), monthly["false_alarms"], color=SERIES[0], width=0.8)
    top = monthly["false_alarms"].nlargest(3)
    for i, v in top.items():
        ax.annotate(f"{labels[i]}: {v}", (i, v), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8, color=TEXT_MUTED)
    step = max(1, len(labels) // 9)
    ax.set_xticks(range(0, len(labels), step), labels[::step], fontsize=8, color=TEXT_MUTED, rotation=0)
    ax.set_ylabel("False alarms (all tickers)", color=TEXT_MUTED, fontsize=9)
    ax.set_title(title, loc="left", fontsize=12, color=TEXT, pad=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path

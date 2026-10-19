"""Static charts saved to results/. Colors come from a validated colorblind-safe palette."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files without needing a display
import matplotlib.pyplot as plt
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

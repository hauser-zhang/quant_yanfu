from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def _apply_clean_style(ax):
    """Apply a clean, publication-like style to the axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", axis="y", alpha=0.2)


def plot_model_comparison(metrics_df: pd.DataFrame, out_path: Path) -> None:
    """Plot model comparison bars for valid/test correlations."""
    fig, ax = plt.subplots(figsize=(10, 4), dpi=200)
    x = range(len(metrics_df))
    ax.bar([i - 0.2 for i in x], metrics_df["valid_corr_raw"], width=0.2, label="valid raw")
    ax.bar([i for i in x], metrics_df["valid_corr_z"], width=0.2, label="valid z")
    ax.bar([i + 0.2 for i in x], metrics_df["test_corr_z"], width=0.2, label="test z")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics_df["model"], rotation=25, ha="right")
    ax.set_title("Model Comparison")
    ax.legend(frameon=False, ncol=3, fontsize=9)
    _apply_clean_style(ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_daily_ic(ic_df: pd.DataFrame, out_path: Path, title: str | None = None) -> None:
    """Plot daily IC time series."""
    if ic_df.empty:
        return
    df = ic_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    fig, ax = plt.subplots(figsize=(10, 3.8), dpi=200)
    ax.plot(df["date"], df["corr"], linewidth=1.0, color="#2A4C8B")
    ax.set_title(title or "Daily IC")
    ax.set_xlabel("Date")
    ax.set_ylabel("IC")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=45)
    _apply_clean_style(ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_ablation(ablation_df: pd.DataFrame, out_path: Path) -> None:
    """Plot ablation bar chart for valid z correlation."""
    fig, ax = plt.subplots(figsize=(10, 4), dpi=200)
    ax.bar(ablation_df["setting"], ablation_df["valid_corr_z"], color="#4E79A7")
    ax.set_title("Ablation (valid z)")
    ax.tick_params(axis="x", rotation=30)
    _apply_clean_style(ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(fi_df: pd.DataFrame, out_path: Path, top_n: int = 30) -> None:
    """Plot top-N feature importances."""
    df = fi_df.sort_values("importance", ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    ax.barh(df["feature"], df["importance"], color="#59A14F")
    ax.invert_yaxis()
    ax.set_title("Feature Importance")
    _apply_clean_style(ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

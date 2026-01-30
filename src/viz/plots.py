from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def _apply_clean_style(ax):
    """Apply a clean, publication-like style to the axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", axis="y", alpha=0.25, color="#D8D2CC")


plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    }
)


def plot_model_comparison(metrics_df: pd.DataFrame, out_path: Path) -> None:
    """Plot model comparison bars for train/valid/test scores."""
    fig, ax = plt.subplots(figsize=(12, 5), dpi=220)
    x = range(len(metrics_df))
    c_train = "#B7C4B1"
    c_valid = "#C9A89C"
    c_test = "#A7B3C4"
    ax.bar([i - 0.25 for i in x], metrics_df["train_score"], width=0.2, label="train score", color=c_train)
    ax.bar([i for i in x], metrics_df["valid_score"], width=0.2, label="valid score", color=c_valid)
    ax.bar([i + 0.25 for i in x], metrics_df["test_score"], width=0.2, label="test score", color=c_test)
    ax.set_xticks(list(x))
    label_col = "model" if "model" in metrics_df.columns else "model_name"
    ax.set_xticklabels(metrics_df[label_col], rotation=25, ha="right")
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

    fig, ax = plt.subplots(figsize=(12, 4.2), dpi=220)
    ax.plot(df["date"], df["corr"], linewidth=1.0, color="#6B7D8A")
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
    """Plot ablation bar chart for valid score."""
    fig, ax = plt.subplots(figsize=(12, 5), dpi=220)
    colors = []
    for _, row in ablation_df.iterrows():
        if bool(row.get("is_full", False)):
            colors.append("#7C8F7A")
        else:
            colors.append("#B4A7B3")
    ax.bar(ablation_df["setting"], ablation_df["valid_score"], color=colors)
    ax.set_title("Ablation (valid score)")
    ax.tick_params(axis="x", rotation=30)
    _apply_clean_style(ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(fi_df: pd.DataFrame, out_path: Path, top_n: int = 30) -> None:
    """Plot top-N feature importances."""
    df = fi_df.sort_values("importance", ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(12, 7), dpi=220)
    ax.barh(df["feature"], df["importance"], color="#9AA8A1")
    ax.invert_yaxis()
    ax.set_title("Feature Importance")
    _apply_clean_style(ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

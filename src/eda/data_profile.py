from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


def compute_missingness(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Compute per-column missing rate table."""
    n_total = len(df)
    rows = []
    for col in df.columns:
        if col == date_col:
            continue
        n_missing = int(df[col].isna().sum())
        rows.append(
            {
                "feature": col,
                "missing_rate": n_missing / max(n_total, 1),
                "n_missing": n_missing,
                "n_total": n_total,
            }
        )
    out = pd.DataFrame(rows).sort_values("missing_rate", ascending=False)
    return out


def compute_y_summary(df: pd.DataFrame, y_col: str = "y") -> pd.DataFrame:
    """Summarize label distribution."""
    s = df[y_col].dropna()
    row = {
        "count": int(s.shape[0]),
        "mean": float(s.mean()) if len(s) else float("nan"),
        "std": float(s.std()) if len(s) else float("nan"),
        "p1": float(s.quantile(0.01)) if len(s) else float("nan"),
        "p5": float(s.quantile(0.05)) if len(s) else float("nan"),
        "p50": float(s.quantile(0.50)) if len(s) else float("nan"),
        "p95": float(s.quantile(0.95)) if len(s) else float("nan"),
        "p99": float(s.quantile(0.99)) if len(s) else float("nan"),
    }
    return pd.DataFrame([row])


def compute_weight_summary(df: pd.DataFrame, weight_col: str = "weight") -> pd.DataFrame:
    """Summarize sample weight distribution."""
    s = df[weight_col].dropna()
    row = {
        "count": int(s.shape[0]),
        "mean": float(s.mean()) if len(s) else float("nan"),
        "std": float(s.std()) if len(s) else float("nan"),
        "min": float(s.min()) if len(s) else float("nan"),
        "p50": float(s.quantile(0.50)) if len(s) else float("nan"),
        "p95": float(s.quantile(0.95)) if len(s) else float("nan"),
        "p99": float(s.quantile(0.99)) if len(s) else float("nan"),
        "max": float(s.max()) if len(s) else float("nan"),
    }
    return pd.DataFrame([row])


def plot_missingness_top30(miss_df: pd.DataFrame, out_path: Path) -> None:
    """Plot top-30 columns by missing rate."""
    top = miss_df.head(30).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(top["feature"], top["missing_rate"])
    ax.set_xlabel("Missing Rate")
    ax.set_title("Top Missing Features")
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def plot_hist(df: pd.DataFrame, col: str, out_path: Path, bins: int = 60, log_x: bool = False) -> None:
    """Plot single-column histogram."""
    s = df[col].dropna()
    if s.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(s, bins=bins)
    if log_x:
        ax.set_xscale("log")
    ax.set_title(f"{col} distribution")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def plot_weight_box(df: pd.DataFrame, weight_col: str, out_path: Path) -> None:
    """Plot weight boxplot."""
    s = df[weight_col].dropna()
    if s.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(s.values, vert=False)
    ax.set_title(f"{weight_col} boxplot")
    ax.set_xlabel(weight_col)
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def run_data_profile(
    df_train: pd.DataFrame,
    tables_dir: Path,
    figures_dir: Path,
    y_col: str = "y_raw",
    weight_col: str = "weight",
    rep_cols: Iterable[str] = ("r_0", "dv_0", "beta", "f_0"),
) -> None:
    """Generate chapter-1 profiling artifacts from train data."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    miss_df = compute_missingness(df_train)
    miss_df.to_csv(tables_dir / "column_missingness.csv", index=False)
    plot_missingness_top30(miss_df, figures_dir / "missingness_top30.png")

    y_summary = compute_y_summary(df_train, y_col=y_col)
    y_summary.to_csv(tables_dir / "y_summary.csv", index=False)
    plot_hist(df_train, y_col, figures_dir / "y_hist.png")

    w_summary = compute_weight_summary(df_train, weight_col=weight_col)
    w_summary.to_csv(tables_dir / "weight_summary.csv", index=False)
    plot_hist(df_train, weight_col, figures_dir / "weight_hist.png")
    plot_weight_box(df_train, weight_col, figures_dir / "weight_box.png")

    for c in rep_cols:
        if c in df_train.columns:
            plot_hist(df_train, c, figures_dir / f"feature_hist_{c.replace('_', '')}.png")

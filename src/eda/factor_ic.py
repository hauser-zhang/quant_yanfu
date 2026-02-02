from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eval.metrics import weighted_corr

META_EXCLUDE = {"id", "y", "y_raw", "y_label", "y_score", "weight", "DateTime", "date"}
DAILY_CONST_PREFIX = ("dow_", "mon_", "mkt_")


def compute_factor_daily_ic(
    df: pd.DataFrame,
    feature_col: str,
    label_col: str = "y_score",
    weight_col: str = "weight",
    date_col: str = "date",
) -> pd.DataFrame:
    """Compute per-day weighted IC for one factor."""
    rows = []
    for d, g in df.groupby(date_col):
        x = g[feature_col].to_numpy()
        y = g[label_col].to_numpy()
        w = g[weight_col].to_numpy()
        mask = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(w)) & (w > 0)
        if mask.sum() == 0:
            continue
        ic_t = weighted_corr(x[mask], y[mask], w[mask])
        W_t = float(w[mask].sum())
        rows.append({"date": d, "ic": ic_t, "W_t": W_t, "n": int(mask.sum())})
    return pd.DataFrame(rows)


def _factor_ic_agg(daily_df: pd.DataFrame, total_days: int) -> dict:
    if daily_df.empty:
        return {
            "ic_mean": np.nan,
            "ic_std": np.nan,
            "ic_ir": np.nan,
            "coverage_days": 0,
            "coverage_ratio": 0.0,
        }
    ww = daily_df["W_t"].to_numpy()
    ic = daily_df["ic"].to_numpy()
    m = np.isfinite(ic) & np.isfinite(ww) & (ww > 0)
    if m.sum() == 0:
        return {
            "ic_mean": np.nan,
            "ic_std": np.nan,
            "ic_ir": np.nan,
            "coverage_days": int(daily_df.shape[0]),
            "coverage_ratio": float(daily_df.shape[0] / max(total_days, 1)),
        }
    ic_mean = float((ww[m] * ic[m]).sum() / ww[m].sum())
    ic_std = float(np.nanstd(ic[m]))
    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_ir": float(ic_mean / (ic_std + 1e-12)),
        "coverage_days": int(daily_df.shape[0]),
        "coverage_ratio": float(daily_df.shape[0] / max(total_days, 1)),
    }


def compute_factor_table(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    label_col: str = "y_score",
    weight_col: str = "weight",
    date_col: str = "date",
) -> pd.DataFrame:
    """Compute aggregated single-factor IC table."""
    total_days = int(df[date_col].nunique())
    rows: List[dict] = []
    for col in feature_cols:
        daily = compute_factor_daily_ic(df, col, label_col=label_col, weight_col=weight_col, date_col=date_col)
        agg = _factor_ic_agg(daily, total_days=total_days)
        rows.append({"feature": col, **agg})
    out = pd.DataFrame(rows)
    out["abs_ic_mean"] = out["ic_mean"].abs()
    out = out.sort_values("abs_ic_mean", ascending=False).drop(columns=["abs_ic_mean"])
    return out


def default_factor_cols(df: pd.DataFrame, extra_exclude: Sequence[str] | None = None) -> List[str]:
    """Select factor columns for scan, excluding meta and daily-constant proxies."""
    extra_exclude = set(extra_exclude or [])
    cols = []
    for c in df.columns:
        if c in META_EXCLUDE or c in extra_exclude:
            continue
        if c.startswith(DAILY_CONST_PREFIX):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def plot_factor_ic_top20(factor_df: pd.DataFrame, out_path: Path) -> None:
    """Plot signed IC bars for top-20 |IC| features."""
    if factor_df.empty:
        return
    top = factor_df.head(20).iloc[::-1]
    vals = top["ic_mean"]
    colors = ["#507DBC" if v >= 0 else "#C06C84" for v in vals]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["feature"], vals, color=colors)
    ax.axvline(0, color="#666666", linewidth=1)
    ax.set_title("Top20 Single-Factor IC (train)")
    ax.set_xlabel("daily-weighted-mean IC")
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def plot_factor_ic_hist(factor_df: pd.DataFrame, out_path: Path) -> None:
    """Plot histogram of factor IC mean."""
    if factor_df.empty:
        return
    x = factor_df["ic_mean"].dropna()
    if x.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(x, bins=40)
    ax.set_title("Distribution of Factor IC Mean")
    ax.set_xlabel("ic_mean")
    ax.set_ylabel("Count")
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


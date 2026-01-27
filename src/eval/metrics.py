from __future__ import annotations

import numpy as np
import pandas as pd


def weighted_corr(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    """Weighted Pearson correlation with NaN and non-positive weight filtering."""
    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred)) & (~np.isnan(w)) & (w > 0)
    if mask.sum() == 0:
        return float("nan")
    y = y_true[mask]
    p = y_pred[mask]
    ww = w[mask]
    wsum = ww.sum()
    if wsum <= 0:
        return float("nan")
    y_mean = (ww * y).sum() / wsum
    p_mean = (ww * p).sum() / wsum
    cov = (ww * (y - y_mean) * (p - p_mean)).sum() / wsum
    var_y = (ww * (y - y_mean) ** 2).sum() / wsum
    var_p = (ww * (p - p_mean) ** 2).sum() / wsum
    denom = np.sqrt(var_y * var_p)
    if denom <= 1e-12:
        return float("nan")
    return float(cov / denom)


def daily_weighted_corr(df: pd.DataFrame, pred_col: str, y_col: str, w_col: str, date_col: str) -> pd.DataFrame:
    """Compute per-date weighted correlation series."""
    out = []
    for date, g in df.groupby(date_col):
        corr = weighted_corr(g[y_col].to_numpy(), g[pred_col].to_numpy(), g[w_col].to_numpy())
        out.append({"date": date, "corr": corr})
    return pd.DataFrame(out)

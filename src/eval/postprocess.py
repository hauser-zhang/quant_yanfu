from __future__ import annotations

import numpy as np
import pandas as pd


def cs_zscore_by_date(df: pd.DataFrame, pred_col: str, weight_col: str, date_col: str) -> pd.DataFrame:
    """Apply weighted cross-sectional z-score within each date."""
    out = df.copy()
    pred_z = []
    for _, g in out.groupby(date_col):
        w = g[weight_col].to_numpy()
        p = g[pred_col].to_numpy()
        mask = (~np.isnan(p)) & (~np.isnan(w)) & (w > 0)
        if mask.sum() == 0:
            pred_z.extend([0.0] * len(g))
            continue
        ww = w[mask]
        pp = p[mask]
        mean = (ww * pp).sum() / ww.sum()
        var = (ww * (pp - mean) ** 2).sum() / ww.sum()
        std = np.sqrt(var)
        if not np.isfinite(std) or std <= 1e-12:
            z = np.zeros(len(g))
        else:
            z = (p - mean) / std
            z = np.where(np.isnan(z), 0.0, z)
        pred_z.extend(z.tolist())
    out["pred_z"] = pred_z
    return out

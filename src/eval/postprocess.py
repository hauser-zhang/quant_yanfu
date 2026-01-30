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


def pred_cs_zscore_by_date(
    df: pd.DataFrame, pred_col: str, weight_col: str, date_col: str, eps: float = 1e-12
) -> pd.DataFrame:
    """Weighted cross-sectional z-score for predictions per date."""
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
        if not np.isfinite(std) or std <= eps:
            z = np.zeros(len(g))
        else:
            z = (p - mean) / std
            z = np.where(np.isnan(z), 0.0, z)
        pred_z.extend(z.tolist())
    out["pred_z"] = pred_z
    return out


def pred_neutralize_by_date(
    df: pd.DataFrame,
    pred_col: str,
    weight_col: str,
    date_col: str,
    industry_col: str,
    beta_col: str,
    indbeta_col: str,
) -> pd.DataFrame:
    """Weighted WLS neutralization per date using industry, beta, indbeta."""
    out = df.copy()
    pred_neu = []
    for _, g in out.groupby(date_col):
        pred = g[pred_col].to_numpy()
        w = g[weight_col].to_numpy()
        ind = g[industry_col]
        beta = g[beta_col]
        indbeta = g[indbeta_col]

        mask = (~ind.isna()) & (~beta.isna()) & (~indbeta.isna()) & (w > 0) & (~np.isnan(pred))
        if mask.sum() < 5:
            pred_neu.extend(pred.tolist())
            continue

        dummies = pd.get_dummies(ind[mask], drop_first=True)
        X = pd.concat(
            [
                pd.Series(1.0, index=dummies.index, name="intercept"),
                dummies,
                beta[mask].rename("beta"),
                indbeta[mask].rename("indbeta"),
            ],
            axis=1,
        ).to_numpy()
        y = pred[mask]
        ww = w[mask]
        try:
            Xw = X * np.sqrt(ww)[:, None]
            yw = y * np.sqrt(ww)
            coef, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
            resid = y - X @ coef
        except Exception:
            resid = y

        pred_full = pred.copy()
        pred_full[mask.to_numpy()] = resid
        pred_neu.extend(pred_full.tolist())
    out["pred_neutral"] = pred_neu
    return out


def apply_pred_postprocess(
    df: pd.DataFrame,
    pred_col: str,
    pipeline: str,
    use_pred_z: bool,
    use_neutralize: bool,
    weight_col: str = "weight",
    date_col: str = "date",
    industry_col: str = "industry",
    beta_col: str = "beta",
    indbeta_col: str = "indbeta",
) -> pd.DataFrame:
    """Apply configurable postprocess pipeline to predictions."""
    out = df.copy()
    if pipeline == "none" or (not use_pred_z and not use_neutralize):
        out["pred_post"] = out[pred_col]
        return out
    cur_col = pred_col

    def _neutral():
        nonlocal out, cur_col
        out = pred_neutralize_by_date(out, cur_col, weight_col, date_col, industry_col, beta_col, indbeta_col)
        cur_col = "pred_neutral"

    def _zscore():
        nonlocal out, cur_col
        out = pred_cs_zscore_by_date(out, cur_col, weight_col, date_col)
        cur_col = "pred_z"

    if pipeline == "neutral_then_z":
        if use_neutralize:
            _neutral()
        if use_pred_z:
            _zscore()
    elif pipeline == "z_then_neutral":
        if use_pred_z:
            _zscore()
        if use_neutralize:
            _neutral()
    else:
        if use_neutralize:
            _neutral()
        if use_pred_z:
            _zscore()

    out["pred_post"] = out[cur_col]
    return out

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp


def _winsorize_group(args):
    g, cols, q_low, q_high, min_n = args
    res = g[cols].copy()
    for col in cols:
        if pd.api.types.is_bool_dtype(res[col]):
            continue
        x = res[col].astype(float)
        x_non = x.dropna()
        if len(x_non) < min_n:
            continue
        lo = x_non.quantile(q_low)
        hi = x_non.quantile(q_high)
        res[col] = x.clip(lo, hi)
    return res


def _zscore_group(args):
    g, cols, min_n, eps = args
    res = g[cols].copy()
    for col in cols:
        x = res[col]
        x_non = x.dropna()
        if len(x_non) < min_n:
            continue
        mu = x_non.mean()
        std = x_non.std()
        if not np.isfinite(std) or std <= eps:
            res[col] = 0.0
        else:
            res[col] = (x - mu) / std
    return res


def _neutralize_group(args):
    g, value_col, weight_col, industry_col, beta_col, indbeta_col = args
    v = g[value_col]
    w = g[weight_col]
    ind = g[industry_col]
    beta = g[beta_col]
    indbeta = g[indbeta_col]
    mask = (~v.isna()) & (~w.isna()) & (w > 0) & (~ind.isna()) & (~beta.isna()) & (~indbeta.isna())
    if mask.sum() < 5:
        return pd.Series(index=g.index, data=v)
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
    y = v[mask].to_numpy()
    ww = w[mask].to_numpy()
    try:
        Xw = X * np.sqrt(ww)[:, None]
        yw = y * np.sqrt(ww)
        coef, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
        resid = y - X @ coef
        res = pd.Series(index=g.index, data=v)
        res.loc[mask.index[mask]] = resid
        return res
    except Exception:
        return pd.Series(index=g.index, data=v)


def _csz_group(args):
    g, value_col, weight_col, eps = args
    v = g[value_col].to_numpy()
    w = g[weight_col].to_numpy()
    mask = (~np.isnan(v)) & (~np.isnan(w)) & (w > 0)
    if mask.sum() == 0:
        return pd.Series(index=g.index, data=0.0)
    ww = w[mask]
    vv = v[mask]
    mu = (ww * vv).sum() / ww.sum()
    var = (ww * (vv - mu) ** 2).sum() / ww.sum()
    std = np.sqrt(var)
    if not np.isfinite(std) or std <= eps:
        return pd.Series(index=g.index, data=0.0)
    return pd.Series(index=g.index, data=(v - mu) / std)


@dataclass
class EncoderState:
    """Store one-hot industry columns and feature ordering."""
    industry_cols: List[str]
    feature_cols: List[str]


def _get_feature_cols(
    df: pd.DataFrame,
    include_missing_features: bool = False,
    include_time_constant_features: bool = False,
    drop_redundant_features: bool = True,
) -> List[str]:
    """Infer feature columns by excluding meta and configured feature families."""
    exclude = {"id", "y", "y_raw", "y_label", "y_score", "weight", "DateTime", "date"}
    exclude_missing = set()
    if not include_missing_features:
        exclude_missing = {
            "r_missing_cnt",
            "dv_missing_cnt",
            "beta_isna",
            "indbeta_isna",
            "any_f_missing",
            "industry_isna",
        }
    exclude_time_constant = set()
    if not include_time_constant_features:
        exclude_time_constant = {c for c in df.columns if c.startswith(("dow_", "mon_", "mkt_"))}
    exclude_redundant = {"ret_last", "rev_30m"} if drop_redundant_features else set()

    return [
        c
        for c in df.columns
        if c not in exclude
        and c not in exclude_missing
        and c not in exclude_time_constant
        and c not in exclude_redundant
    ]


def _encode_industry(df: pd.DataFrame, industry_cols: List[str] | None = None) -> Tuple[pd.DataFrame, List[str]]:
    """One-hot encode industry with consistent column alignment."""
    if "industry" not in df.columns:
        return df.copy(), []
    dummies = pd.get_dummies(df["industry"], prefix="industry", dummy_na=True)
    if industry_cols is None:
        industry_cols = list(dummies.columns)
    else:
        for col in industry_cols:
            if col not in dummies.columns:
                dummies[col] = 0
        dummies = dummies[industry_cols]
    out = df.drop(columns=["industry"]).copy()
    out = pd.concat([out, dummies], axis=1)
    return out, industry_cols


def winsorize_by_date(
    df: pd.DataFrame,
    cols: List[str],
    q_low: float,
    q_high: float,
    min_n: int,
    date_col: str = "date",
    show_progress: bool = False,
    desc: str = "Winsorize",
    n_workers: int = 1,
) -> pd.DataFrame:
    """Winsorize columns per date using non-weighted quantiles."""
    out = df.copy()
    groups = list(out.groupby(date_col))

    it = groups
    if n_workers and n_workers > 1:
        with mp.Pool(processes=n_workers) as pool:
            mapped = pool.imap_unordered(
                _winsorize_group,
                [(g, cols, q_low, q_high, min_n) for _, g in groups],
                chunksize=1,
            )
            if show_progress:
                mapped = tqdm(mapped, total=len(groups), desc=desc, ncols=80)
            for res in mapped:
                for col in cols:
                    if not pd.api.types.is_float_dtype(out[col]):
                        out[col] = out[col].astype(float)
                out.loc[res.index, cols] = res[cols]
    else:
        if show_progress:
            it = tqdm(groups, total=len(groups), desc=desc, ncols=80)
        for _, g in it:
            res = _winsorize_group((g, cols, q_low, q_high, min_n))
            for col in cols:
                if not pd.api.types.is_float_dtype(out[col]):
                    out[col] = out[col].astype(float)
            out.loc[res.index, cols] = res[cols]
    return out


def zscore_by_date(
    df: pd.DataFrame,
    cols: List[str],
    min_n: int,
    date_col: str = "date",
    eps: float = 1e-12,
    show_progress: bool = False,
    desc: str = "Z-score",
    n_workers: int = 1,
) -> pd.DataFrame:
    """Standardize columns per date using non-weighted mean/std."""
    out = df.copy()
    groups = list(out.groupby(date_col))

    if n_workers and n_workers > 1:
        with mp.Pool(processes=n_workers) as pool:
            mapped = pool.imap_unordered(
                _zscore_group,
                [(g, cols, min_n, eps) for _, g in groups],
                chunksize=1,
            )
            if show_progress:
                mapped = tqdm(mapped, total=len(groups), desc=desc, ncols=80)
            for res in mapped:
                out.loc[res.index, cols] = res[cols]
    else:
        it = groups
        if show_progress:
            it = tqdm(groups, total=len(groups), desc=desc, ncols=80)
        for _, g in it:
            res = _zscore_group((g, cols, min_n, eps))
            out.loc[res.index, cols] = res[cols]
    return out


def neutralize_by_date(
    df: pd.DataFrame,
    value_col: str,
    weight_col: str,
    date_col: str,
    industry_col: str,
    beta_col: str,
    indbeta_col: str,
    show_progress: bool = False,
    desc: str = "Neutralize",
    n_workers: int = 1,
) -> pd.Series:
    """Neutralize value per date by WLS on industry/beta/indbeta."""
    out = df[value_col].copy()
    groups = list(df.groupby(date_col))

    if n_workers and n_workers > 1:
        with mp.Pool(processes=n_workers) as pool:
            mapped = pool.imap_unordered(
                _neutralize_group,
                [(g, value_col, weight_col, industry_col, beta_col, indbeta_col) for _, g in groups],
                chunksize=1,
            )
            if show_progress:
                mapped = tqdm(mapped, total=len(groups), desc=desc, ncols=80)
            for res in mapped:
                out.loc[res.index] = res
    else:
        it = groups
        if show_progress:
            it = tqdm(groups, total=len(groups), desc=desc, ncols=80)
        for _, g in it:
            res = _neutralize_group((g, value_col, weight_col, industry_col, beta_col, indbeta_col))
            out.loc[res.index] = res
    return out


def cs_zscore_by_date_weighted(
    df: pd.DataFrame,
    value_col: str,
    weight_col: str,
    date_col: str,
    eps: float = 1e-12,
    show_progress: bool = False,
    desc: str = "CS Z-score",
    n_workers: int = 1,
) -> pd.Series:
    """Weighted cross-sectional z-score per date."""
    out = df[value_col].copy()
    groups = list(df.groupby(date_col))

    if n_workers and n_workers > 1:
        with mp.Pool(processes=n_workers) as pool:
            mapped = pool.imap_unordered(
                _csz_group,
                [(g, value_col, weight_col, eps) for _, g in groups],
                chunksize=1,
            )
            if show_progress:
                mapped = tqdm(mapped, total=len(groups), desc=desc, ncols=80)
            for res in mapped:
                out.loc[res.index] = res
    else:
        it = groups
        if show_progress:
            it = tqdm(groups, total=len(groups), desc=desc, ncols=80)
        for _, g in it:
            res = _csz_group((g, value_col, weight_col, eps))
            out.loc[res.index] = res
    return out


def label_transform_by_date(
    df: pd.DataFrame,
    label_col: str,
    weight_col: str,
    date_col: str,
    mode: str,
    q_low: float,
    q_high: float,
    min_n: int,
    eps: float = 1e-12,
    show_progress: bool = False,
    desc_prefix: str = "Label",
    n_workers: int = 1,
) -> pd.Series:
    """Transform label per date with modes: raw | winsor_csz | neu_winsor_csz."""
    out = df[label_col].copy()
    if mode == "raw":
        return out

    tmp = df[[label_col, weight_col, date_col]].copy()
    if mode == "winsor_csz":
        tmp = winsorize_by_date(
            tmp,
            [label_col],
            q_low,
            q_high,
            min_n,
            date_col=date_col,
            show_progress=show_progress,
            desc=f"{desc_prefix}: winsorize",
            n_workers=n_workers,
        )
        return cs_zscore_by_date_weighted(
            tmp,
            label_col,
            weight_col,
            date_col,
            eps=eps,
            show_progress=show_progress,
            desc=f"{desc_prefix}: zscore",
            n_workers=n_workers,
        )

    if mode == "neu_winsor_csz":
        tmp_full = df[[label_col, weight_col, date_col, "industry", "beta", "indbeta"]].copy()
        tmp_full[label_col] = neutralize_by_date(
            tmp_full,
            value_col=label_col,
            weight_col=weight_col,
            date_col=date_col,
            industry_col="industry",
            beta_col="beta",
            indbeta_col="indbeta",
            show_progress=show_progress,
            desc=f"{desc_prefix}: neutralize",
            n_workers=n_workers,
        )
        tmp_full = winsorize_by_date(
            tmp_full,
            [label_col],
            q_low,
            q_high,
            min_n,
            date_col=date_col,
            show_progress=show_progress,
            desc=f"{desc_prefix}: winsorize",
            n_workers=n_workers,
        )
        return cs_zscore_by_date_weighted(
            tmp_full,
            label_col,
            weight_col,
            date_col,
            eps=eps,
            show_progress=show_progress,
            desc=f"{desc_prefix}: zscore",
            n_workers=n_workers,
        )

    return out


def fit_transform_train(
    df_train: pd.DataFrame,
    label_col: str = "y",
    include_missing_features: bool = False,
    include_time_constant_features: bool = False,
    drop_redundant_features: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str], EncoderState]:
    """Fit encoders on train set and return X/y/w."""
    df = df_train.copy()
    # Ensure no duplicated columns (can happen after merges or repeated label columns)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df[df[label_col].notna()]

    feature_cols = _get_feature_cols(
        df,
        include_missing_features=include_missing_features,
        include_time_constant_features=include_time_constant_features,
        drop_redundant_features=drop_redundant_features,
    )
    df_feat = df[feature_cols + [label_col, "weight"]].copy()

    df_feat, industry_cols = _encode_industry(df_feat)

    y = df_feat[label_col].to_numpy(dtype=np.float32)
    w = df_feat["weight"].to_numpy(dtype=np.float32)

    X = df_feat.drop(columns=[label_col, "weight"])
    feature_names = list(X.columns)
    state = EncoderState(industry_cols=industry_cols, feature_cols=feature_names)
    return X, y, w, feature_names, state


def transform_eval(
    df_eval: pd.DataFrame,
    state: EncoderState,
    label_col: str = "y",
    include_missing_features: bool = False,
    include_time_constant_features: bool = False,
    drop_redundant_features: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Transform eval set using saved encoder state."""
    df = df_eval.copy()
    # Ensure no duplicated columns
    df = df.loc[:, ~df.columns.duplicated()]
    df = df[df[label_col].notna()]

    feature_cols = _get_feature_cols(
        df,
        include_missing_features=include_missing_features,
        include_time_constant_features=include_time_constant_features,
        drop_redundant_features=drop_redundant_features,
    )
    df_feat = df[feature_cols + [label_col, "weight"]].copy()
    df_feat, _ = _encode_industry(df_feat, industry_cols=state.industry_cols)

    X = df_feat.drop(columns=[label_col, "weight"])
    for col in state.feature_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[state.feature_cols]

    y = df_feat[label_col].to_numpy(dtype=np.float32)
    w = df_feat["weight"].to_numpy(dtype=np.float32)
    return X, y, w

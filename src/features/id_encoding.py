from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class IdTEState:
    """State for id target encoding transform."""

    prior: float
    alpha: float
    stats: Dict[object, Tuple[float, int]]


def _date_block_folds(df: pd.DataFrame, date_col: str, folds: int) -> pd.Series:
    dates = sorted(df[date_col].dropna().unique().tolist())
    date_to_fold = {d: i % max(folds, 1) for i, d in enumerate(dates)}
    return df[date_col].map(date_to_fold).fillna(0).astype(int)


def fit_transform_id_te(
    train_df: pd.DataFrame,
    id_col: str = "id",
    y_col: str = "y_score",
    date_col: str = "date",
    folds: int = 5,
    alpha: float = 50.0,
) -> Tuple[pd.Series, IdTEState]:
    """OOF id target encoding for train split."""
    df = train_df[[id_col, y_col, date_col]].copy()
    prior = float(df[y_col].mean())
    fold_ids = _date_block_folds(df, date_col=date_col, folds=folds)
    te = pd.Series(index=df.index, dtype=float)

    for k in range(max(folds, 1)):
        tr = df[fold_ids != k]
        va = df[fold_ids == k]
        grp = tr.groupby(id_col)[y_col].agg(["sum", "count"])
        enc = (grp["sum"] + alpha * prior) / (grp["count"] + alpha)
        te.loc[va.index] = va[id_col].map(enc).fillna(prior).values

    grp_full = df.groupby(id_col)[y_col].agg(["sum", "count"])
    stats = {k: (float(v["sum"]), int(v["count"])) for k, v in grp_full.iterrows()}
    state = IdTEState(prior=prior, alpha=float(alpha), stats=stats)
    return te, state


def transform_id_te(df: pd.DataFrame, state: IdTEState, id_col: str = "id") -> pd.Series:
    """Apply id target encoding to new split using train aggregates."""
    vals = []
    a = state.alpha
    p = state.prior
    for k in df[id_col].values:
        if k in state.stats:
            s, c = state.stats[k]
            vals.append((s + a * p) / (c + a))
        else:
            vals.append(p)
    return pd.Series(vals, index=df.index, dtype=float)


from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class EncoderState:
    """Store one-hot industry columns and feature ordering."""
    industry_cols: List[str]
    feature_cols: List[str]


def _get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Infer feature columns by excluding meta/target columns."""
    exclude = {"id", "y", "weight", "DateTime", "date"}
    return [c for c in df.columns if c not in exclude]


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


def fit_transform_train(df_train: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str], EncoderState]:
    """Fit encoders on train set and return X/y/w."""
    df = df_train.copy()
    df = df[df["y"].notna()]

    feature_cols = _get_feature_cols(df)
    df_feat = df[feature_cols + ["y", "weight"]].copy()

    df_feat, industry_cols = _encode_industry(df_feat)

    y = df_feat["y"].to_numpy(dtype=np.float32)
    w = df_feat["weight"].to_numpy(dtype=np.float32)

    X = df_feat.drop(columns=["y", "weight"])
    feature_names = list(X.columns)
    state = EncoderState(industry_cols=industry_cols, feature_cols=feature_names)
    return X, y, w, feature_names, state


def transform_eval(df_eval: pd.DataFrame, state: EncoderState) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Transform eval set using saved encoder state."""
    df = df_eval.copy()
    df = df[df["y"].notna()]

    feature_cols = _get_feature_cols(df)
    df_feat = df[feature_cols + ["y", "weight"]].copy()
    df_feat, _ = _encode_industry(df_feat, industry_cols=state.industry_cols)

    X = df_feat.drop(columns=["y", "weight"])
    for col in state.feature_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[state.feature_cols]

    y = df_feat["y"].to_numpy(dtype=np.float32)
    w = df_feat["weight"].to_numpy(dtype=np.float32)
    return X, y, w

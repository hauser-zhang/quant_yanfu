from __future__ import annotations

import argparse
import json
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.dataset import load_range, list_daily_paths, load_paths
from src.data.prep import (
    fit_transform_train,
    transform_eval,
    winsorize_by_date,
    zscore_by_date,
    label_transform_by_date,
)
from src.eval.metrics import daily_weighted_corr, daily_weighted_mean_ic
from src.eval.postprocess import apply_pred_postprocess
from src.eda.factor_ic import compute_factor_table, default_factor_cols, plot_factor_ic_hist, plot_factor_ic_top20
from src.eda.data_profile import run_data_profile
from src.features.id_encoding import fit_transform_id_te, transform_id_te
from src.utils.exp_io import make_run_dirs, save_config, save_table
from src.analysis.ablation import build_ablation_variants, filter_variant_columns, make_ablation_tables
from src.split.forward_folds import generate_forward_folds
from src.models.registry import MODEL_SPECS
from src.models.train_sklearn import train_ridge, train_elasticnet, train_rf, train_extra_trees
from src.models.train_lgbm import train_lgbm
from src.models.train_torch import train_torch_model
from src.viz.plots import (
    plot_model_comparison,
    plot_daily_ic,
    plot_ablation,
    plot_feature_importance,
    plot_torch_training_history,
)


def _get_splits(split_mode: str) -> List[Dict[str, Tuple[str, str]]]:
    """Define date splits for training/validation/testing."""
    if split_mode == "simple":
        return [
            {
                "train": ("2016-01-01", "2018-12-31"),
                "valid": ("2019-01-01", "2019-12-31"),
                "test": ("2020-01-01", "2020-12-31"),
            }
        ]
    if split_mode == "forward":
        return [
            {
                "train": ("2016-01-01", "2017-12-31"),
                "valid": ("2018-01-01", "2018-12-31"),
                "test": ("2019-01-01", "2019-12-31"),
            },
            {
                "train": ("2016-01-01", "2018-12-31"),
                "valid": ("2019-01-01", "2019-12-31"),
                "test": ("2020-01-01", "2020-12-31"),
            },
            {
                "train": ("2016-01-01", "2019-12-31"),
                "valid": ("2020-01-01", "2020-12-31"),
                "test": ("2020-01-01", "2020-12-31"),
            },
        ]
    raise ValueError(f"Unknown split_mode: {split_mode}")


def _train_model(
    model_name: str,
    X_train,
    y_train,
    w_train,
    X_valid,
    y_valid,
    w_valid,
    gpu_id: int | None,
    n_jobs: int,
    params: dict | None = None,
):
    """Train a model and return (model, predict_fn)."""
    if model_name.startswith("torch_"):
        def _assert_finite_df(X, name: str):
            if hasattr(X, "fillna"):
                xf = X.fillna(0).to_numpy(dtype=np.float32)
                if np.isfinite(xf).all():
                    return
                bad = [c for c in X.columns if not np.isfinite(X[c].fillna(0).to_numpy(dtype=np.float32)).all()]
                raise ValueError(f"Non-finite values in {name} after fillna(0): {bad[:20]}")
        _assert_finite_df(X_train, "X_train")
        _assert_finite_df(X_valid, "X_valid")

    params = params or {}
    if model_name in {"torch_cnn", "torch_rnn", "torch_lstm", "torch_gru"} and "feature_names" not in params:
        if hasattr(X_train, "columns"):
            params = dict(params)
            params["feature_names"] = list(X_train.columns)
    if model_name == "ridge":
        return train_ridge(X_train, y_train, w_train, **params)
    if model_name == "elasticnet":
        return train_elasticnet(X_train, y_train, w_train, **params)
    if model_name == "rf":
        return train_rf(X_train, y_train, w_train, n_jobs=n_jobs, **params)
    if model_name == "extra_trees":
        return train_extra_trees(X_train, y_train, w_train, n_jobs=n_jobs, **params)
    if model_name == "lgbm":
        return train_lgbm(X_train, y_train, w_train, X_valid, y_valid, w_valid, num_threads=n_jobs, **params)
    if model_name == "torch_linear":
        device = f"cuda:{gpu_id}" if gpu_id is not None else None
        return train_torch_model(
            X_train, y_train, w_train, X_valid, y_valid, w_valid, model_type="linear", device=device, **params
        )
    if model_name == "torch_mlp":
        device = f"cuda:{gpu_id}" if gpu_id is not None else None
        return train_torch_model(
            X_train, y_train, w_train, X_valid, y_valid, w_valid, model_type="mlp", device=device, **params
        )
    if model_name == "torch_cnn":
        device = f"cuda:{gpu_id}" if gpu_id is not None else None
        return train_torch_model(
            X_train, y_train, w_train, X_valid, y_valid, w_valid, model_type="cnn", device=device, **params
        )
    if model_name == "torch_rnn":
        device = f"cuda:{gpu_id}" if gpu_id is not None else None
        return train_torch_model(
            X_train, y_train, w_train, X_valid, y_valid, w_valid, model_type="rnn", device=device, **params
        )
    if model_name == "torch_lstm":
        device = f"cuda:{gpu_id}" if gpu_id is not None else None
        return train_torch_model(
            X_train, y_train, w_train, X_valid, y_valid, w_valid, model_type="lstm", device=device, **params
        )
    if model_name == "torch_gru":
        device = f"cuda:{gpu_id}" if gpu_id is not None else None
        return train_torch_model(
            X_train, y_train, w_train, X_valid, y_valid, w_valid, model_type="gru", device=device, **params
        )
    if model_name == "catboost":
        try:
            from catboost import CatBoostRegressor
        except Exception as e:
            raise ImportError("catboost not installed") from e
        model = CatBoostRegressor(
            loss_function="RMSE",
            depth=params.get("depth", 6),
            learning_rate=params.get("learning_rate", 0.05),
            iterations=params.get("iterations", 500),
            verbose=False,
            random_seed=42,
            thread_count=n_jobs,
        )
        model.fit(X_train.fillna(0), y_train, sample_weight=w_train)
        return model, lambda Xnew: model.predict(Xnew.fillna(0))
    if model_name == "xgb":
        try:
            from xgboost import XGBRegressor
        except Exception as e:
            raise ImportError("xgboost not installed") from e
        model = XGBRegressor(
            n_estimators=params.get("n_estimators", 500),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            random_state=42,
            n_jobs=n_jobs,
        )
        model.fit(X_train.fillna(0), y_train, sample_weight=w_train)
        return model, lambda Xnew: model.predict(Xnew.fillna(0))
    raise ValueError(f"Unknown model: {model_name}")


def _iter_param_grid(grid: dict) -> List[dict]:
    """Expand a dict of lists into a list of param dicts."""
    if not grid:
        return [{}]
    keys = list(grid.keys())
    values = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]
    combos = []
    for vals in itertools.product(*values):
        combos.append(dict(zip(keys, vals)))
    return combos


def _load_param_grid(path_or_json: str | None) -> dict:
    if not path_or_json:
        return {}
    try:
        if Path(path_or_json).exists():
            return json.loads(Path(path_or_json).read_text())
    except Exception:
        pass
    try:
        return json.loads(path_or_json)
    except Exception:
        return {}


def _get_feature_groups(feature_names: List[str]) -> Dict[str, List[str]]:
    """Group features for ablation and interpretation."""
    r_feats = {
        "mom_2h",
        "mom_1d",
        "rev_30m",
        "ret_last",
        "vol_std",
        "absret_sum",
        "ret_early",
        "ret_late",
        "ret_late_minus_early",
    }
    dv_feats = {"dv_log_last", "dv_log_mean_prev", "dv_shock", "dv_log_sum"}
    pv_feats = {"pv_corr"}
    f_feats = {f"f_{i}" for i in range(10)}
    r_raw = {f"r_{i}" for i in range(20)}
    dv_raw = {f"dv_{i}" for i in range(20)}
    raw_all = set(r_raw) | set(dv_raw) | set(f_feats) | {"beta", "indbeta"}

    groups = {
        "G_r": [f for f in feature_names if f in r_feats],
        "G_dv": [f for f in feature_names if f in dv_feats],
        "G_pv": [f for f in feature_names if f in pv_feats],
        "G_r_raw": [f for f in feature_names if f in r_raw],
        "G_dv_raw": [f for f in feature_names if f in dv_raw],
        "G_f": [f for f in feature_names if f in f_feats],
        "G_risk": [f for f in feature_names if f in {"beta", "indbeta"} or f.startswith("industry_")],
        "G_raw_all": [f for f in feature_names if f in raw_all or f.startswith("industry_")],
    }
    return groups


def _param_hash(params: dict) -> str:
    """Stable short hash for parameter dicts."""
    s = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:8]


def _param_tag(params: dict, max_len: int = 80) -> str:
    """Create a readable, filesystem-safe tag from params."""
    if not params:
        return "default"
    items = [f"{k}={params[k]}" for k in sorted(params.keys())]
    tag = "_".join(items)
    tag = re.sub(r"[^A-Za-z0-9._-]+", "-", tag)
    if len(tag) > max_len:
        tag = f"{tag[:max_len]}_{_param_hash(params)}"
    return tag


def _apply_dv_log1p(df: pd.DataFrame, pattern: str = r"^dv_\d+$") -> pd.DataFrame:
    """Apply log1p to dv_* columns (clip to non-negative)."""
    cols = [c for c in df.columns if re.match(pattern, c)]
    if not cols:
        return df
    out = df.copy()
    for col in cols:
        out[col] = np.log1p(np.clip(out[col].astype(float), 0, None))
    return out


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time features from DateTime/date."""
    out = df.copy()
    if "DateTime" in out.columns:
        dt = pd.to_datetime(out["DateTime"], errors="coerce")
    else:
        dt = pd.to_datetime(out["date"], errors="coerce")
    dow = dt.dt.dayofweek.fillna(0).astype(int)
    mon = dt.dt.month.fillna(1).astype(int)
    out["dow_idx"] = dow
    out["mon_idx"] = mon
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    out["mon_sin"] = np.sin(2 * np.pi * (mon - 1) / 12.0)
    out["mon_cos"] = np.cos(2 * np.pi * (mon - 1) / 12.0)
    return out


def _weighted_mean_std(x: pd.Series, w: pd.Series) -> Tuple[float, float]:
    """Weighted mean/std with NaN and non-positive weight filtering."""
    mask = (~x.isna()) & (~w.isna()) & (w > 0)
    if mask.sum() == 0:
        return float("nan"), float("nan")
    xx = x[mask].astype(float).to_numpy()
    ww = w[mask].astype(float).to_numpy()
    wsum = ww.sum()
    if wsum <= 0:
        return float("nan"), float("nan")
    mu = (ww * xx).sum() / wsum
    var = (ww * (xx - mu) ** 2).sum() / wsum
    return float(mu), float(np.sqrt(var))


def _add_market_state_features(df: pd.DataFrame, use_r_cols: List[str], dv_col: str) -> pd.DataFrame:
    """Add market state features by date using weights."""
    out = df.copy()
    if "weight" not in out.columns or "date" not in out.columns:
        return out
    if not use_r_cols:
        return out
    r_src = out[use_r_cols].sum(axis=1) if len(use_r_cols) > 1 else out[use_r_cols[0]]
    dv_src = out[dv_col] if dv_col in out.columns else None
    rows = []
    for date, g in out.groupby("date"):
        w = g["weight"]
        mu_r, std_r = _weighted_mean_std(r_src.loc[g.index], w)
        if dv_src is not None:
            mu_dv, std_dv = _weighted_mean_std(dv_src.loc[g.index], w)
        else:
            mu_dv, std_dv = float("nan"), float("nan")
        rows.append(
            {
                "date": date,
                "mkt_r_mean": mu_r,
                "mkt_r_std": std_r,
                "mkt_dv_mean": mu_dv,
                "mkt_dv_std": std_dv,
            }
        )
    mkt_df = pd.DataFrame(rows)
    out = out.merge(mkt_df, on="date", how="left")
    return out


def _add_industry_state_features(df: pd.DataFrame, use_r_cols: List[str], dv_col: str) -> pd.DataFrame:
    """Add industry state features by date+industry using weights."""
    out = df.copy()
    if "industry" not in out.columns or "weight" not in out.columns or "date" not in out.columns:
        return out
    if not use_r_cols:
        return out
    r_src = out[use_r_cols].sum(axis=1) if len(use_r_cols) > 1 else out[use_r_cols[0]]
    dv_src = out[dv_col] if dv_col in out.columns else None
    rows = []
    for (date, ind), g in out.groupby(["date", "industry"]):
        w = g["weight"]
        mu_r, std_r = _weighted_mean_std(r_src.loc[g.index], w)
        if dv_src is not None:
            mu_dv, std_dv = _weighted_mean_std(dv_src.loc[g.index], w)
        else:
            mu_dv, std_dv = float("nan"), float("nan")
        rows.append(
            {
                "date": date,
                "industry": ind,
                "ind_r_mean": mu_r,
                "ind_r_std": std_r,
                "ind_dv_mean": mu_dv,
                "ind_dv_std": std_dv,
            }
        )
    ind_df = pd.DataFrame(rows)
    out = out.merge(ind_df, on=["date", "industry"], how="left")
    return out


def _dump_debug_snapshots(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
    run_dir: Path,
    fold_idx: int,
    debug_n_rows: int,
    debug_n_features: int,
):
    """Dump debug snapshots and stats after preprocessing."""
    out_dir = run_dir / "debug" / f"fold{fold_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cols = [
        "id",
        "date",
        "weight",
        "y_raw",
        "y_label",
        "industry",
        "beta",
        "indbeta",
        "r_0",
        "r_19",
        "dv_0",
        "dv_19",
        "f_0",
        "f_1",
    ]
    extra_prefixes = ("dow_", "mon_", "mkt_", "ind_")

    def _select_cols(df: pd.DataFrame) -> List[str]:
        cols = [c for c in base_cols if c in df.columns]
        state_cols = [c for c in df.columns if c.startswith(extra_prefixes)]
        cols.extend([c for c in state_cols if c not in cols])
        remaining = [c for c in df.columns if c not in cols]
        rng = np.random.RandomState(42)
        if debug_n_features > len(cols) and remaining:
            k = min(debug_n_features - len(cols), len(remaining))
            cols.extend(rng.choice(remaining, size=k, replace=False).tolist())
        return cols

    def _dump_head(df: pd.DataFrame, name: str):
        cols = _select_cols(df)
        df.head(debug_n_rows)[cols].to_csv(out_dir / f"{name}_head.csv", index=False)

    _dump_head(df_train, "train")
    _dump_head(df_valid, "valid")
    _dump_head(df_test, "test")

    def _stats(df: pd.DataFrame, cols: List[str]):
        stats = {}
        for c in cols:
            if c not in df.columns:
                continue
            s = df[c]
            stats[c] = {
                "mean": float(s.mean()),
                "std": float(s.std()),
                "p1": float(s.quantile(0.01)),
                "p99": float(s.quantile(0.99)),
                "missing_rate": float(s.isna().mean()),
            }
        return stats

    key_cols = _select_cols(df_train)
    (out_dir / "x_stats_train.json").write_text(json.dumps(_stats(df_train, key_cols), indent=2))
    (out_dir / "x_stats_valid.json").write_text(json.dumps(_stats(df_valid, key_cols), indent=2))


def _sample_debug_rows_by_date(df: pd.DataFrame, date_col: str = "date", n_per_date: int = 20) -> pd.DataFrame:
    """Sample earliest+latest date rows for lightweight preprocessing inspection."""
    if df.empty or date_col not in df.columns:
        return df.head(0).copy()
    dates = sorted(df[date_col].dropna().unique().tolist())
    if not dates:
        return df.head(0).copy()
    pick = [dates[0]]
    if len(dates) > 1:
        pick.append(dates[-1])
    parts = []
    for d in pick:
        g = df[df[date_col] == d]
        parts.append(g.head(n_per_date))
    return pd.concat(parts, ignore_index=True) if parts else df.head(0).copy()


def _save_preprocess_snapshot(df: pd.DataFrame, out_path: Path, n_per_date: int = 20) -> None:
    """Save compact stage snapshot for debug_preprocess."""
    snap = _sample_debug_rows_by_date(df, n_per_date=n_per_date)
    keep_cols = [
        c
        for c in [
            "id",
            "date",
            "weight",
            "y_raw",
            "y",
            "r_0",
            "dv_0",
            "beta",
            "f_0",
            "ind_r_mean",
            "ind_dv_mean",
        ]
        if c in snap.columns
    ]
    extra = [c for c in snap.columns if c.startswith(("ind_", "mkt_", "dow_", "mon_"))][:20]
    cols = keep_cols + [c for c in extra if c not in keep_cols]
    if not cols:
        cols = list(snap.columns[:40])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    snap[cols].to_csv(out_path, index=False)


def _save_zscore_sanity(df: pd.DataFrame, out_path: Path, date_col: str = "date") -> None:
    """Save per-date sanity stats after zscore for key columns."""
    key_cols = [c for c in ["r_0", "dv_0", "beta", "f_0"] if c in df.columns]
    ind_cols = [c for c in df.columns if c.startswith("ind_")]
    if ind_cols:
        key_cols.append(ind_cols[0])
    rows = []
    for d, g in df.groupby(date_col):
        for c in key_cols:
            x = g[c].dropna()
            rows.append(
                {
                    "date": d,
                    "feature": c,
                    "mean": float(x.mean()) if len(x) else np.nan,
                    "std": float(x.std()) if len(x) else np.nan,
                    "min": float(x.min()) if len(x) else np.nan,
                    "max": float(x.max()) if len(x) else np.nan,
                    "non_nan_count": int(len(x)),
                }
            )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def _sample_days_by_year(paths: List[Path], max_days_per_year: int, seed: int) -> List[Path]:
    """Sample a fixed number of days per year for quick demos."""
    by_year: Dict[str, List[Path]] = {}
    for p in paths:
        year = p.parts[-3]
        by_year.setdefault(year, []).append(p)
    rng = np.random.RandomState(seed)
    sampled = []
    for year, items in by_year.items():
        items_sorted = sorted(items)
        if max_days_per_year is None or max_days_per_year <= 0:
            sampled.extend(items_sorted)
            continue
        if len(items_sorted) <= max_days_per_year:
            sampled.extend(items_sorted)
        else:
            idx = rng.choice(len(items_sorted), size=max_days_per_year, replace=False)
            sampled.extend([items_sorted[i] for i in idx])
    return sorted(sampled)


def _train_eval_one(
    model_name: str,
    X_train,
    y_train,
    w_train,
    X_valid,
    y_valid,
    w_valid,
    X_test,
    y_test,
    w_test,
    meta_train,
    meta_valid,
    meta_test,
    gpu_id: int | None,
    n_jobs: int,
    save_preds: bool,
    run_dir: Path,
    fold_idx: int,
    postprocess_pipeline: str,
    use_pred_z: bool,
    use_neutralize: bool,
    params: dict | None = None,
    save_ic: bool = True,
    id_train=None,
    id_valid=None,
    id_test=None,
):
    """Train one model and evaluate on valid/test splits."""
    model, predict_fn = _train_model(
        model_name,
        X_train,
        y_train,
        w_train,
        X_valid,
        y_valid,
        w_valid,
        gpu_id,
        n_jobs,
        params=params,
    )
    try:
        pred_train = predict_fn(X_train, id_train)
        pred_valid = predict_fn(X_valid, id_valid)
        pred_test = predict_fn(X_test, id_test)
    except TypeError:
        pred_train = predict_fn(X_train)
        pred_valid = predict_fn(X_valid)
        pred_test = predict_fn(X_test)

    df_train_pred = meta_train.copy()
    df_valid_pred = meta_valid.copy()
    df_test_pred = meta_test.copy()
    df_train_pred["pred"] = pred_train
    df_valid_pred["pred"] = pred_valid
    df_test_pred["pred"] = pred_test

    df_train_post = apply_pred_postprocess(
        df_train_pred,
        pred_col="pred",
        pipeline=postprocess_pipeline,
        use_pred_z=use_pred_z,
        use_neutralize=use_neutralize,
    )
    df_valid_post = apply_pred_postprocess(
        df_valid_pred,
        pred_col="pred",
        pipeline=postprocess_pipeline,
        use_pred_z=use_pred_z,
        use_neutralize=use_neutralize,
    )
    df_test_post = apply_pred_postprocess(
        df_test_pred,
        pred_col="pred",
        pipeline=postprocess_pipeline,
        use_pred_z=use_pred_z,
        use_neutralize=use_neutralize,
    )

    train_score = daily_weighted_mean_ic(df_train_post, "pred_post", "y_score", "weight", "date")
    valid_score = daily_weighted_mean_ic(df_valid_post, "pred_post", "y_score", "weight", "date")
    test_score = daily_weighted_mean_ic(df_test_post, "pred_post", "y_score", "weight", "date")

    if save_preds:
        df_train_pred.to_csv(run_dir / "predictions" / f"{model_name}_fold{fold_idx}_train.csv", index=False)
        df_valid_pred.to_csv(run_dir / "predictions" / f"{model_name}_fold{fold_idx}_valid.csv", index=False)
        df_test_pred.to_csv(run_dir / "predictions" / f"{model_name}_fold{fold_idx}_test.csv", index=False)

    if save_ic:
        # Save daily IC series for this model (valid/test)
        ic_valid = daily_weighted_corr(df_valid_post, "pred_post", "y_score", "weight", "date")
        ic_test = daily_weighted_corr(df_test_post, "pred_post", "y_score", "weight", "date")
        ic_valid.to_csv(run_dir / "ic_series" / f"{model_name}_fold{fold_idx}_valid.csv", index=False)
        ic_test.to_csv(run_dir / "ic_series" / f"{model_name}_fold{fold_idx}_test.csv", index=False)
        plot_daily_ic(
            ic_valid,
            run_dir / "ic_series" / f"{model_name}_fold{fold_idx}_valid.png",
            title=f"Daily IC (valid) - {model_name}",
        )
        plot_daily_ic(
            ic_test,
            run_dir / "ic_series" / f"{model_name}_fold{fold_idx}_test.png",
            title=f"Daily IC (test) - {model_name}",
        )

    return {
        "fold": fold_idx,
        "model_name": model_name,
        "train_score": train_score,
        "valid_score": valid_score,
        "test_score": test_score,
    }


def _perm_importance(
    predict_fn,
    X_valid,
    meta_valid,
    n_rows: int,
    postprocess_pipeline: str,
    use_pred_z: bool,
    use_neutralize: bool,
    n_repeats: int = 3,
):
    """Permutation importance using drop in daily-weighted-mean IC score."""
    if len(X_valid) > n_rows:
        idx = np.random.RandomState(42).choice(len(X_valid), size=n_rows, replace=False)
        Xv = X_valid.iloc[idx].copy()
        mv = meta_valid.iloc[idx].reset_index(drop=True)
    else:
        Xv = X_valid.copy()
        mv = meta_valid.reset_index(drop=True)

    idv = mv["id"].to_numpy() if "id" in mv.columns else None
    try:
        base_pred = predict_fn(Xv, idv)
    except TypeError:
        base_pred = predict_fn(Xv)
    df_base = mv.copy()
    df_base["pred"] = base_pred
    df_base = apply_pred_postprocess(
        df_base,
        pred_col="pred",
        pipeline=postprocess_pipeline,
        use_pred_z=use_pred_z,
        use_neutralize=use_neutralize,
    )
    base = daily_weighted_mean_ic(df_base, "pred_post", "y_score", "weight", "date")

    rows = []
    for col in Xv.columns:
        deltas = []
        for r in range(max(1, n_repeats)):
            Xp = Xv.copy()
            Xp[col] = np.random.RandomState(42 + r).permutation(Xp[col].values)
            try:
                pred = predict_fn(Xp, idv)
            except TypeError:
                pred = predict_fn(Xp)
            dfp = mv.copy()
            dfp["pred"] = pred
            dfp = apply_pred_postprocess(
                dfp,
                pred_col="pred",
                pipeline=postprocess_pipeline,
                use_pred_z=use_pred_z,
                use_neutralize=use_neutralize,
            )
            score = daily_weighted_mean_ic(dfp, "pred_post", "y_score", "weight", "date")
            deltas.append(base - score)
        rows.append({"feature": col, "delta_ic": float(np.mean(deltas)), "std": float(np.std(deltas)), "n_repeats": int(max(1, n_repeats))})
    return pd.DataFrame(rows)


def main() -> None:
    """CLI entry for running experiments and evaluations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/project_5year", help="Root path of daily data folders.")
    parser.add_argument("--run_name", type=str, required=True, help="Experiment name for res/experiments/{run_name}.")
    parser.add_argument("--split_mode", type=str, default="simple", help="Data split: simple or forward.")
    parser.add_argument("--feature_set", type=str, default="raw_only", help="Feature scope: raw_only|raw_plus_econ|all_extended.")
    parser.add_argument("--forward_start_date", type=str, default="2016-01-01", help="Forward split start date.")
    parser.add_argument("--forward_end_date", type=str, default="2020-12-31", help="Forward split end date.")
    parser.add_argument("--forward_step_months", type=int, default=6, help="Forward fold step size in months.")
    parser.add_argument("--forward_val_months", type=int, default=6, help="Forward validation window in months.")
    parser.add_argument("--forward_test_months", type=int, default=6, help="Forward test window in months.")
    parser.add_argument("--forward_train_months", type=int, default=36, help="Forward train window in months.")
    parser.add_argument("--forward_mode", type=str, default="rolling", help="Forward mode: rolling|expanding.")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model list (e.g. lgbm,rf,torch_mlp).")
    parser.add_argument("--gpu_id", type=int, default=2, help="GPU id for torch models (None for CPU).")
    parser.add_argument("--sample_days_per_year", type=int, default=0, help="Sample N days per year for quick demo (0=full).")
    parser.add_argument("--sample_seed", type=int, default=42, help="Random seed for day sampling.")
    parser.add_argument("--parallel_models", type=int, default=1, help="Parallel model training workers.")
    parser.add_argument("--parallel_grid", type=int, default=1, help="Parallel workers for param grid evaluation.")
    parser.add_argument("--max_evals", type=int, default=0, help="Randomly sample at most N param combos (0=all).")
    parser.add_argument("--grid_seed", type=int, default=42, help="Random seed for param sampling.")
    parser.add_argument("--n_jobs", type=int, default=16, help="CPU threads per model (rf/extra/lgbm/xgb/catboost).")
    parser.add_argument("--label_mode", type=str, default="raw", help="Label mode: raw | winsor_csz | neu_winsor_csz.")
    parser.add_argument(
        "--run_factor_ic",
        "--run-factor-ic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run single-factor IC sanity check on train split.",
    )
    parser.add_argument(
        "--include_missing_features",
        "--include-missing-features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include missingness indicator/count features in model input.",
    )
    parser.add_argument(
        "--include_time_constant_features",
        "--include-time-constant-features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include dow_/mon_/mkt_ columns in model input.",
    )
    parser.add_argument(
        "--drop_redundant_features",
        "--drop-redundant-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop redundant features such as ret_last/rev_30m.",
    )
    parser.add_argument(
        "--use_id_embedding",
        "--use-id-embedding",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use torch id embedding.",
    )
    parser.add_argument("--id_emb_dim", type=int, default=8, help="Embedding dim for id embedding.")
    parser.add_argument("--id_dropout_p", type=float, default=0.1, help="Entity dropout probability for id embedding.")
    parser.add_argument(
        "--use_id_encoding",
        "--use-id-encoding",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use tabular id target encoding feature.",
    )
    parser.add_argument("--id_te_alpha", type=float, default=50.0, help="Smoothing alpha for id target encoding.")
    parser.add_argument("--id_te_folds", type=int, default=5, help="Fold count for OOF id target encoding.")
    parser.add_argument(
        "--dv_log1p",
        "--dv-log1p",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply log1p to dv_*.",
    )
    parser.add_argument(
        "--add_time_features",
        "--add-time-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add cyclical time features.",
    )
    parser.add_argument(
        "--add_market_state_features",
        "--add-market-state-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add market state features.",
    )
    parser.add_argument(
        "--add_industry_state_features",
        "--add-industry-state-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add industry state features.",
    )
    parser.add_argument("--x_winsorize_by_date", action="store_true", help="Feature winsorization by date (X only).")
    parser.add_argument("--x_zscore_by_date", action="store_true", help="Feature z-score by date (X only).")
    parser.add_argument("--zscore_f_features", action=argparse.BooleanOptionalAction, default=False, help="Whether to zscore f_* features by date.")
    parser.add_argument("--debug_preprocess", action=argparse.BooleanOptionalAction, default=False, help="Save stage-by-stage preprocessing debug artifacts.")
    parser.add_argument("--data_workers", type=int, default=4, help="Parallel workers for daily data loading.")
    parser.add_argument("--preprocess_workers", type=int, default=0, help="Parallel workers for preprocessing (0=use data_workers).")
    parser.add_argument("--q_low", type=float, default=0.01, help="Winsor lower quantile.")
    parser.add_argument("--q_high", type=float, default=0.99, help="Winsor upper quantile.")
    parser.add_argument("--min_n", type=int, default=50, help="Min samples per date for winsor/zscore.")
    parser.add_argument("--eps", type=float, default=1e-12, help="Epsilon for weighted z-score.")
    parser.add_argument("--postprocess_pipeline", type=str, default="none", help="Prediction postprocess: none|neutral_then_z|z_then_neutral.")
    parser.add_argument("--use_pred_z", action="store_true", help="Apply pred z-score in postprocess.")
    parser.add_argument("--use_neutralize", action="store_true", help="Apply pred neutralization in postprocess.")
    parser.add_argument("--use_feat", action="store_true", help="Use feature files data_matrix_feat.csv if available.")
    parser.add_argument("--save_preds", action="store_true", help="Save per-model predictions for valid/test.")
    parser.add_argument("--perm_rows", type=int, default=20000, help="Rows for permutation importance.")
    parser.add_argument(
        "--param_grid",
        type=str,
        default=None,
        help="JSON path or JSON string mapping model->param grid.",
    )
    parser.add_argument("--dump_debug_snapshots", action="store_true", help="Dump debug snapshots after preprocessing.")
    parser.add_argument("--debug_n_rows", type=int, default=2000, help="Rows to dump in debug snapshots.")
    parser.add_argument("--debug_n_features", type=int, default=60, help="Number of features to include in debug snapshots.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging and checks.")
    parser.add_argument(
        "--resume_completed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse per-model fold results if they already exist under tuning/.",
    )
    parser.add_argument("--optim", type=str, default="adamw", help="Torch optimizer: adamw|adam.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Torch learning rate default.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Torch weight decay default.")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="Torch grad clip norm (0 disables).")
    parser.add_argument("--best_metric", type=str, default="ic", help="Best-checkpoint metric for torch: ic|loss.")
    parser.add_argument("--early_stop_metric", type=str, default="ic", help="Early-stop metric for torch: ic|loss.")
    parser.add_argument("--early_stop_patience", type=int, default=6, help="Torch early stop patience.")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0, help="Min improvement for early stop and best checkpoint.")
    parser.add_argument("--scheduler_type", type=str, default="plateau_ic", help="Torch LR scheduler: none|plateau_ic|cosine.")
    parser.add_argument("--plateau_factor", type=float, default=0.8, help="ReduceLROnPlateau factor.")
    parser.add_argument("--plateau_patience", type=int, default=8, help="ReduceLROnPlateau patience.")
    parser.add_argument("--plateau_min_lr", type=float, default=1e-5, help="ReduceLROnPlateau min lr.")
    args = parser.parse_args()
    if args.label_mode not in {"raw", "winsor_csz", "neu_winsor_csz"}:
        raise ValueError("label_mode must be one of: raw | winsor_csz | neu_winsor_csz")
    if args.feature_set not in {"raw_only", "raw_plus_econ", "all_extended"}:
        raise ValueError("feature_set must be one of: raw_only | raw_plus_econ | all_extended")

    run_dir = Path("res/experiments") / args.run_name
    dirs = make_run_dirs(run_dir, enable_debug=args.debug or args.dump_debug_snapshots or args.debug_preprocess)
    (run_dir / "predictions").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)  # backward compatibility
    (run_dir / "ic_series").mkdir(exist_ok=True)
    (run_dir / "feature_importance").mkdir(exist_ok=True)  # backward compatibility
    (run_dir / "tuning").mkdir(exist_ok=True)
    (run_dir / "tuning" / "torch_loss").mkdir(parents=True, exist_ok=True)

    param_grid_all = _load_param_grid(args.param_grid)
    if param_grid_all:
        (run_dir / "tuning" / "param_grid.json").write_text(json.dumps(param_grid_all, indent=2))

    config = {
        "run_name": args.run_name,
        "split_mode": args.split_mode,
        "feature_set": args.feature_set,
        "forward_start_date": args.forward_start_date,
        "forward_end_date": args.forward_end_date,
        "forward_step_months": args.forward_step_months,
        "forward_val_months": args.forward_val_months,
        "forward_test_months": args.forward_test_months,
        "forward_train_months": args.forward_train_months,
        "forward_mode": args.forward_mode,
        "models": args.models,
        "use_feat": args.use_feat,
        "data_root": args.data_root,
        "gpu_id": args.gpu_id,
        "sample_days_per_year": args.sample_days_per_year,
        "sample_seed": args.sample_seed,
        "parallel_models": args.parallel_models,
        "parallel_grid": args.parallel_grid,
        "max_evals": args.max_evals,
        "grid_seed": args.grid_seed,
        "n_jobs": args.n_jobs,
        "label_mode": args.label_mode,
        "run_factor_ic": args.run_factor_ic,
        "include_missing_features": args.include_missing_features,
        "include_time_constant_features": args.include_time_constant_features,
        "exclude_time_constant_features": (not args.include_time_constant_features),
        "drop_redundant_features": args.drop_redundant_features,
        "use_id_embedding": args.use_id_embedding,
        "id_emb_dim": args.id_emb_dim,
        "id_dropout_p": args.id_dropout_p,
        "use_id_encoding": args.use_id_encoding,
        "id_te_alpha": args.id_te_alpha,
        "id_te_folds": args.id_te_folds,
        "dv_log1p": args.dv_log1p,
        "add_time_features": args.add_time_features,
        "add_market_state_features": args.add_market_state_features,
        "add_industry_state_features": args.add_industry_state_features,
        "x_winsorize_by_date": args.x_winsorize_by_date,
        "x_zscore_by_date": args.x_zscore_by_date,
        "zscore_f_features": args.zscore_f_features,
        "debug_preprocess": args.debug_preprocess,
        "data_workers": args.data_workers,
        "preprocess_workers": args.preprocess_workers,
        "q_low": args.q_low,
        "q_high": args.q_high,
        "min_n": args.min_n,
        "eps": args.eps,
        "postprocess_pipeline": args.postprocess_pipeline,
        "use_pred_z": args.use_pred_z,
        "use_neutralize": args.use_neutralize,
        "param_grid": args.param_grid,
        "dump_debug_snapshots": args.dump_debug_snapshots,
        "debug_n_rows": args.debug_n_rows,
        "debug_n_features": args.debug_n_features,
        "debug": args.debug,
        "resume_completed": args.resume_completed,
        "optim": args.optim,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip_norm": args.grad_clip_norm,
        "best_metric": args.best_metric,
        "early_stop_metric": args.early_stop_metric,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "scheduler_type": args.scheduler_type,
        "plateau_factor": args.plateau_factor,
        "plateau_patience": args.plateau_patience,
        "plateau_min_lr": args.plateau_min_lr,
    }
    save_config(run_dir, config)

    model_list = args.models.split(",") if args.models else list(MODEL_SPECS.keys())
    model_list = [m for m in model_list if m in MODEL_SPECS]
    if param_grid_all:
        param_grid_all = {k: v for k, v in param_grid_all.items() if k in model_list}

    metrics_rows = []
    pred_cache = {}
    if args.split_mode == "forward":
        splits = generate_forward_folds(
            start_date=args.forward_start_date,
            end_date=args.forward_end_date,
            train_months=args.forward_train_months,
            val_months=args.forward_val_months,
            test_months=args.forward_test_months,
            step_months=args.forward_step_months,
            mode=args.forward_mode,
        )
        if not splits:
            raise ValueError("No forward folds generated; check forward_* date/window settings.")
    else:
        splits = _get_splits(args.split_mode)
    config["splits"] = splits
    save_config(run_dir, config)

    for fold_idx, split in enumerate(tqdm(splits, desc="Folds", ncols=80)):
        print(f"[Step1] Load data (fold {fold_idx})")
        train_paths = list_daily_paths(args.data_root, split["train"][0], split["train"][1])
        valid_paths = list_daily_paths(args.data_root, split["valid"][0], split["valid"][1])
        test_paths = list_daily_paths(args.data_root, split["test"][0], split["test"][1])

        if args.sample_days_per_year and args.sample_days_per_year > 0:
            train_paths = _sample_days_by_year(train_paths, args.sample_days_per_year, args.sample_seed)
            valid_paths = _sample_days_by_year(valid_paths, args.sample_days_per_year, args.sample_seed + 1)
            test_paths = _sample_days_by_year(test_paths, args.sample_days_per_year, args.sample_seed + 2)

        df_train = load_paths(train_paths, use_feat=args.use_feat, n_workers=args.data_workers, show_progress=True, desc="Load train")
        df_valid = load_paths(valid_paths, use_feat=args.use_feat, n_workers=args.data_workers, show_progress=True, desc="Load valid")
        df_test = load_paths(test_paths, use_feat=args.use_feat, n_workers=args.data_workers, show_progress=True, desc="Load test")
        print(f"[Step1] Loaded train/valid/test (fold {fold_idx})")

        # Guard against duplicate columns in raw data
        dup_train = df_train.columns[df_train.columns.duplicated()].tolist()
        dup_valid = df_valid.columns[df_valid.columns.duplicated()].tolist()
        dup_test = df_test.columns[df_test.columns.duplicated()].tolist()
        if dup_train or dup_valid or dup_test:
            raise ValueError(f"Duplicate columns found. train={dup_train}, valid={dup_valid}, test={dup_test}")
        df_train = df_train.loc[:, ~df_train.columns.duplicated()]
        df_valid = df_valid.loc[:, ~df_valid.columns.duplicated()]
        df_test = df_test.loc[:, ~df_test.columns.duplicated()]

        if args.debug:
            print(f"[DEBUG] train rows={len(df_train)}, cols={df_train.shape[1]}")
            print(f"[DEBUG] valid rows={len(df_valid)}, cols={df_valid.shape[1]}")
            print(f"[DEBUG] test  rows={len(df_test)},  cols={df_test.shape[1]}")
            print("[DEBUG] first columns:", df_train.columns[:10].tolist())

        # Preserve raw labels for evaluation
        df_train["y_raw"] = df_train["y"]
        df_valid["y_raw"] = df_valid["y"]
        df_test["y_raw"] = df_test["y"]

        # Optional dv log1p transform (before X preprocessing)
        if args.dv_log1p:
            df_train = _apply_dv_log1p(df_train)
            df_valid = _apply_dv_log1p(df_valid)
            df_test = _apply_dv_log1p(df_test)
            if args.debug_preprocess:
                _save_preprocess_snapshot(
                    df_train,
                    run_dir / "debug" / "x_snapshot_after_log1p.csv",
                    n_per_date=20,
                )

        # Optional time/state features
        if args.add_time_features:
            df_train = _add_time_features(df_train)
            df_valid = _add_time_features(df_valid)
            df_test = _add_time_features(df_test)

        use_r_cols = [c for c in df_train.columns if c in {"r_0", "r_1", "r_2", "r_3"}]
        if not use_r_cols and "r_0" in df_train.columns:
            use_r_cols = ["r_0"]
        dv_col = "dv_0" if "dv_0" in df_train.columns else ""

        if args.add_market_state_features:
            df_train = _add_market_state_features(df_train, use_r_cols, dv_col)
            df_valid = _add_market_state_features(df_valid, use_r_cols, dv_col)
            df_test = _add_market_state_features(df_test, use_r_cols, dv_col)

        if args.add_industry_state_features:
            df_train = _add_industry_state_features(df_train, use_r_cols, dv_col)
            df_valid = _add_industry_state_features(df_valid, use_r_cols, dv_col)
            df_test = _add_industry_state_features(df_test, use_r_cols, dv_col)

        print(f"[Step2] Feature preprocessing (fold {fold_idx})")
        # Feature preprocessing (winsorize / zscore) on numeric columns
        exclude_cols = {"id", "y", "y_raw", "weight", "DateTime", "date", "industry"}
        num_cols = [
            c
            for c in df_train.columns
            if c not in exclude_cols
            and pd.api.types.is_numeric_dtype(df_train[c])
            and not pd.api.types.is_bool_dtype(df_train[c])
        ]
        # ind_* varies across stocks within day and should be standardized.
        # f_* is already normalized/slow-moving; exclude from per-date zscore by default.
        zscore_exclude_prefixes = ("dow_", "mon_", "mkt_")
        if not args.zscore_f_features:
            zscore_exclude_prefixes = zscore_exclude_prefixes + ("f_",)
        zscore_cols = [c for c in num_cols if not c.startswith(zscore_exclude_prefixes)]
        pp_workers = args.preprocess_workers or args.data_workers
        if args.x_winsorize_by_date:
            df_train = winsorize_by_date(
                df_train,
                num_cols,
                args.q_low,
                args.q_high,
                args.min_n,
                show_progress=True,
                desc="Winsorize X train",
                n_workers=pp_workers,
            )
            df_valid = winsorize_by_date(
                df_valid,
                num_cols,
                args.q_low,
                args.q_high,
                args.min_n,
                show_progress=True,
                desc="Winsorize X valid",
                n_workers=pp_workers,
            )
            df_test = winsorize_by_date(
                df_test,
                num_cols,
                args.q_low,
                args.q_high,
                args.min_n,
                show_progress=True,
                desc="Winsorize X test",
                n_workers=pp_workers,
            )
            if args.debug_preprocess:
                _save_preprocess_snapshot(
                    df_train,
                    run_dir / "debug" / "x_snapshot_after_winsor.csv",
                    n_per_date=20,
                )
        if args.x_zscore_by_date:
            df_train = zscore_by_date(
                df_train, zscore_cols, args.min_n, show_progress=True, desc="Z-score X train", n_workers=pp_workers
            )
            df_valid = zscore_by_date(
                df_valid, zscore_cols, args.min_n, show_progress=True, desc="Z-score X valid", n_workers=pp_workers
            )
            df_test = zscore_by_date(
                df_test, zscore_cols, args.min_n, show_progress=True, desc="Z-score X test", n_workers=pp_workers
            )
            if args.debug_preprocess:
                _save_preprocess_snapshot(
                    df_train,
                    run_dir / "debug" / "x_snapshot_after_zscore.csv",
                    n_per_date=20,
                )
                _save_zscore_sanity(df_train, run_dir / "debug" / "zscore_sanity_by_date.csv", date_col="date")
        print(f"[Step2] Feature preprocessing done (fold {fold_idx})")

        print(f"[Step2b] Label transform: {args.label_mode} (fold {fold_idx})")
        # Label preprocessing
        df_train["y_label"] = label_transform_by_date(
            df_train,
            "y",
            "weight",
            "date",
            args.label_mode,
            args.q_low,
            args.q_high,
            args.min_n,
            args.eps,
            show_progress=True,
            desc_prefix="Label train",
            n_workers=pp_workers,
        )
        df_valid["y_label"] = label_transform_by_date(
            df_valid,
            "y",
            "weight",
            "date",
            args.label_mode,
            args.q_low,
            args.q_high,
            args.min_n,
            args.eps,
            show_progress=True,
            desc_prefix="Label valid",
            n_workers=pp_workers,
        )
        df_test["y_label"] = label_transform_by_date(
            df_test,
            "y",
            "weight",
            "date",
            args.label_mode,
            args.q_low,
            args.q_high,
            args.min_n,
            args.eps,
            show_progress=True,
            desc_prefix="Label test",
            n_workers=pp_workers,
        )
        print(f"[Step2b] Label transform done (fold {fold_idx})")

        # y_score follows label mode by design
        df_train["y_score"] = df_train["y_label"]
        df_valid["y_score"] = df_valid["y_label"]
        df_test["y_score"] = df_test["y_label"]

        # Build tabular id target encoding (OOF on train), enabled by feature selection variants.
        te_train, te_state = fit_transform_id_te(
            df_train[df_train["y_label"].notna()],
            id_col="id",
            y_col="y_score",
            date_col="date",
            folds=args.id_te_folds,
            alpha=args.id_te_alpha,
        )
        df_train.loc[te_train.index, "id_te"] = te_train.values
        df_valid["id_te"] = transform_id_te(df_valid, te_state, id_col="id")
        df_test["id_te"] = transform_id_te(df_test, te_state, id_col="id")

        # Chapter-1 profiling artifacts from train split.
        run_data_profile(
            df_train,
            tables_dir=dirs["tables"],
            figures_dir=dirs["figures"],
            y_col="y_raw",
            weight_col="weight",
        )

        if args.run_factor_ic:
            feat_scan = default_factor_cols(df_train)
            factor_tbl = compute_factor_table(df_train, feat_scan, label_col="y_score", weight_col="weight", date_col="date")
            save_table(factor_tbl, dirs["factor_ic"] / "factor_ic_table.csv")
            plot_factor_ic_top20(factor_tbl, dirs["factor_ic"] / "factor_ic_top20.png")
            plot_factor_ic_hist(factor_tbl, dirs["factor_ic"] / "factor_ic_hist.png")

        if args.dump_debug_snapshots:
            _dump_debug_snapshots(
                df_train,
                df_valid,
                df_test,
                run_dir,
                fold_idx,
                args.debug_n_rows,
                args.debug_n_features,
            )

        if args.debug:
            def _stat(s):
                return {
                    "count": int(s.notna().sum()),
                    "mean": float(s.mean()),
                    "std": float(s.std()),
                    "min": float(s.min()),
                    "max": float(s.max()),
                }
            print("[DEBUG] label_mode:", args.label_mode)
            print("[DEBUG] y_label train stats:", _stat(df_train["y_label"]))
            print("[DEBUG] y_label valid stats:", _stat(df_valid["y_label"]))

        print(f"[Step3] Building train/valid/test matrices (fold {fold_idx})")
        df_train_main = df_train.copy()
        df_valid_main = df_valid.copy()
        df_test_main = df_test.copy()
        if not args.use_id_encoding:
            for dfx in (df_train_main, df_valid_main, df_test_main):
                if "id_te" in dfx.columns:
                    dfx.drop(columns=["id_te"], inplace=True)

        X_train, y_train, w_train, feature_names, state = fit_transform_train(
            df_train_main,
            label_col="y_label",
            include_missing_features=args.include_missing_features,
            include_time_constant_features=args.include_time_constant_features,
            drop_redundant_features=args.drop_redundant_features,
            feature_set=args.feature_set,
            use_id_encoding=args.use_id_encoding,
        )
        X_valid, y_valid, w_valid = transform_eval(
            df_valid_main,
            state,
            label_col="y_label",
            include_missing_features=args.include_missing_features,
            include_time_constant_features=args.include_time_constant_features,
            drop_redundant_features=args.drop_redundant_features,
            feature_set=args.feature_set,
            use_id_encoding=args.use_id_encoding,
        )
        X_test, y_test, w_test = transform_eval(
            df_test_main,
            state,
            label_col="y_label",
            include_missing_features=args.include_missing_features,
            include_time_constant_features=args.include_time_constant_features,
            drop_redundant_features=args.drop_redundant_features,
            feature_set=args.feature_set,
            use_id_encoding=args.use_id_encoding,
        )
        feature_groups = _get_feature_groups(feature_names)
        (dirs["tables"] / "feature_cols_used.txt").write_text("\n".join(feature_names) + "\n")
        pd.DataFrame({"feature": feature_names}).to_csv(dirs["tables"] / "feature_cols_used.csv", index=False)

        # Dump torch first batch if requested
        if args.dump_debug_snapshots and any(m.startswith("torch_") for m in model_list):
            dbg_dir = run_dir / "debug" / f"fold{fold_idx}"
            dbg_dir.mkdir(parents=True, exist_ok=True)
            n = min(args.debug_n_rows, len(X_train))
            Xb = X_train.head(n)
            yb = y_train[:n]
            wb = w_train[:n]
            np.savez(
                dbg_dir / "torch_first_batch.npz",
                X_batch=Xb.to_numpy(),
                y_batch=yb,
                w_batch=wb,
                feature_names=np.array(list(Xb.columns)),
            )
            if "r_0" in Xb.columns and "r_19" in Xb.columns and "dv_0" in Xb.columns and "dv_19" in Xb.columns:
                r_cols = [f"r_{i}" for i in range(20) if f"r_{i}" in Xb.columns]
                dv_cols = [f"dv_{i}" for i in range(20) if f"dv_{i}" in Xb.columns]
                if len(r_cols) == 20 and len(dv_cols) == 20:
                    np.savez(
                        dbg_dir / "torch_first_batch_seq.npz",
                        r_seq=Xb[r_cols].to_numpy(),
                        dv_seq=Xb[dv_cols].to_numpy(),
                    )
        print(f"[Step3] Matrices ready (fold {fold_idx})")

        if args.debug:
            print(f"[DEBUG] X_train shape={X_train.shape}, y_train shape={y_train.shape}")
            print(f"[DEBUG] X_valid shape={X_valid.shape}, y_valid shape={y_valid.shape}")
            print(f"[DEBUG] feature count={len(feature_names)}")
            print("[DEBUG] feature names:", feature_names)

        date_valid = df_valid.loc[df_valid["y_label"].notna(), "date"].reset_index(drop=True)
        date_test = df_test.loc[df_test["y_label"].notna(), "date"].reset_index(drop=True)

        meta_train = df_train.loc[
            df_train["y_label"].notna(),
            ["id", "date", "weight", "y_raw", "y_score", "industry", "beta", "indbeta"],
        ].reset_index(drop=True)
        meta_valid = df_valid.loc[
            df_valid["y_label"].notna(),
            ["id", "date", "weight", "y_raw", "y_score", "industry", "beta", "indbeta"],
        ].reset_index(drop=True)
        meta_test = df_test.loc[
            df_test["y_label"].notna(),
            ["id", "date", "weight", "y_raw", "y_score", "industry", "beta", "indbeta"],
        ].reset_index(drop=True)

        # id->idx mapping for optional torch id embedding (0 reserved for UNK).
        id_train_idx = id_valid_idx = id_test_idx = None
        if args.use_id_embedding:
            train_ids = meta_train["id"].astype(str)
            uniq = sorted(train_ids.unique().tolist())
            id_map = {k: i + 1 for i, k in enumerate(uniq)}
            id_train_idx = train_ids.map(id_map).fillna(0).astype(int).to_numpy()
            id_valid_idx = meta_valid["id"].astype(str).map(id_map).fillna(0).astype(int).to_numpy()
            id_test_idx = meta_test["id"].astype(str).map(id_map).fillna(0).astype(int).to_numpy()
            config["id_vocab_size"] = len(id_map) + 1
        valid_dates_arr = meta_valid["date"].to_numpy()

        # Save small random sample of train X/y for scale inspection (with id/date)
        sample_n = 1000
        df_train_f = df_train[df_train["y_label"].notna()].reset_index(drop=True)

        n_rows = len(X_train)
        rng = np.random.default_rng(42)
        if n_rows <= sample_n:
            sample_idx = np.arange(n_rows)
        else:
            sample_idx = rng.choice(n_rows, size=sample_n, replace=False)
        sample_idx = np.sort(sample_idx)  # keep row order

        dbg = X_train.iloc[sample_idx].copy()
        dbg["id"] = df_train_f["id"].iloc[sample_idx].to_numpy()
        dbg["date"] = df_train_f["date"].iloc[sample_idx].to_numpy()
        y_dbg = y_train[sample_idx]
        w_dbg = w_train[sample_idx]
        if hasattr(y_dbg, "ndim") and y_dbg.ndim > 1:
            y_dbg = y_dbg[:, 0]
        if hasattr(w_dbg, "ndim") and w_dbg.ndim > 1:
            w_dbg = w_dbg[:, 0]
        dbg["y"] = y_dbg
        dbg["weight"] = w_dbg
        dbg.to_csv(run_dir / "debug" / f"train_xy_sample_fold{fold_idx}.csv", index=False)

        best_params_by_model = {}

        def _run_one(mname: str):
            print(f"[Step4] Start model: {mname} (fold {fold_idx})")
            result_json = run_dir / "tuning" / f"{mname}_fold{fold_idx}_result.json"
            if args.resume_completed and result_json.exists():
                try:
                    cached = json.loads(result_json.read_text())
                    print(f"[Step4] Reuse cached model result: {mname} (fold {fold_idx})")
                    return cached
                except Exception:
                    if args.debug:
                        print(f"[DEBUG] Failed to load cached result for {mname} fold={fold_idx}, retrain.")
            Xtr_use, Xva_use, Xte_use = X_train, X_valid, X_test
            if mname in {"torch_cnn", "torch_rnn", "torch_lstm", "torch_gru"}:
                raw_cols = feature_groups.get("G_raw_all", [])
                if not raw_cols:
                    raise ValueError("G_raw_all is empty; cannot train sequence models")
                Xtr_use = X_train[raw_cols]
                Xva_use = X_valid[raw_cols]
                Xte_use = X_test[raw_cols]
            grid = param_grid_all.get(mname, {})
            param_list = _iter_param_grid(grid)
            if args.debug:
                print(f"[DEBUG] {mname} grid size={len(param_list)}")
            if args.max_evals and len(param_list) > args.max_evals:
                rng = np.random.RandomState(args.grid_seed)
                idx = rng.choice(len(param_list), size=args.max_evals, replace=False)
                param_list = [param_list[i] for i in idx]
                if args.debug:
                    print(f"[DEBUG] {mname} sampled grid size={len(param_list)}")
            grid_total = max(1, len(param_list))

            def _eval_params(param_idx: int, params: dict):
                train_params = dict(params)
                if mname in {"torch_linear", "torch_mlp", "torch_cnn", "torch_rnn", "torch_lstm", "torch_gru"}:
                    tag = _param_tag(params)
                    train_params["log_path"] = str(
                        run_dir / "tuning" / "torch_loss" / f"{mname}_fold{fold_idx}_{tag}.csv"
                    )
                    train_params["feature_names"] = list(Xtr_use.columns)
                    train_params["valid_dates"] = valid_dates_arr
                    train_params["use_id_embedding"] = args.use_id_embedding
                    train_params["id_emb_dim"] = args.id_emb_dim
                    train_params["id_dropout_p"] = args.id_dropout_p
                    train_params["id_train"] = id_train_idx
                    train_params["id_valid"] = id_valid_idx
                    train_params.setdefault("optim", args.optim)
                    train_params.setdefault("lr", args.lr)
                    train_params.setdefault("weight_decay", args.weight_decay)
                    train_params.setdefault("grad_clip_norm", args.grad_clip_norm)
                    train_params.setdefault("best_metric", args.best_metric)
                    train_params.setdefault("early_stop_metric", args.early_stop_metric)
                    train_params.setdefault("early_stop_patience", args.early_stop_patience)
                    train_params.setdefault("early_stop_min_delta", args.early_stop_min_delta)
                    train_params.setdefault("scheduler_type", args.scheduler_type)
                    train_params.setdefault("plateau_factor", args.plateau_factor)
                    train_params.setdefault("plateau_patience", args.plateau_patience)
                    train_params.setdefault("plateau_min_lr", args.plateau_min_lr)
                if mname in {"torch_cnn", "torch_rnn", "torch_lstm", "torch_gru"}:
                    train_params["feature_names"] = list(Xtr_use.columns)
                if mname.startswith("torch_"):
                    train_params["debug"] = args.debug
                    train_params["grid_idx"] = param_idx
                    train_params["grid_total"] = grid_total
                    if params:
                        train_params["param_tag"] = _param_tag(params, max_len=60)
                return _train_eval_one(
                    mname,
                    Xtr_use,
                    y_train,
                    w_train,
                    Xva_use,
                    y_valid,
                    w_valid,
                    Xte_use,
                    y_test,
                    w_test,
                    meta_train,
                    meta_valid,
                    meta_test,
                    args.gpu_id,
                    args.n_jobs,
                    save_preds=False,
                    run_dir=run_dir,
                    fold_idx=fold_idx,
                    postprocess_pipeline=args.postprocess_pipeline,
                    use_pred_z=args.use_pred_z,
                    use_neutralize=args.use_neutralize,
                    params=train_params,
                    save_ic=False,
                    id_train=id_train_idx,
                    id_valid=id_valid_idx,
                    id_test=id_test_idx,
                )

            if len(param_list) == 1:
                best_params = param_list[0]
                train_params = dict(best_params)
                if mname in {"torch_linear", "torch_mlp", "torch_cnn", "torch_rnn", "torch_lstm", "torch_gru"}:
                    tag = _param_tag(best_params)
                    train_params["log_path"] = str(
                        run_dir / "tuning" / "torch_loss" / f"{mname}_fold{fold_idx}_{tag}.csv"
                    )
                    train_params["feature_names"] = list(Xtr_use.columns)
                    train_params["valid_dates"] = valid_dates_arr
                    train_params["use_id_embedding"] = args.use_id_embedding
                    train_params["id_emb_dim"] = args.id_emb_dim
                    train_params["id_dropout_p"] = args.id_dropout_p
                    train_params["id_train"] = id_train_idx
                    train_params["id_valid"] = id_valid_idx
                    train_params.setdefault("optim", args.optim)
                    train_params.setdefault("lr", args.lr)
                    train_params.setdefault("weight_decay", args.weight_decay)
                    train_params.setdefault("grad_clip_norm", args.grad_clip_norm)
                    train_params.setdefault("best_metric", args.best_metric)
                    train_params.setdefault("early_stop_metric", args.early_stop_metric)
                    train_params.setdefault("early_stop_patience", args.early_stop_patience)
                    train_params.setdefault("early_stop_min_delta", args.early_stop_min_delta)
                    train_params.setdefault("scheduler_type", args.scheduler_type)
                    train_params.setdefault("plateau_factor", args.plateau_factor)
                    train_params.setdefault("plateau_patience", args.plateau_patience)
                    train_params.setdefault("plateau_min_lr", args.plateau_min_lr)
                if mname in {"torch_cnn", "torch_rnn", "torch_lstm", "torch_gru"}:
                    train_params["feature_names"] = list(Xtr_use.columns)
                if mname.startswith("torch_"):
                    train_params["debug"] = args.debug
                    train_params["grid_idx"] = 0
                    train_params["grid_total"] = 1
                    if best_params:
                        train_params["param_tag"] = _param_tag(best_params, max_len=60)
                res = _train_eval_one(
                    mname,
                    Xtr_use,
                    y_train,
                    w_train,
                    Xva_use,
                    y_valid,
                    w_valid,
                    Xte_use,
                    y_test,
                    w_test,
                    meta_train,
                    meta_valid,
                    meta_test,
                    args.gpu_id,
                    args.n_jobs,
                    save_preds=args.save_preds,
                    run_dir=run_dir,
                    fold_idx=fold_idx,
                    postprocess_pipeline=args.postprocess_pipeline,
                    use_pred_z=args.use_pred_z,
                    use_neutralize=args.use_neutralize,
                    params=train_params,
                    save_ic=True,
                    id_train=id_train_idx,
                    id_valid=id_valid_idx,
                    id_test=id_test_idx,
                )
                if best_params:
                    tune_df = pd.DataFrame([{**best_params, "valid_score": res["valid_score"], "test_score": res["test_score"]}])
                    tune_df.to_csv(run_dir / "tuning" / f"{mname}_fold{fold_idx}.csv", index=False)
                res["params"] = best_params
                try:
                    result_json.write_text(json.dumps(res, default=str))
                except Exception:
                    pass
                print(f"[Step4] Done model: {mname} (fold {fold_idx}) best_valid={res['valid_score']:.6f}")
                return res

            tune_rows = []
            best_res = None
            best_params = None
            if args.parallel_grid and args.parallel_grid > 1:
                # NOTE: If models themselves use multithreading, reduce --n_jobs to avoid oversubscription.
                with ThreadPoolExecutor(max_workers=args.parallel_grid) as ex:
                    futures = {ex.submit(_eval_params, i, p): p for i, p in enumerate(param_list)}
                    for fut in tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc=f"Grid {mname} (fold {fold_idx})",
                        ncols=80,
                        leave=False,
                    ):
                        params = futures[fut]
                        try:
                            res = fut.result()
                        except Exception:
                            continue
                        row = {"valid_score": res["valid_score"], "test_score": res["test_score"]}
                        row.update(params)
                        tune_rows.append(row)
                        if best_res is None or res["valid_score"] > best_res["valid_score"]:
                            best_res = res
                            best_params = params
            else:
                for i, params in enumerate(
                    tqdm(param_list, desc=f"Grid {mname} (fold {fold_idx})", ncols=80, leave=False)
                ):
                    res = _eval_params(i, params)
                    row = {"valid_score": res["valid_score"], "test_score": res["test_score"]}
                    row.update(params)
                    tune_rows.append(row)
                    if best_res is None or res["valid_score"] > best_res["valid_score"]:
                        best_res = res
                        best_params = params

            tune_df = pd.DataFrame(tune_rows)
            tune_df.to_csv(run_dir / "tuning" / f"{mname}_fold{fold_idx}.csv", index=False)

            res = _train_eval_one(
                mname,
                Xtr_use,
                y_train,
                w_train,
                Xva_use,
                y_valid,
                w_valid,
                Xte_use,
                y_test,
                w_test,
                meta_train,
                meta_valid,
                meta_test,
                args.gpu_id,
                args.n_jobs,
                save_preds=args.save_preds,
                run_dir=run_dir,
                fold_idx=fold_idx,
                postprocess_pipeline=args.postprocess_pipeline,
                use_pred_z=args.use_pred_z,
                use_neutralize=args.use_neutralize,
                params=best_params,
                save_ic=True,
                id_train=id_train_idx,
                id_valid=id_valid_idx,
                id_test=id_test_idx,
            )
            res["params"] = best_params or {}
            try:
                result_json.write_text(json.dumps(res, default=str))
            except Exception:
                pass
            print(f"[Step4] Done model: {mname} (fold {fold_idx}) best_valid={res['valid_score']:.6f}")
            return res

        if args.parallel_models > 1:
            with ThreadPoolExecutor(max_workers=args.parallel_models) as ex:
                futures = {ex.submit(_run_one, m): m for m in model_list if m in MODEL_SPECS}
                for fut in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Models (fold {fold_idx})",
                    ncols=80,
                ):
                    try:
                        res = fut.result()
                        res.update(
                            {
                                "fold_id": split.get("fold_id", fold_idx),
                                "train_start": split["train"][0],
                                "train_end": split["train"][1],
                                "val_start": split["valid"][0],
                                "val_end": split["valid"][1],
                                "test_start": split["test"][0],
                                "test_end": split["test"][1],
                                "feature_set": args.feature_set,
                                "label_mode": args.label_mode,
                                "split_mode": args.split_mode,
                                "n_train": int(len(X_train)),
                                "n_val": int(len(X_valid)),
                                "n_test": int(len(X_test)),
                            }
                        )
                        metrics_rows.append(res)
                        best_params_by_model[res["model_name"]] = res.get("params", {})
                    except Exception as e:
                        print(f"[WARN] Model failed: {futures[fut]} -> {e}")
                        continue
        else:
            for model_name in tqdm(model_list, desc=f"Models (fold {fold_idx})", ncols=80):
                if model_name not in MODEL_SPECS:
                    continue
                try:
                    res = _run_one(model_name)
                    res.update(
                        {
                            "fold_id": split.get("fold_id", fold_idx),
                            "train_start": split["train"][0],
                            "train_end": split["train"][1],
                            "val_start": split["valid"][0],
                            "val_end": split["valid"][1],
                            "test_start": split["test"][0],
                            "test_end": split["test"][1],
                            "feature_set": args.feature_set,
                            "label_mode": args.label_mode,
                            "split_mode": args.split_mode,
                            "n_train": int(len(X_train)),
                            "n_val": int(len(X_valid)),
                            "n_test": int(len(X_test)),
                        }
                    )
                    metrics_rows.append(res)
                    best_params_by_model[res["model_name"]] = res.get("params", {})
                except Exception as e:
                    print(f"[WARN] Model failed: {model_name} -> {e}")
                    continue

        # Cache data for retraining best model on this fold
        pred_cache[fold_idx] = {
            "feature_names": feature_names,
            "X_train": X_train,
            "y_train": y_train,
            "w_train": w_train,
            "meta_train": meta_train,
            "X_valid": X_valid,
            "y_valid": y_valid,
            "w_valid": w_valid,
            "meta_valid": meta_valid,
            "X_test": X_test,
            "y_test": y_test,
            "w_test": w_test,
            "meta_test": meta_test,
            "id_train_idx": id_train_idx,
            "id_valid_idx": id_valid_idx,
            "id_test_idx": id_test_idx,
            "df_train": df_train,
            "df_valid": df_valid,
            "df_test": df_test,
            "best_params_by_model": best_params_by_model,
        }

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(run_dir / "metrics_raw.csv", index=False)
    if args.split_mode == "forward" and not metrics_df.empty:
        fcols = [
            "fold_id",
            "train_start",
            "train_end",
            "val_start",
            "val_end",
            "test_start",
            "test_end",
            "model_name",
            "feature_set",
            "label_mode",
            "valid_score",
            "test_score",
            "n_train",
            "n_val",
            "n_test",
        ]
        forward_df = metrics_df[fcols].rename(
            columns={"valid_score": "valid_daily_ic", "test_score": "test_daily_ic"}
        )
        save_table(forward_df, dirs["tables"] / "metrics_forward_folds.csv")
        summary = (
            forward_df.groupby(["model_name", "feature_set", "label_mode"], as_index=False)
            .agg(
                mean_valid_ic=("valid_daily_ic", "mean"),
                mean_test_ic=("test_daily_ic", "mean"),
                std_valid_ic=("valid_daily_ic", "std"),
                std_test_ic=("test_daily_ic", "std"),
            )
        )
        summary["IR_test"] = summary["mean_test_ic"] / (summary["std_test_ic"] + 1e-12)
        save_table(summary, dirs["tables"] / "metrics_forward_summary.csv")
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(9, 5))
            for m, g in forward_df.groupby("model_name"):
                g = g.sort_values("fold_id")
                ax.plot(g["fold_id"], g["test_daily_ic"], marker="o", label=m)
            ax.set_xlabel("fold_id")
            ax.set_ylabel("test_daily_ic")
            ax.set_title("Forward Test IC by Fold")
            ax.legend(loc="best", fontsize=8)
            fig.savefig(dirs["figures"] / "forward_test_ic_by_fold.png", dpi=180, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 4))
            s2 = summary.sort_values("mean_test_ic", ascending=False)
            ax.bar(s2["model_name"], s2["mean_test_ic"], yerr=s2["std_test_ic"].fillna(0).to_numpy())
            ax.set_ylabel("mean_test_ic")
            ax.set_title("Forward Summary (mean±std)")
            ax.tick_params(axis="x", rotation=30)
            fig.savefig(dirs["figures"] / "forward_summary_bar.png", dpi=180, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            pass

    if metrics_df.empty:
        return

    metrics_agg = metrics_df.groupby("model_name").mean(numeric_only=True).reset_index()
    metrics_agg = metrics_agg.sort_values("valid_score", ascending=False)
    metrics_agg["run_name"] = args.run_name
    metrics_agg["split_mode"] = args.split_mode
    metrics_agg["label_mode"] = args.label_mode
    metrics_agg = metrics_agg.rename(
        columns={
            "train_score": "train_daily_ic",
            "valid_score": "valid_daily_ic",
            "test_score": "test_daily_ic",
        }
    )
    metrics_agg = metrics_agg[
        ["model_name", "split_mode", "label_mode", "train_daily_ic", "valid_daily_ic", "test_daily_ic", "run_name"]
    ]
    metrics_agg.to_csv(run_dir / "metrics.csv", index=False)
    save_table(metrics_agg, dirs["tables"] / "model_comparison.csv")
    save_table(
        metrics_agg[["model_name", "split_mode", "label_mode", "valid_daily_ic", "test_daily_ic"]],
        dirs["tables"] / "split_comparison.csv",
    )

    plot_model_comparison(
        metrics_agg.rename(
            columns={"train_daily_ic": "train_score", "valid_daily_ic": "valid_score", "test_daily_ic": "test_score"}
        ),
        run_dir / "plots" / "model_comparison.png",
    )
    plot_model_comparison(
        metrics_agg.rename(
            columns={"train_daily_ic": "train_score", "valid_daily_ic": "valid_score", "test_daily_ic": "test_score"}
        ),
        dirs["figures"] / "model_comparison_bar.png",
    )
    plot_model_comparison(
        metrics_agg.rename(
            columns={"train_daily_ic": "train_score", "valid_daily_ic": "valid_score", "test_daily_ic": "test_score"}
        ),
        dirs["figures"] / "split_comparison.png",
    )

    best = metrics_agg.iloc[0]["model_name"]
    print(f"[Step5] Best model selected: {best}")
    last_fold = len(splits) - 1
    best_data = pred_cache[last_fold]
    best_params = best_data.get("best_params_by_model", {}).get(best, {})
    Xtr_best = best_data["X_train"]
    Xva_best = best_data["X_valid"]
    Xte_best = best_data["X_test"]
    feat_names_best = best_data["feature_names"]
    if best in {"torch_cnn", "torch_rnn", "torch_lstm", "torch_gru"}:
        raw_cols = _get_feature_groups(best_data["feature_names"]).get("G_raw_all", [])
        if not raw_cols:
            raise ValueError("G_raw_all is empty; cannot train sequence models")
        Xtr_best = Xtr_best[raw_cols]
        Xva_best = Xva_best[raw_cols]
        Xte_best = Xte_best[raw_cols]
        feat_names_best = list(raw_cols)
        best_params = dict(best_params)
        best_params["feature_names"] = feat_names_best
        best_params["valid_dates"] = best_data["meta_valid"]["date"].to_numpy()
        best_params["id_train"] = best_data.get("id_train_idx")
        best_params["id_valid"] = best_data.get("id_valid_idx")
        best_params["use_id_embedding"] = args.use_id_embedding
        best_params["id_emb_dim"] = args.id_emb_dim
        best_params["id_dropout_p"] = args.id_dropout_p
        best_params.setdefault("optim", args.optim)
        best_params.setdefault("lr", args.lr)
        best_params.setdefault("weight_decay", args.weight_decay)
        best_params.setdefault("grad_clip_norm", args.grad_clip_norm)
        best_params.setdefault("best_metric", args.best_metric)
        best_params.setdefault("early_stop_metric", args.early_stop_metric)
        best_params.setdefault("early_stop_patience", args.early_stop_patience)
        best_params.setdefault("early_stop_min_delta", args.early_stop_min_delta)
        best_params.setdefault("scheduler_type", args.scheduler_type)
        best_params.setdefault("plateau_factor", args.plateau_factor)
        best_params.setdefault("plateau_patience", args.plateau_patience)
        best_params.setdefault("plateau_min_lr", args.plateau_min_lr)
    elif best in {"torch_linear", "torch_mlp"}:
        best_params = dict(best_params)
        best_params["valid_dates"] = best_data["meta_valid"]["date"].to_numpy()
        best_params["id_train"] = best_data.get("id_train_idx")
        best_params["id_valid"] = best_data.get("id_valid_idx")
        best_params["use_id_embedding"] = args.use_id_embedding
        best_params["id_emb_dim"] = args.id_emb_dim
        best_params["id_dropout_p"] = args.id_dropout_p
        best_params.setdefault("optim", args.optim)
        best_params.setdefault("lr", args.lr)
        best_params.setdefault("weight_decay", args.weight_decay)
        best_params.setdefault("grad_clip_norm", args.grad_clip_norm)
        best_params.setdefault("best_metric", args.best_metric)
        best_params.setdefault("early_stop_metric", args.early_stop_metric)
        best_params.setdefault("early_stop_patience", args.early_stop_patience)
        best_params.setdefault("early_stop_min_delta", args.early_stop_min_delta)
        best_params.setdefault("scheduler_type", args.scheduler_type)
        best_params.setdefault("plateau_factor", args.plateau_factor)
        best_params.setdefault("plateau_patience", args.plateau_patience)
        best_params.setdefault("plateau_min_lr", args.plateau_min_lr)
    best_model, best_predict_fn = _train_model(
        best,
        Xtr_best,
        best_data["y_train"],
        best_data["w_train"],
        Xva_best,
        best_data["y_valid"],
        best_data["w_valid"],
        args.gpu_id,
        args.n_jobs,
        params=best_params,
    )
    best_entry = {
        "model": best_model,
        "predict_fn": best_predict_fn,
        **best_data,
        "X_train": Xtr_best,
        "X_valid": Xva_best,
        "X_test": Xte_best,
        "feature_names": feat_names_best,
    }

    # Daily IC for best model (valid/test)
    try:
        pred_best_valid = best_entry["predict_fn"](best_entry["X_valid"], best_data.get("id_valid_idx"))
        pred_best_test = best_entry["predict_fn"](best_entry["X_test"], best_data.get("id_test_idx"))
    except TypeError:
        pred_best_valid = best_entry["predict_fn"](best_entry["X_valid"])
        pred_best_test = best_entry["predict_fn"](best_entry["X_test"])
    df_best_valid = best_entry["meta_valid"].copy()
    df_best_test = best_entry["meta_test"].copy()
    df_best_valid["pred"] = pred_best_valid
    df_best_test["pred"] = pred_best_test

    df_best_valid = apply_pred_postprocess(
        df_best_valid,
        pred_col="pred",
        pipeline=args.postprocess_pipeline,
        use_pred_z=args.use_pred_z,
        use_neutralize=args.use_neutralize,
    )
    df_best_test = apply_pred_postprocess(
        df_best_test,
        pred_col="pred",
        pipeline=args.postprocess_pipeline,
        use_pred_z=args.use_pred_z,
        use_neutralize=args.use_neutralize,
    )

    ic_best_valid = daily_weighted_corr(df_best_valid, "pred_post", "y_score", "weight", "date")
    ic_best_test = daily_weighted_corr(df_best_test, "pred_post", "y_score", "weight", "date")
    ic_best_valid.to_csv(run_dir / "ic_series" / "valid.csv", index=False)
    ic_best_test.to_csv(run_dir / "ic_series" / "test.csv", index=False)
    plot_daily_ic(ic_best_valid, run_dir / "plots" / "daily_ic_best_valid.png", title=f"Daily IC (valid) - {best}")
    plot_daily_ic(ic_best_valid, dirs["figures"] / "daily_ic_best_valid.png", title=f"Daily IC (valid) - {best}")
    plot_daily_ic(ic_best_test, dirs["figures"] / "daily_ic_best_test.png", title=f"Daily IC (test) - {best}")

    print("[Step6] Feature ablation start")
    seq_model = best in {"torch_cnn", "torch_rnn", "torch_lstm", "torch_gru"}
    seq_required = {f"r_{i}" for i in range(20)} | {f"dv_{i}" for i in range(20)}
    if args.feature_set == "raw_only":
        # raw_only already excludes engineered/extended feature families; skip drop/add-one ablation.
        variant_order = ["RAW_ONLY_BASE"]
        variant_cfg = {"RAW_ONLY_BASE": {"use_id": False, "use_missing": False}}
    elif args.feature_set == "raw_plus_econ":
        variant_order = ["BASE", "-IND", "-ECON", "-BETA"]
        variant_cfg = {
            "BASE": {"use_id": False, "use_missing": False},
            "-IND": {"use_id": False, "use_missing": False},
            "-ECON": {"use_id": False, "use_missing": False},
            "-BETA": {"use_id": False, "use_missing": False},
        }
    else:
        variant_order = ["CORE", "CORE+ID", "CORE+MISSING", "CORE+ID+MISSING", "FULL", "FULL-ID", "-MISSING", "-IND", "-ECON", "-BETA"]
        variant_cfg = build_ablation_variants()
    ablation_rows = []

    for vname in variant_order:
        cfg_v = variant_cfg[vname]
        use_missing_v = cfg_v["use_missing"]
        use_id_v = cfg_v["use_id"]
        use_id_emb_v = bool(use_id_v and best.startswith("torch"))
        use_id_enc_v = bool(use_id_v and (not best.startswith("torch")))

        dftr_v = best_data["df_train"].copy()
        dfva_v = best_data["df_valid"].copy()
        dfte_v = best_data["df_test"].copy()
        if not use_id_enc_v:
            for dfx in (dftr_v, dfva_v, dfte_v):
                if "id_te" in dfx.columns:
                    dfx.drop(columns=["id_te"], inplace=True)

        Xtr_v, ytr_v, wtr_v, feat_v, st_v = fit_transform_train(
            dftr_v,
            label_col="y_label",
            include_missing_features=use_missing_v,
            include_time_constant_features=args.include_time_constant_features,
            drop_redundant_features=args.drop_redundant_features,
            feature_set=args.feature_set,
            use_id_encoding=use_id_enc_v,
        )
        Xva_v, yva_v, wva_v = transform_eval(
            dfva_v,
            st_v,
            label_col="y_label",
            include_missing_features=use_missing_v,
            include_time_constant_features=args.include_time_constant_features,
            drop_redundant_features=args.drop_redundant_features,
            feature_set=args.feature_set,
            use_id_encoding=use_id_enc_v,
        )
        Xte_v, yte_v, wte_v = transform_eval(
            dfte_v,
            st_v,
            label_col="y_label",
            include_missing_features=use_missing_v,
            include_time_constant_features=args.include_time_constant_features,
            drop_redundant_features=args.drop_redundant_features,
            feature_set=args.feature_set,
            use_id_encoding=use_id_enc_v,
        )

        keep_cols = filter_variant_columns(list(feat_v), vname, include_missing_features=use_missing_v)
        if seq_model and not seq_required.issubset(set(keep_cols)):
            if args.debug:
                print(f"[DEBUG] Skip variant {vname}: missing required seq cols.")
            continue
        Xtr_v = Xtr_v[keep_cols]
        Xva_v = Xva_v[keep_cols]
        Xte_v = Xte_v[keep_cols]

        meta_tr_v = dftr_v.loc[dftr_v["y_label"].notna(), ["id", "date", "weight", "y_score"]].reset_index(drop=True)
        meta_va_v = dfva_v.loc[dfva_v["y_label"].notna(), ["id", "date", "weight", "y_score"]].reset_index(drop=True)
        meta_te_v = dfte_v.loc[dfte_v["y_label"].notna(), ["id", "date", "weight", "y_score"]].reset_index(drop=True)

        id_tr_v = id_va_v = id_te_v = None
        if use_id_emb_v:
            id_map_v = {k: i + 1 for i, k in enumerate(sorted(meta_tr_v["id"].astype(str).unique().tolist()))}
            id_tr_v = meta_tr_v["id"].astype(str).map(id_map_v).fillna(0).astype(int).to_numpy()
            id_va_v = meta_va_v["id"].astype(str).map(id_map_v).fillna(0).astype(int).to_numpy()
            id_te_v = meta_te_v["id"].astype(str).map(id_map_v).fillna(0).astype(int).to_numpy()

        p_v = dict(best_params)
        if best.startswith("torch"):
            p_v["feature_names"] = list(Xtr_v.columns)
            p_v["valid_dates"] = meta_va_v["date"].to_numpy()
            p_v["use_id_embedding"] = use_id_emb_v
            p_v["id_emb_dim"] = args.id_emb_dim
            p_v["id_dropout_p"] = args.id_dropout_p
            p_v["id_train"] = id_tr_v
            p_v["id_valid"] = id_va_v

        m_v, pred_v = _train_model(
            best, Xtr_v, ytr_v, wtr_v, Xva_v, yva_v, wva_v, args.gpu_id, args.n_jobs, params=p_v
        )
        try:
            pred_va = pred_v(Xva_v, id_va_v)
            pred_te = pred_v(Xte_v, id_te_v)
        except TypeError:
            pred_va = pred_v(Xva_v)
            pred_te = pred_v(Xte_v)

        dfpv = meta_va_v.copy()
        dfpt = meta_te_v.copy()
        dfpv["pred"] = pred_va
        dfpt["pred"] = pred_te
        dfpv = apply_pred_postprocess(
            dfpv, pred_col="pred", pipeline=args.postprocess_pipeline, use_pred_z=args.use_pred_z, use_neutralize=args.use_neutralize
        )
        dfpt = apply_pred_postprocess(
            dfpt, pred_col="pred", pipeline=args.postprocess_pipeline, use_pred_z=args.use_pred_z, use_neutralize=args.use_neutralize
        )
        valid_sc = daily_weighted_mean_ic(dfpv, "pred_post", "y_score", "weight", "date")
        test_sc = daily_weighted_mean_ic(dfpt, "pred_post", "y_score", "weight", "date")
        ablation_rows.append(
            {
                "variant_name": vname,
                "model_name": best,
                "split_mode": args.split_mode,
                "label_mode": args.label_mode,
                "valid_score": valid_sc,
                "test_score": test_sc,
                "n_features": len(keep_cols),
            }
        )
        (dirs["ablation"] / f"variant_{vname}_feature_cols.txt").write_text("\n".join(keep_cols) + "\n")

    add_df, drop_df = make_ablation_tables(ablation_rows)
    if args.feature_set == "raw_only":
        # Keep output files reproducible while clearly indicating ablation is intentionally skipped.
        base_df = pd.DataFrame(ablation_rows)
        save_table(base_df, dirs["ablation"] / "ablation_addone.csv")
        save_table(base_df, dirs["ablation"] / "ablation_dropone.csv")
    else:
        save_table(add_df, dirs["ablation"] / "ablation_addone.csv")
        save_table(drop_df, dirs["ablation"] / "ablation_dropone.csv")
    if not drop_df.empty:
        plot_ablation(
            drop_df.rename(columns={"variant_name": "setting", "valid_score": "valid_score"}),
            run_dir / "ablation" / "ablation.png",
        )
    print("[Step6] Feature ablation done")

    print("[Step7] Feature importance start")
    # Feature importance
    fi_rows = []
    model = best_entry["model"]
    feat_for_best = list(best_entry["X_valid"].columns)
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        fi_rows = [{"feature": f, "importance": float(v)} for f, v in zip(feat_for_best, fi)]
    elif hasattr(model, "booster_"):
        try:
            booster = model.booster_
            names = list(booster.feature_name()) if hasattr(booster, "feature_name") else feat_for_best
            gain = booster.feature_importance(importance_type="gain")
            split = booster.feature_importance(importance_type="split")
            fi_rows = [{"feature": f, "gain": float(g), "split_count": float(s)} for f, g, s in zip(names, gain, split)]
        except Exception:
            fi_rows = []

    if fi_rows:
        fi_df = pd.DataFrame(fi_rows)
        if "gain" in fi_df.columns:
            fi_df = fi_df.sort_values("gain", ascending=False)
            save_table(fi_df, dirs["importance"] / "lgbm_gain_importance.csv")
            plot_feature_importance(
                fi_df.rename(columns={"gain": "importance"})[["feature", "importance"]],
                dirs["figures"] / "lgbm_gain_top30.png",
            )
        else:
            fi_df = fi_df.sort_values("importance", ascending=False)
            save_table(fi_df, dirs["importance"] / "builtin_importance.csv")
            plot_feature_importance(fi_df, dirs["figures"] / "builtin_importance_top30.png")

    # Permutation importance
    perm_df = _perm_importance(
        best_entry["predict_fn"],
        best_entry["X_valid"],
        best_entry["meta_valid"],
        n_rows=args.perm_rows,
        postprocess_pipeline=args.postprocess_pipeline,
        use_pred_z=args.use_pred_z,
        use_neutralize=args.use_neutralize,
        n_repeats=3,
    )
    save_table(perm_df, dirs["importance"] / "permutation_importance.csv")
    plot_feature_importance(
        perm_df.rename(columns={"delta_ic": "importance"})[["feature", "importance"]],
        dirs["figures"] / "permutation_top30.png",
    )
    print("[Step7] Feature importance done")

    print("[Step8] Torch history plots start")
    torch_hist_dir = run_dir / "tuning" / "torch_loss"
    torch_fig_dir = dirs["figures"] / "torch_history"
    torch_fig_dir.mkdir(parents=True, exist_ok=True)
    if torch_hist_dir.exists():
        for csv_path in sorted(torch_hist_dir.glob("*.csv")):
            try:
                hist_df = pd.read_csv(csv_path)
                out_png = torch_fig_dir / f"{csv_path.stem}.png"
                plot_torch_training_history(hist_df, out_png, title=csv_path.stem)
            except Exception as e:
                if args.debug:
                    print(f"[DEBUG] Skip torch history plot {csv_path.name}: {e}")
    print("[Step8] Torch history plots done")

    # Optional SHAP
    try:
        import shap

        if hasattr(model, "predict"):
            X_sample = best_entry["X_valid"].sample(min(5000, len(best_entry["X_valid"])), random_state=42)
            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample)
            vals = np.abs(shap_values.values).mean(axis=0)
            shap_df = pd.DataFrame({"feature": X_sample.columns, "importance": vals})
            shap_df.to_csv(run_dir / "feature_importance" / "shap.csv", index=False)
            plot_feature_importance(shap_df, run_dir / "feature_importance" / "shap.png")
    except Exception:
        pass

    save_config(run_dir, config)


if __name__ == "__main__":
    main()

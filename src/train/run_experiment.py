from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.dataset import load_range, list_daily_paths, load_paths
from src.data.prep import fit_transform_train, transform_eval
from src.eval.metrics import weighted_corr, daily_weighted_corr
from src.eval.postprocess import cs_zscore_by_date
from src.models.registry import MODEL_SPECS
from src.models.train_sklearn import train_ridge, train_elasticnet, train_rf, train_extra_trees
from src.models.train_lgbm import train_lgbm
from src.models.train_torch import train_torch_linear, train_torch_mlp
from src.viz.plots import plot_model_comparison, plot_daily_ic, plot_ablation, plot_feature_importance


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
):
    """Train a model and return (model, predict_fn)."""
    if model_name == "ridge":
        return train_ridge(X_train, y_train, w_train)
    if model_name == "elasticnet":
        return train_elasticnet(X_train, y_train, w_train)
    if model_name == "rf":
        return train_rf(X_train, y_train, w_train, n_jobs=n_jobs)
    if model_name == "extra_trees":
        return train_extra_trees(X_train, y_train, w_train, n_jobs=n_jobs)
    if model_name == "lgbm":
        return train_lgbm(X_train, y_train, w_train, X_valid, y_valid, w_valid, num_threads=n_jobs)
    if model_name == "torch_linear":
        device = f"cuda:{gpu_id}" if gpu_id is not None else None
        return train_torch_linear(X_train, y_train, w_train, X_valid, y_valid, w_valid, device=device)
    if model_name == "torch_mlp":
        device = f"cuda:{gpu_id}" if gpu_id is not None else None
        return train_torch_mlp(X_train, y_train, w_train, X_valid, y_valid, w_valid, device=device)
    if model_name == "catboost":
        try:
            from catboost import CatBoostRegressor
        except Exception as e:
            raise ImportError("catboost not installed") from e
        model = CatBoostRegressor(
            loss_function="RMSE",
            depth=6,
            learning_rate=0.05,
            iterations=500,
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
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=n_jobs,
        )
        model.fit(X_train.fillna(0), y_train, sample_weight=w_train)
        return model, lambda Xnew: model.predict(Xnew.fillna(0))
    raise ValueError(f"Unknown model: {model_name}")


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
        "r_missing_cnt",
    }
    dv_feats = {"dv_log_last", "dv_log_mean_prev", "dv_shock", "dv_log_sum", "dv_missing_cnt"}
    pv_feats = {"pv_corr"}
    f_feats = {f"f_{i}" for i in range(10)}
    missing_feats = {"beta_isna", "indbeta_isna", "any_f_missing", "industry_isna"}

    groups = {
        "G_r": [f for f in feature_names if f in r_feats],
        "G_dv": [f for f in feature_names if f in dv_feats],
        "G_pv": [f for f in feature_names if f in pv_feats],
        "G_f": [f for f in feature_names if f in f_feats],
        "G_missing": [f for f in feature_names if f in missing_feats],
        "G_risk": [f for f in feature_names if f in {"beta", "indbeta"} or f.startswith("industry_")],
    }
    return groups


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
    date_valid,
    date_test,
    gpu_id: int | None,
    n_jobs: int,
    save_preds: bool,
    run_dir: Path,
    fold_idx: int,
):
    """Train one model and evaluate on valid/test splits."""
    model, predict_fn = _train_model(
        model_name, X_train, y_train, w_train, X_valid, y_valid, w_valid, gpu_id, n_jobs
    )
    pred_valid = predict_fn(X_valid)
    pred_test = predict_fn(X_test)

    df_valid_pred = pd.DataFrame({"date": date_valid, "y": y_valid, "weight": w_valid, "pred": pred_valid})
    df_test_pred = pd.DataFrame({"date": date_test, "y": y_test, "weight": w_test, "pred": pred_test})

    valid_corr_raw = weighted_corr(y_valid, pred_valid, w_valid)
    test_corr_raw = weighted_corr(y_test, pred_test, w_test)

    df_valid_z = cs_zscore_by_date(df_valid_pred, "pred", "weight", "date")
    df_test_z = cs_zscore_by_date(df_test_pred, "pred", "weight", "date")

    daily_ic_valid = daily_weighted_corr(df_valid_z, "pred_z", "y", "weight", "date")
    daily_ic_test = daily_weighted_corr(df_test_z, "pred_z", "y", "weight", "date")

    valid_corr_z = weighted_corr(
        df_valid_z["y"].to_numpy(), df_valid_z["pred_z"].to_numpy(), df_valid_z["weight"].to_numpy()
    )
    test_corr_z = weighted_corr(
        df_test_z["y"].to_numpy(), df_test_z["pred_z"].to_numpy(), df_test_z["weight"].to_numpy()
    )

    valid_corr_z_daily_mean = daily_ic_valid["corr"].mean() if not daily_ic_valid.empty else float("nan")
    test_corr_z_daily_mean = daily_ic_test["corr"].mean() if not daily_ic_test.empty else float("nan")

    if save_preds:
        df_valid_pred.to_csv(run_dir / "predictions" / f"{model_name}_fold{fold_idx}_valid.csv", index=False)
        df_test_pred.to_csv(run_dir / "predictions" / f"{model_name}_fold{fold_idx}_test.csv", index=False)

    # Save daily IC series for this model (valid/test)
    ic_valid = daily_weighted_corr(df_valid_pred, "pred", "y", "weight", "date")
    ic_test = daily_weighted_corr(df_test_pred, "pred", "y", "weight", "date")
    ic_valid.to_csv(run_dir / "plots" / "daily_ic" / f"{model_name}_fold{fold_idx}_valid.csv", index=False)
    ic_test.to_csv(run_dir / "plots" / "daily_ic" / f"{model_name}_fold{fold_idx}_test.csv", index=False)
    plot_daily_ic(
        ic_valid,
        run_dir / "plots" / "daily_ic" / f"{model_name}_fold{fold_idx}_valid.png",
        title=f"Daily IC (valid) - {model_name}",
    )
    plot_daily_ic(
        ic_test,
        run_dir / "plots" / "daily_ic" / f"{model_name}_fold{fold_idx}_test.png",
        title=f"Daily IC (test) - {model_name}",
    )

    return {
        "fold": fold_idx,
        "model": model_name,
        "valid_corr_raw": valid_corr_raw,
        "valid_corr_z": valid_corr_z,
        "valid_corr_z_daily_mean": valid_corr_z_daily_mean,
        "test_corr_raw": test_corr_raw,
        "test_corr_z": test_corr_z,
        "test_corr_z_daily_mean": test_corr_z_daily_mean,
    }


def _perm_importance(predict_fn, X_valid, y_valid, w_valid, date_valid, n_rows: int = 20000):
    if len(X_valid) > n_rows:
        idx = np.random.RandomState(42).choice(len(X_valid), size=n_rows, replace=False)
        Xv = X_valid.iloc[idx].copy()
        yv = y_valid[idx]
        wv = w_valid[idx]
        dv = date_valid.iloc[idx].reset_index(drop=True)
    else:
        Xv = X_valid.copy()
        yv = y_valid
        wv = w_valid
        dv = date_valid.reset_index(drop=True)

    base_pred = predict_fn(Xv)
    df_base = pd.DataFrame({"date": dv, "y": yv, "weight": wv, "pred": base_pred})
    df_base = cs_zscore_by_date(df_base, "pred", "weight", "date")
    base = weighted_corr(df_base["y"].to_numpy(), df_base["pred_z"].to_numpy(), df_base["weight"].to_numpy())

    rows = []
    for col in Xv.columns:
        Xp = Xv.copy()
        Xp[col] = np.random.RandomState(42).permutation(Xp[col].values)
        pred = predict_fn(Xp)
        dfp = pd.DataFrame({"date": dv, "y": yv, "weight": wv, "pred": pred})
        dfp = cs_zscore_by_date(dfp, "pred", "weight", "date")
        score = weighted_corr(dfp["y"].to_numpy(), dfp["pred_z"].to_numpy(), dfp["weight"].to_numpy())
        rows.append({"feature": col, "importance": base - score})
    return pd.DataFrame(rows)


def main() -> None:
    """CLI entry for running experiments and evaluations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/project_5year")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--split_mode", type=str, default="simple")
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--gpu_id", type=int, default=2)
    parser.add_argument("--sample_days_per_year", type=int, default=0)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--parallel_models", type=int, default=1)
    parser.add_argument("--n_jobs", type=int, default=16)
    parser.add_argument("--use_feat", action="store_true")
    parser.add_argument("--save_preds", action="store_true")
    parser.add_argument("--perm_rows", type=int, default=20000)
    args = parser.parse_args()

    run_dir = Path("res/experiments") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    (run_dir / "plots" / "daily_ic").mkdir(exist_ok=True)
    (run_dir / "ablation").mkdir(exist_ok=True)
    (run_dir / "feature_importance").mkdir(exist_ok=True)

    config = {
        "run_name": args.run_name,
        "split_mode": args.split_mode,
        "models": args.models,
        "use_feat": args.use_feat,
        "data_root": args.data_root,
        "gpu_id": args.gpu_id,
        "sample_days_per_year": args.sample_days_per_year,
        "sample_seed": args.sample_seed,
        "parallel_models": args.parallel_models,
        "n_jobs": args.n_jobs,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    model_list = args.models.split(",") if args.models else list(MODEL_SPECS.keys())

    metrics_rows = []
    pred_cache = {}
    splits = _get_splits(args.split_mode)

    for fold_idx, split in enumerate(tqdm(splits, desc="Folds", ncols=80)):
        train_paths = list_daily_paths(args.data_root, split["train"][0], split["train"][1])
        valid_paths = list_daily_paths(args.data_root, split["valid"][0], split["valid"][1])
        test_paths = list_daily_paths(args.data_root, split["test"][0], split["test"][1])

        if args.sample_days_per_year and args.sample_days_per_year > 0:
            train_paths = _sample_days_by_year(train_paths, args.sample_days_per_year, args.sample_seed)
            valid_paths = _sample_days_by_year(valid_paths, args.sample_days_per_year, args.sample_seed + 1)
            test_paths = _sample_days_by_year(test_paths, args.sample_days_per_year, args.sample_seed + 2)

        df_train = load_paths(train_paths, use_feat=args.use_feat)
        df_valid = load_paths(valid_paths, use_feat=args.use_feat)
        df_test = load_paths(test_paths, use_feat=args.use_feat)

        X_train, y_train, w_train, feature_names, state = fit_transform_train(df_train)
        X_valid, y_valid, w_valid = transform_eval(df_valid, state)
        X_test, y_test, w_test = transform_eval(df_test, state)

        date_valid = df_valid.loc[df_valid["y"].notna(), "date"].reset_index(drop=True)
        date_test = df_test.loc[df_test["y"].notna(), "date"].reset_index(drop=True)

        def _run_one(mname: str):
            return _train_eval_one(
                mname,
                X_train,
                y_train,
                w_train,
                X_valid,
                y_valid,
                w_valid,
                X_test,
                y_test,
                w_test,
                date_valid,
                date_test,
                args.gpu_id,
                args.n_jobs,
                args.save_preds,
                run_dir,
                fold_idx,
            )

        if args.parallel_models > 1:
            with ThreadPoolExecutor(max_workers=args.parallel_models) as ex:
                futures = {
                    ex.submit(_run_one, m): m for m in model_list if m in MODEL_SPECS
                }
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Models (fold {fold_idx})", ncols=80):
                    try:
                        res = fut.result()
                        metrics_rows.append(res)
                    except Exception:
                        continue
        else:
            for model_name in tqdm(model_list, desc=f"Models (fold {fold_idx})", ncols=80):
                if model_name not in MODEL_SPECS:
                    continue
                try:
                    res = _run_one(model_name)
                    metrics_rows.append(res)
                except Exception:
                    continue

        # Cache data for retraining best model on this fold
        pred_cache[fold_idx] = {
            "feature_names": feature_names,
            "X_train": X_train,
            "y_train": y_train,
            "w_train": w_train,
            "X_valid": X_valid,
            "y_valid": y_valid,
            "w_valid": w_valid,
            "date_valid": date_valid,
        }

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(run_dir / "metrics_raw.csv", index=False)

    if metrics_df.empty:
        return

    metrics_agg = metrics_df.groupby("model").mean(numeric_only=True).reset_index()
    metrics_agg = metrics_agg.sort_values("valid_corr_z", ascending=False)
    metrics_agg.to_csv(run_dir / "metrics.csv", index=False)

    plot_model_comparison(metrics_agg, run_dir / "plots" / "model_comparison.png")

    best = metrics_agg.iloc[0]["model"]
    last_fold = len(splits) - 1
    best_data = pred_cache[last_fold]
    best_model, best_predict_fn = _train_model(
        best,
        best_data["X_train"],
        best_data["y_train"],
        best_data["w_train"],
        best_data["X_valid"],
        best_data["y_valid"],
        best_data["w_valid"],
        args.gpu_id,
        args.n_jobs,
    )
    best_entry = {
        "model": best_model,
        "predict_fn": best_predict_fn,
        **best_data,
    }

    # Daily IC for best model (valid)
    ic_best_valid = daily_weighted_corr(best_entry["valid"], "pred", "y", "weight", "date")
    ic_best_valid.to_csv(run_dir / "plots" / "daily_ic_best_valid.csv", index=False)
    plot_daily_ic(
        ic_best_valid,
        run_dir / "plots" / "daily_ic_best_valid.png",
        title=f"Daily IC (valid) - {best}",
    )

    # Feature ablation
    groups = _get_feature_groups(best_entry["feature_names"])
    ablation_rows = []
    for gname, gcols in groups.items():
        if not gcols:
            continue
        Xtr_g = best_entry["X_train"][gcols]
        Xva_g = best_entry["X_valid"][gcols]
        model_g, pred_fn_g = _train_model(
            best,
            Xtr_g,
            best_entry["y_train"],
            best_entry["w_train"],
            Xva_g,
            best_entry["y_valid"],
            best_entry["w_valid"],
            args.gpu_id,
        )
        pred = pred_fn_g(Xva_g)
        dfp = pd.DataFrame(
            {
                "date": best_entry["date_valid"],
                "y": best_entry["y_valid"],
                "weight": best_entry["w_valid"],
                "pred": pred,
            }
        )
        dfp = cs_zscore_by_date(dfp, "pred", "weight", "date")
        corr_z = weighted_corr(dfp["y"].to_numpy(), dfp["pred_z"].to_numpy(), dfp["weight"].to_numpy())
        ablation_rows.append({"setting": f"only_{gname}", "valid_corr_z": corr_z})

    for gname, gcols in groups.items():
        if not gcols:
            continue
        keep_cols = [c for c in best_entry["feature_names"] if c not in gcols]
        Xtr_g = best_entry["X_train"][keep_cols]
        Xva_g = best_entry["X_valid"][keep_cols]
        model_g, pred_fn_g = _train_model(
            best,
            Xtr_g,
            best_entry["y_train"],
            best_entry["w_train"],
            Xva_g,
            best_entry["y_valid"],
            best_entry["w_valid"],
            args.gpu_id,
        )
        pred = pred_fn_g(Xva_g)
        dfp = pd.DataFrame(
            {
                "date": best_entry["date_valid"],
                "y": best_entry["y_valid"],
                "weight": best_entry["w_valid"],
                "pred": pred,
            }
        )
        dfp = cs_zscore_by_date(dfp, "pred", "weight", "date")
        corr_z = weighted_corr(dfp["y"].to_numpy(), dfp["pred_z"].to_numpy(), dfp["weight"].to_numpy())
        ablation_rows.append({"setting": f"drop_{gname}", "valid_corr_z": corr_z})

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(run_dir / "ablation" / "ablation.csv", index=False)
    plot_ablation(ablation_df, run_dir / "ablation" / "ablation.png")

    # Feature importance
    fi_rows = []
    model = best_entry["model"]
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        fi_rows = [{"feature": f, "importance": float(v)} for f, v in zip(feature_names, fi)]
    elif hasattr(model, "booster_"):
        try:
            fi = model.booster_.feature_importance(importance_type="gain")
            fi_rows = [{"feature": f, "importance": float(v)} for f, v in zip(feature_names, fi)]
        except Exception:
            fi_rows = []

    if fi_rows:
        fi_df = pd.DataFrame(fi_rows).sort_values("importance", ascending=False)
        fi_df.to_csv(run_dir / "feature_importance" / "builtin.csv", index=False)
        plot_feature_importance(fi_df, run_dir / "feature_importance" / "builtin.png")

    # Permutation importance
    perm_df = _perm_importance(
        best_entry["predict_fn"],
        best_entry["X_valid"],
        best_entry["y_valid"],
        best_entry["w_valid"],
        best_entry["date_valid"],
        n_rows=args.perm_rows,
    )
    perm_df.to_csv(run_dir / "feature_importance" / "permutation.csv", index=False)
    plot_feature_importance(perm_df, run_dir / "feature_importance" / "permutation.png")

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


if __name__ == "__main__":
    main()

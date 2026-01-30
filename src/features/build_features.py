import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def _nansum_allnan(arr: np.ndarray) -> np.ndarray:
    """Row-wise nansum; all-NaN rows return NaN."""
    out = np.nansum(arr, axis=1)
    all_nan = np.all(np.isnan(arr), axis=1)
    out = out.astype(float)
    out[all_nan] = np.nan
    return out


def _nanmean_allnan(arr: np.ndarray) -> np.ndarray:
    """Row-wise nanmean; all-NaN rows return NaN."""
    cnt = np.sum(~np.isnan(arr), axis=1)
    s = np.nansum(arr, axis=1)
    out = np.full(s.shape, np.nan, dtype=float)
    mask = cnt > 0
    out[mask] = s[mask] / cnt[mask]
    return out


def _nanstd_allnan(arr: np.ndarray) -> np.ndarray:
    """Row-wise nanstd; all-NaN rows return NaN."""
    cnt = np.sum(~np.isnan(arr), axis=1)
    mean = _nanmean_allnan(arr)
    diff = arr - mean[:, None]
    var = _nansum_allnan(diff ** 2) / np.where(cnt > 0, cnt, 1)
    out = np.sqrt(var)
    out[cnt == 0] = np.nan
    return out


def _row_corr(x: np.ndarray, y: np.ndarray, min_pairs: int = 3) -> np.ndarray:
    """Row-wise correlation between x and y with NaN-safe pairs."""
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        xi = x[i]
        yi = y[i]
        mask = ~np.isnan(xi) & ~np.isnan(yi)
        if mask.sum() < min_pairs:
            continue
        xv = xi[mask]
        yv = yi[mask]
        xv = xv - xv.mean()
        yv = yv - yv.mean()
        denom = np.sqrt((xv ** 2).sum() * (yv ** 2).sum())
        if denom <= 0:
            out[i] = np.nan
        else:
            out[i] = (xv * yv).sum() / denom
    return out


def _resolve_daily_file(day_folder: Path) -> Optional[Path]:
    """Resolve feature/raw daily file path for a given day folder."""
    csv_path = day_folder / "data_matrix.csv"
    if csv_path.exists():
        return csv_path
    gz_path = day_folder / "data_matrix.csv.gz"
    if gz_path.exists():
        return gz_path
    return None


def _stack_cols(df: pd.DataFrame, prefix: str, n: int) -> np.ndarray:
    """Stack prefix_i columns into a fixed-width matrix with NaN fill."""
    arr = np.full((len(df), n), np.nan, dtype=float)
    for i in range(n):
        col = f"{prefix}_{i}"
        if col in df.columns:
            arr[:, i] = df[col].to_numpy(dtype=float)
    return arr


def _build_features_for_file(input_path: Path, output_path: Path) -> Tuple[int, int]:
    """Build feature CSV for a single daily file."""
    df = pd.read_csv(input_path, compression="infer")

    f_cols = [f"f_{i}" for i in range(10) if f"f_{i}" in df.columns]

    r = _stack_cols(df, "r", 20)
    dv = _stack_cols(df, "dv", 20)

    # Returns features
    df["mom_2h"] = _nansum_allnan(r[:, 0:4])
    df["mom_1d"] = _nansum_allnan(r[:, 0:20])
    df["rev_30m"] = -r[:, 0]
    df["ret_last"] = r[:, 0]
    df["vol_std"] = _nanstd_allnan(r[:, 0:20])
    df["absret_sum"] = _nansum_allnan(np.abs(r[:, 0:20]))
    df["ret_early"] = _nansum_allnan(r[:, 16:20])
    df["ret_late"] = _nansum_allnan(r[:, 0:4])
    df["ret_late_minus_early"] = df["ret_late"] - df["ret_early"]
    df["r_missing_cnt"] = np.isnan(r[:, 0:20]).sum(axis=1)

    # Dollar volume features
    dv_log = np.log1p(dv)
    df["dv_log_last"] = np.log1p(dv[:, 0])
    df["dv_log_mean_prev"] = _nanmean_allnan(dv_log[:, 1:20])
    df["dv_shock"] = df["dv_log_last"] - df["dv_log_mean_prev"]
    df["dv_log_sum"] = _nansum_allnan(dv_log[:, 0:20])
    df["dv_missing_cnt"] = np.isnan(dv[:, 0:20]).sum(axis=1)

    # Price-volume correlation
    df["pv_corr"] = _row_corr(r[:, 0:20], dv_log[:, 0:20])

    # Missing indicators
    df["beta_isna"] = df["beta"].isna() if "beta" in df.columns else True
    df["indbeta_isna"] = df["indbeta"].isna() if "indbeta" in df.columns else True
    df["any_f_missing"] = df[f_cols].isna().any(axis=1) if f_cols else True
    df["industry_isna"] = df["industry"].isna() if "industry" in df.columns else True

    df.to_csv(output_path, index=False)
    return len(df), len(df.columns)


def list_day_folders(data_root: Path, start_date: Optional[str], end_date: Optional[str]) -> List[Path]:
    """List day folders within optional date range."""
    if start_date and end_date:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        days = []
        cur = start
        while cur <= end:
            days.append(cur)
            cur += pd.Timedelta(days=1)
        folders = []
        for d in days:
            path = data_root / d.strftime("%Y") / d.strftime("%m") / d.strftime("%d")
            if path.exists():
                folders.append(path)
        return folders

    # no date range -> scan
    folders = []
    for year in sorted(data_root.iterdir()):
        if not year.is_dir():
            continue
        for month in sorted(year.iterdir()):
            if not month.is_dir():
                continue
            for day in sorted(month.iterdir()):
                if day.is_dir():
                    folders.append(day)
    return folders


def process_day_folder(day_folder: Path, overwrite: bool = False) -> dict:
    """Process one day folder into data_matrix_feat.csv."""
    start = time.time()
    input_path = _resolve_daily_file(day_folder)
    if input_path is None:
        return {"date": day_folder.as_posix(), "status": "missing_input", "rows": 0, "cols": 0, "error": ""}

    output_path = day_folder / "data_matrix_feat.csv"
    if output_path.exists() and not overwrite:
        # If a previous run created an empty file, rebuild it.
        try:
            if output_path.stat().st_size == 0:
                pass
            else:
                return {"date": day_folder.as_posix(), "status": "skipped", "rows": 0, "cols": 0, "error": ""}
        except FileNotFoundError:
            return {"date": day_folder.as_posix(), "status": "skipped", "rows": 0, "cols": 0, "error": ""}

    try:
        rows, cols = _build_features_for_file(input_path, output_path)
        status = "ok"
        err = ""
    except Exception as e:
        rows, cols = 0, 0
        status = "failed"
        err = str(e)
    elapsed = time.time() - start
    return {"date": day_folder.as_posix(), "status": status, "rows": rows, "cols": cols, "error": err, "elapsed": elapsed}


def build_features(data_root: Path, start_date: Optional[str], end_date: Optional[str], n_workers: int, overwrite: bool) -> pd.DataFrame:
    """Run feature building across all day folders."""
    day_folders = list_day_folders(data_root, start_date, end_date)
    results = []
    if n_workers <= 1:
        for folder in tqdm(day_folders, desc="Build features", ncols=80):
            results.append(process_day_folder(folder, overwrite=overwrite))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(process_day_folder, folder, overwrite): folder for folder in day_folders}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Build features", ncols=80):
                results.append(fut.result())
    return pd.DataFrame(results)


def main() -> None:
    """CLI entry for feature building."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/project_5year")
    parser.add_argument("--start_date", type=str, default=None)
    parser.add_argument("--end_date", type=str, default=None)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    print("[Step1] Feature building start")
    df_log = build_features(data_root, args.start_date, args.end_date, args.n_workers, args.overwrite)
    print("[Step1] Feature building done")

    if args.run_name:
        out_dir = Path("res/experiments") / args.run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "feature_build_log.csv"
    else:
        log_path = Path("res/feature_build_log.csv")
        log_path.parent.mkdir(parents=True, exist_ok=True)

    df_log.to_csv(log_path, index=False)


if __name__ == "__main__":
    main()

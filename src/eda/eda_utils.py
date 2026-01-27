import os
from typing import Iterable, List, Dict, Tuple

import pandas as pd


def list_data_files(root: str) -> List[str]:
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name == "data_matrix.csv":
                paths.append(os.path.join(dirpath, name))
    return sorted(paths)


def iter_csv_chunks(paths: Iterable[str], chunksize: int = 200_000):
    for path in paths:
        for chunk in pd.read_csv(path, chunksize=chunksize):
            yield path, chunk


def get_schema(sample_path: str) -> Tuple[List[str], Dict[str, str]]:
    df = pd.read_csv(sample_path, nrows=5)
    return list(df.columns), {c: str(t) for c, t in df.dtypes.items()}


def summarize_y(paths: Iterable[str], chunksize: int = 200_000) -> pd.DataFrame:
    total = 0
    missing = 0
    series_parts = []
    for _, chunk in iter_csv_chunks(paths, chunksize=chunksize):
        if "y" not in chunk.columns:
            continue
        y = chunk["y"]
        total += len(y)
        missing += y.isna().sum()
        series_parts.append(y.dropna())
    if not series_parts:
        return pd.DataFrame()
    y_all = pd.concat(series_parts, ignore_index=True)
    summary = y_all.describe(percentiles=[0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99])
    out = summary.to_frame(name="y").reset_index().rename(columns={"index": "stat"})
    out.loc[len(out)] = ["count_total", total]
    out.loc[len(out)] = ["missing", missing]
    out.loc[len(out)] = ["missing_rate", (missing / total) if total else None]
    return out


def summarize_missingness(paths: Iterable[str], columns: List[str], chunksize: int = 200_000) -> pd.DataFrame:
    total = 0
    missing_counts = {c: 0 for c in columns}
    for _, chunk in iter_csv_chunks(paths, chunksize=chunksize):
        total += len(chunk)
        for c in columns:
            if c in chunk.columns:
                missing_counts[c] += chunk[c].isna().sum()
    rows = []
    for c in columns:
        miss = missing_counts[c]
        rows.append({"column": c, "missing": miss, "missing_rate": (miss / total) if total else None})
    return pd.DataFrame(rows)


def summarize_id_date_range(paths: Iterable[str], chunksize: int = 200_000) -> pd.DataFrame:
    id_min = {}
    id_max = {}
    for _, chunk in iter_csv_chunks(paths, chunksize=chunksize):
        if "id" not in chunk.columns or "DateTime" not in chunk.columns:
            continue
        ids = chunk["id"].to_numpy()
        dt = pd.to_datetime(chunk["DateTime"], errors="coerce")
        temp = pd.DataFrame({"id": ids, "dt": dt}).dropna()
        if temp.empty:
            continue
        grp = temp.groupby("id")["dt"].agg(["min", "max"])
        for idx, row in grp.iterrows():
            cur_min = id_min.get(idx)
            cur_max = id_max.get(idx)
            new_min = row["min"] if cur_min is None else min(cur_min, row["min"])
            new_max = row["max"] if cur_max is None else max(cur_max, row["max"])
            id_min[idx] = new_min
            id_max[idx] = new_max
    rows = [{"id": k, "start": id_min[k], "end": id_max[k]} for k in id_min.keys()]
    return pd.DataFrame(rows).sort_values("id").reset_index(drop=True)

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


def list_daily_paths(data_root: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Path]:
    """List day folders under data_root with optional date range."""
    root = Path(data_root)
    if start_date and end_date:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        days = []
        cur = start
        while cur <= end:
            days.append(cur)
            cur += pd.Timedelta(days=1)
        out = []
        for d in days:
            path = root / d.strftime("%Y") / d.strftime("%m") / d.strftime("%d")
            if path.exists():
                out.append(path)
        return out

    out = []
    if not root.exists():
        return out
    for year in sorted(root.iterdir()):
        if not year.is_dir():
            continue
        for month in sorted(year.iterdir()):
            if not month.is_dir():
                continue
            for day in sorted(month.iterdir()):
                if day.is_dir():
                    out.append(day)
    return out


def resolve_daily_file(day_folder: Path, prefer_feat: bool = True) -> Optional[Path]:
    """Resolve the preferred daily file (feat > raw > gz)."""
    if prefer_feat:
        feat = day_folder / "data_matrix_feat.csv"
        if feat.exists():
            return feat
    csv_path = day_folder / "data_matrix.csv"
    if csv_path.exists():
        return csv_path
    gz_path = day_folder / "data_matrix.csv.gz"
    if gz_path.exists():
        return gz_path
    return None


def load_daily(day_folder: Path, use_feat: bool = True, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Load a single day file and append date column."""
    path = resolve_daily_file(day_folder, prefer_feat=use_feat)
    if path is None:
        return pd.DataFrame()
    try:
        if path.stat().st_size == 0:
            return pd.DataFrame()
        df = pd.read_csv(path, compression="infer", usecols=columns)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    date_str = f"{day_folder.parts[-3]}-{day_folder.parts[-2]}-{day_folder.parts[-1]}"
    df["date"] = date_str
    return df


def load_range(
    data_root: str,
    start_date: str,
    end_date: str,
    use_feat: bool = True,
    columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Load a date range by concatenating daily files."""
    paths = list_daily_paths(data_root, start_date, end_date)
    return load_paths(paths, use_feat=use_feat, columns=columns)


def load_paths(
    paths: Iterable[Path],
    use_feat: bool = True,
    columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Load a list of day folders and concatenate them."""
    frames = []
    for day_folder in paths:
        df = load_daily(day_folder, use_feat=use_feat, columns=columns)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

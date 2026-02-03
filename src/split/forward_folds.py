from __future__ import annotations

from typing import List

import pandas as pd


def _month_end(dt: pd.Timestamp) -> pd.Timestamp:
    return (dt + pd.offsets.MonthEnd(0)).normalize()


def _month_start(dt: pd.Timestamp) -> pd.Timestamp:
    return dt.normalize().replace(day=1)


def generate_forward_folds(
    start_date: str = "2016-01-01",
    end_date: str = "2020-12-31",
    train_months: int = 36,
    val_months: int = 6,
    test_months: int = 6,
    step_months: int = 6,
    mode: str = "rolling",
) -> List[dict]:
    """Generate chronological forward folds (H1/H2 by default)."""
    if mode not in {"rolling", "expanding"}:
        raise ValueError("forward mode must be one of: rolling, expanding")

    start = _month_start(pd.to_datetime(start_date))
    end = _month_end(pd.to_datetime(end_date))
    cursor = _month_start(start + pd.DateOffset(months=train_months))
    folds = []
    fold_id = 0
    while True:
        val_start = cursor
        val_end = _month_end(val_start + pd.DateOffset(months=val_months) - pd.DateOffset(days=1))
        test_start = _month_start(val_end + pd.DateOffset(days=1))
        test_end = _month_end(test_start + pd.DateOffset(months=test_months) - pd.DateOffset(days=1))
        if test_end > end:
            break

        train_end = _month_end(val_start - pd.DateOffset(days=1))
        if mode == "rolling":
            train_start = _month_start(train_end - pd.DateOffset(months=train_months - 1))
            if train_start < start:
                cursor = _month_start(cursor + pd.DateOffset(months=step_months))
                continue
        else:
            train_start = start

        folds.append(
            {
                "fold_id": fold_id,
                "train_start": train_start.strftime("%Y-%m-%d"),
                "train_end": train_end.strftime("%Y-%m-%d"),
                "val_start": val_start.strftime("%Y-%m-%d"),
                "val_end": val_end.strftime("%Y-%m-%d"),
                "test_start": test_start.strftime("%Y-%m-%d"),
                "test_end": test_end.strftime("%Y-%m-%d"),
                "train": (train_start.strftime("%Y-%m-%d"), train_end.strftime("%Y-%m-%d")),
                "valid": (val_start.strftime("%Y-%m-%d"), val_end.strftime("%Y-%m-%d")),
                "test": (test_start.strftime("%Y-%m-%d"), test_end.strftime("%Y-%m-%d")),
            }
        )
        fold_id += 1
        cursor = _month_start(cursor + pd.DateOffset(months=step_months))
    return folds


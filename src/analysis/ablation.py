from __future__ import annotations

from typing import Dict, Iterable, List, Set

import pandas as pd


def drop_by_prefix(cols: Iterable[str], prefixes: Iterable[str]) -> List[str]:
    pref = tuple(prefixes)
    return [c for c in cols if not c.startswith(pref)]


def filter_variant_columns(
    feature_cols: List[str],
    variant_name: str,
    include_missing_features: bool = True,
) -> List[str]:
    """Filter columns for ablation variants by name rules."""
    cols = list(feature_cols)
    miss_cols = {"r_missing_cnt", "dv_missing_cnt", "beta_isna", "indbeta_isna", "any_f_missing", "industry_isna"}
    beta_cols = {"beta", "indbeta"}
    econ_prefixes = ("mom_", "rev_", "vol_", "absret_", "dv_log_", "dv_shock", "pv_corr", "ret_")

    if variant_name in {"FULL", "CORE"}:
        return cols
    if variant_name in {"-MISSING", "CORE+ID", "CORE+ID+MISSING", "FULL-ID"}:
        if variant_name in {"-MISSING", "FULL-ID"}:
            return [c for c in cols if c not in miss_cols]
    if variant_name in {"-IND"}:
        return [c for c in cols if not c.startswith("industry_") and not c.startswith("ind_")]
    if variant_name in {"-ECON"}:
        out = []
        for c in cols:
            if c.startswith(econ_prefixes):
                continue
            out.append(c)
        return out
    if variant_name in {"-BETA"}:
        return [c for c in cols if c not in beta_cols]
    return cols


def build_ablation_variants() -> Dict[str, Dict[str, bool]]:
    """Variant config for add-one and drop-one."""
    return {
        "CORE": {"use_id": False, "use_missing": False},
        "CORE+ID": {"use_id": True, "use_missing": False},
        "CORE+MISSING": {"use_id": False, "use_missing": True},
        "CORE+ID+MISSING": {"use_id": True, "use_missing": True},
        "FULL": {"use_id": True, "use_missing": True},
        "FULL-ID": {"use_id": False, "use_missing": True},
        "-MISSING": {"use_id": True, "use_missing": False},
        "-IND": {"use_id": True, "use_missing": True},
        "-ECON": {"use_id": True, "use_missing": True},
        "-BETA": {"use_id": True, "use_missing": True},
    }


def make_ablation_tables(rows: List[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split combined rows into add-one and drop-one tables with deltas."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df, df
    add_names = {"CORE", "CORE+ID", "CORE+MISSING", "CORE+ID+MISSING"}
    drop_names = {"FULL", "FULL-ID", "-MISSING", "-IND", "-ECON", "-BETA"}

    add_df = df[df["variant_name"].isin(add_names)].copy()
    drop_df = df[df["variant_name"].isin(drop_names)].copy()

    core_valid = add_df.loc[add_df["variant_name"] == "CORE", "valid_score"]
    core_test = add_df.loc[add_df["variant_name"] == "CORE", "test_score"]
    full_valid = drop_df.loc[drop_df["variant_name"] == "FULL", "valid_score"]
    full_test = drop_df.loc[drop_df["variant_name"] == "FULL", "test_score"]
    c_v = float(core_valid.iloc[0]) if not core_valid.empty else float("nan")
    c_t = float(core_test.iloc[0]) if not core_test.empty else float("nan")
    f_v = float(full_valid.iloc[0]) if not full_valid.empty else float("nan")
    f_t = float(full_test.iloc[0]) if not full_test.empty else float("nan")

    add_df["delta_valid"] = add_df["valid_score"] - c_v
    add_df["delta_test"] = add_df["test_score"] - c_t
    drop_df["delta_valid"] = drop_df["valid_score"] - f_v
    drop_df["delta_test"] = drop_df["test_score"] - f_t
    return add_df, drop_df


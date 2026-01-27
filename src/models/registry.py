from __future__ import annotations

from typing import Dict, List


def _try_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


MODEL_SPECS: Dict[str, dict] = {
    "lgbm": {"type": "lgbm"},
    "rf": {"type": "sklearn"},
    "extra_trees": {"type": "sklearn"},
    "ridge": {"type": "sklearn"},
    "elasticnet": {"type": "sklearn"},
    "torch_linear": {"type": "torch"},
    "torch_mlp": {"type": "torch"},
}

if _try_import("catboost"):
    MODEL_SPECS["catboost"] = {"type": "catboost"}

if _try_import("xgboost"):
    MODEL_SPECS["xgb"] = {"type": "xgb"}


def list_models() -> List[str]:
    return list(MODEL_SPECS.keys())

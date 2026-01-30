from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def train_lgbm(
    X_train,
    y_train,
    w_train,
    X_valid,
    y_valid,
    w_valid,
    num_threads: int = 16,
    **kwargs,
) -> Tuple[object, Callable]:
    """Train LightGBM regressor with early stopping."""
    try:
        import lightgbm as lgb
    except Exception as e:
        raise ImportError("lightgbm is not installed") from e

    model = lgb.LGBMRegressor(
        n_estimators=kwargs.get("n_estimators", 2000),
        learning_rate=kwargs.get("learning_rate", 0.03),
        num_leaves=kwargs.get("num_leaves", 64),
        max_depth=kwargs.get("max_depth", -1),
        min_data_in_leaf=kwargs.get("min_data_in_leaf", 20),
        subsample=kwargs.get("subsample", 0.8),
        colsample_bytree=kwargs.get("colsample_bytree", 0.8),
        reg_lambda=kwargs.get("reg_lambda", 0.0),
        num_threads=num_threads,
        random_state=42,
    )
    model.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        eval_set=[(X_valid, y_valid)],
        eval_sample_weight=[w_valid],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model, lambda Xnew: model.predict(Xnew)

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor


def _prepare_X(X):
    """Fill NaNs and convert to float32 numpy array."""
    return X.fillna(0).to_numpy(dtype=np.float32)


def train_ridge(X, y, w, **kwargs) -> Tuple[object, Callable]:
    """Train Ridge regression with sample weights."""
    model = Ridge(alpha=kwargs.get("alpha", 1.0))
    model.fit(_prepare_X(X), y, sample_weight=w)
    return model, lambda Xnew: model.predict(_prepare_X(Xnew))


def train_elasticnet(X, y, w, **kwargs) -> Tuple[object, Callable]:
    """Train ElasticNet regression with sample weights."""
    model = ElasticNet(
        alpha=kwargs.get("alpha", 0.001),
        l1_ratio=kwargs.get("l1_ratio", 0.5),
        max_iter=kwargs.get("max_iter", 2000),
    )
    model.fit(_prepare_X(X), y, sample_weight=w)
    return model, lambda Xnew: model.predict(_prepare_X(Xnew))


def train_rf(X, y, w, n_jobs: int = 16, **kwargs) -> Tuple[object, Callable]:
    """Train RandomForest regressor with sample weights."""
    model = RandomForestRegressor(
        n_estimators=kwargs.get("n_estimators", 300),
        max_depth=kwargs.get("max_depth", None),
        min_samples_leaf=kwargs.get("min_samples_leaf", 1),
        n_jobs=n_jobs,
        random_state=42,
    )
    model.fit(_prepare_X(X), y, sample_weight=w)
    return model, lambda Xnew: model.predict(_prepare_X(Xnew))


def train_extra_trees(X, y, w, n_jobs: int = 16, **kwargs) -> Tuple[object, Callable]:
    """Train ExtraTrees regressor with sample weights."""
    model = ExtraTreesRegressor(
        n_estimators=kwargs.get("n_estimators", 300),
        max_depth=kwargs.get("max_depth", None),
        min_samples_leaf=kwargs.get("min_samples_leaf", 1),
        n_jobs=n_jobs,
        random_state=42,
    )
    model.fit(_prepare_X(X), y, sample_weight=w)
    return model, lambda Xnew: model.predict(_prepare_X(Xnew))

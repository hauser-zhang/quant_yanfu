from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def _try_import_torch():
    """Import torch with a helpful error if missing."""
    try:
        import torch
        return torch
    except Exception as e:
        raise ImportError("torch is not installed") from e


def _prepare_X(X):
    """Fill NaNs and convert to float32 numpy array."""
    return X.fillna(0).to_numpy(dtype=np.float32)


def _weighted_mse(pred, target, weight):
    """Weighted MSE with normalization by weight sum."""
    return (weight * (pred - target) ** 2).sum() / (weight.sum() + 1e-12)


def train_torch_linear(
    X_train,
    y_train,
    w_train,
    X_valid,
    y_valid,
    w_valid,
    max_epochs: int = 50,
    device: str | None = None,
):
    """Train a linear model with weighted MSE on the given device."""
    torch = _try_import_torch()

    Xtr = _prepare_X(X_train)
    Xva = _prepare_X(X_valid)
    ytr = y_train.astype(np.float32)
    yva = y_valid.astype(np.float32)
    wtr = w_train.astype(np.float32)
    wva = w_valid.astype(np.float32)

    torch.manual_seed(42)
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    model = torch.nn.Linear(Xtr.shape[1], 1).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_loss = float("inf")
    best_state = None

    for _ in range(max_epochs):
        model.train()
        xb = torch.from_numpy(Xtr).to(dev)
        yb = torch.from_numpy(ytr).view(-1, 1).to(dev)
        wb = torch.from_numpy(wtr).view(-1, 1).to(dev)
        pred = model(xb)
        loss = _weighted_mse(pred, yb, wb)
        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            xv = torch.from_numpy(Xva).to(dev)
            yv = torch.from_numpy(yva).view(-1, 1).to(dev)
            wv = torch.from_numpy(wva).view(-1, 1).to(dev)
            pv = model(xv)
            vloss = _weighted_mse(pv, yv, wv).item()
        if vloss < best_loss:
            best_loss = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    def predict_fn(Xnew):
        model.eval()
        with torch.no_grad():
            xn = torch.from_numpy(_prepare_X(Xnew)).to(dev)
            return model(xn).view(-1).cpu().numpy()

    return model, predict_fn


def train_torch_mlp(
    X_train,
    y_train,
    w_train,
    X_valid,
    y_valid,
    w_valid,
    max_epochs: int = 80,
    device: str | None = None,
):
    """Train a small MLP with weighted MSE on the given device."""
    torch = _try_import_torch()

    Xtr = _prepare_X(X_train)
    Xva = _prepare_X(X_valid)
    ytr = y_train.astype(np.float32)
    yva = y_valid.astype(np.float32)
    wtr = w_train.astype(np.float32)
    wva = w_valid.astype(np.float32)

    torch.manual_seed(42)
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    model = torch.nn.Sequential(
        torch.nn.Linear(Xtr.shape[1], 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
    ).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_loss = float("inf")
    best_state = None

    for _ in range(max_epochs):
        model.train()
        xb = torch.from_numpy(Xtr).to(dev)
        yb = torch.from_numpy(ytr).view(-1, 1).to(dev)
        wb = torch.from_numpy(wtr).view(-1, 1).to(dev)
        pred = model(xb)
        loss = _weighted_mse(pred, yb, wb)
        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            xv = torch.from_numpy(Xva).to(dev)
            yv = torch.from_numpy(yva).view(-1, 1).to(dev)
            wv = torch.from_numpy(wva).view(-1, 1).to(dev)
            pv = model(xv)
            vloss = _weighted_mse(pv, yv, wv).item()
        if vloss < best_loss:
            best_loss = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    def predict_fn(Xnew):
        model.eval()
        with torch.no_grad():
            xn = torch.from_numpy(_prepare_X(Xnew)).to(dev)
            return model(xn).view(-1).cpu().numpy()

    return model, predict_fn

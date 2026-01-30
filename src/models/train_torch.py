from __future__ import annotations

from typing import Callable, Tuple
import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm
from src.eval.metrics import weighted_corr


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


def _mlp_blocks(in_dim, hidden_dims, dropout, batch_norm):
    """Build MLP blocks and return (blocks, last_dim)."""
    torch = _try_import_torch()
    blocks = torch.nn.ModuleList()
    prev = in_dim
    for h in hidden_dims:
        layers = [torch.nn.Linear(prev, h)]
        if batch_norm:
            layers.append(torch.nn.BatchNorm1d(h))
        layers.append(torch.nn.ReLU())
        if dropout and dropout > 0:
            layers.append(torch.nn.Dropout(dropout))
        blocks.append(torch.nn.Sequential(*layers))
        prev = h
    return blocks, prev


class MLPRegressor(_try_import_torch().nn.Module):
    """Simple MLP regressor with optional residual links."""

    def __init__(self, in_dim, hidden_dims, dropout=0.0, batch_norm=True, residual=False):
        super().__init__()
        self.blocks, last_dim = _mlp_blocks(in_dim, hidden_dims, dropout, batch_norm)
        self.residual = residual
        self.out = _try_import_torch().nn.Linear(last_dim, 1)

    def forward(self, x):
        for block in self.blocks:
            z = block(x)
            if self.residual and z.shape == x.shape:
                x = x + z
            else:
                x = z
        return self.out(x)


class CNNBackbone(_try_import_torch().nn.Module):
    """1D CNN backbone for (seq_len, feat_dim) inputs."""

    def __init__(self, in_channels=2, channels=(32, 64), kernel_size=3, dropout=0.1, batch_norm=True, residual=True):
        super().__init__()
        torch = _try_import_torch()
        self.blocks = torch.nn.ModuleList()
        self.residual = residual
        prev = in_channels
        for ch in channels:
            layers = [torch.nn.Conv1d(prev, ch, kernel_size=kernel_size, padding=kernel_size // 2)]
            if batch_norm:
                layers.append(torch.nn.BatchNorm1d(ch))
            layers.append(torch.nn.ReLU())
            if dropout and dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            self.blocks.append(torch.nn.Sequential(*layers))
            prev = ch
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.out_dim = prev

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        for block in self.blocks:
            z = block(x)
            if self.residual and z.shape == x.shape:
                x = x + z
            else:
                x = z
        x = self.pool(x).squeeze(-1)
        return x


class RNNBackbone(_try_import_torch().nn.Module):
    """RNN/LSTM/GRU backbone for (seq_len, feat_dim) inputs."""

    def __init__(
        self,
        rnn_type="lstm",
        hidden=64,
        num_layers=1,
        dropout=0.1,
        bidirectional=False,
        input_dim=2,
        residual=True,
    ):
        super().__init__()
        torch = _try_import_torch()
        rnn_cls = {"rnn": torch.nn.RNN, "lstm": torch.nn.LSTM, "gru": torch.nn.GRU}[rnn_type]
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.out_dim = hidden * (2 if bidirectional else 1)
        self.residual = residual
        self.res_proj = torch.nn.Linear(input_dim, self.out_dim) if residual else None

    def forward(self, x):
        out, _ = self.rnn(x)
        h = out[:, -1, :]
        if self.residual and self.res_proj is not None:
            h = h + self.res_proj(x[:, -1, :])
        return h


class SequenceHybridRegressor(_try_import_torch().nn.Module):
    """Sequence model with alpha (MLP) + risk (CAPM-like) heads."""

    def __init__(
        self,
        r_idx,
        dv_idx,
        f_idx,
        risk_idx,
        backbone_type="cnn",
        dropout=0.1,
        batch_norm=True,
        residual=True,
        f_hidden_dim=32,
        f_layers=1,
        alpha_hidden_dim=64,
        alpha_layers=2,
        cnn_channels=(32, 64),
        kernel_size=3,
        rnn_hidden=64,
        rnn_layers=1,
        bidirectional=False,
    ):
        super().__init__()
        torch = _try_import_torch()

        self.r_idx = r_idx
        self.dv_idx = dv_idx
        self.f_idx = f_idx
        self.risk_idx = risk_idx

        if backbone_type == "cnn":
            self.backbone = CNNBackbone(
                in_channels=2,
                channels=cnn_channels,
                kernel_size=kernel_size,
                dropout=dropout,
                batch_norm=batch_norm,
                residual=residual,
            )
        elif backbone_type in {"rnn", "lstm", "gru"}:
            self.backbone = RNNBackbone(
                rnn_type=backbone_type,
                hidden=rnn_hidden,
                num_layers=rnn_layers,
                dropout=dropout,
                bidirectional=bidirectional,
                input_dim=2,
                residual=residual,
            )
        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}")

        # f branch
        f_hidden_dims = [f_hidden_dim] * max(0, int(f_layers))
        self.f_blocks, f_last = _mlp_blocks(len(f_idx), f_hidden_dims, dropout, batch_norm) if f_idx else (None, 0)
        self.f_out = torch.nn.Linear(f_last, f_hidden_dim) if f_idx else None

        # alpha head
        alpha_in_dim = self.backbone.out_dim + (f_hidden_dim if f_idx else 0)
        alpha_hidden_dims = [alpha_hidden_dim] * max(0, int(alpha_layers))
        self.alpha = MLPRegressor(
            alpha_in_dim,
            alpha_hidden_dims,
            dropout=dropout,
            batch_norm=batch_norm,
            residual=residual,
        )

        # risk head: linear on industry/beta/indbeta
        self.risk = torch.nn.Linear(len(risk_idx), 1, bias=False) if risk_idx else None

    def forward(self, x):
        torch = _try_import_torch()
        r = x[:, self.r_idx]
        dv = x[:, self.dv_idx]
        seq = torch.stack([r, dv], dim=-1)  # (B, T, 2)
        seq_emb = self.backbone(seq)

        if self.f_idx:
            f = x[:, self.f_idx]
            for block in self.f_blocks:
                f = block(f)
            f = self.f_out(f)
            alpha_in = torch.cat([seq_emb, f], dim=1)
        else:
            alpha_in = seq_emb

        alpha = self.alpha(alpha_in)
        if self.risk is not None:
            risk = self.risk(x[:, self.risk_idx])
            return alpha + risk
        return alpha


def train_torch_model(
    X_train,
    y_train,
    w_train,
    X_valid,
    y_valid,
    w_valid,
    model_type: str = "mlp",
    max_epochs: int = 80,
    device: str | None = None,
    **kwargs,
) -> Tuple[object, Callable]:
    """Train a torch model with weighted MSE on the given device."""
    torch = _try_import_torch()
    log_path = kwargs.get("log_path", None)
    feature_names = kwargs.get("feature_names", None)
    batch_size = int(kwargs.get("batch_size", 4096))
    debug = bool(kwargs.get("debug", False))
    show_progress = bool(kwargs.get("show_progress", True))
    num_workers = int(kwargs.get("num_workers", 8))
    log_every = int(kwargs.get("log_every", 1))
    early_stop_patience = int(kwargs.get("early_stop_patience", 0))
    early_stop_min_delta = float(kwargs.get("early_stop_min_delta", 0.0))
    grid_idx = kwargs.get("grid_idx", None)
    grid_total = kwargs.get("grid_total", None)
    param_tag = kwargs.get("param_tag", None)

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
    lr = kwargs.get("lr", 1e-3)
    max_epochs = kwargs.get("max_epochs", max_epochs)

    if model_type in {"linear", "mlp"}:
        num_layers = int(kwargs.get("num_layers", 2))
        hidden_dim = int(kwargs.get("hidden_dim", 128))
        dropout = float(kwargs.get("dropout", 0.0))
        batch_norm = bool(kwargs.get("batch_norm", True))
        residual = bool(kwargs.get("residual", False))
        hidden_dims = [] if model_type == "linear" or num_layers <= 0 else [hidden_dim] * num_layers
        model = MLPRegressor(Xtr.shape[1], hidden_dims, dropout=dropout, batch_norm=batch_norm, residual=residual)
    else:
        if not feature_names:
            raise ValueError("feature_names must be provided for sequence models")
        r_cols = [f"r_{i}" for i in range(20) if f"r_{i}" in feature_names]
        dv_cols = [f"dv_{i}" for i in range(20) if f"dv_{i}" in feature_names]
        if len(r_cols) != 20 or len(dv_cols) != 20:
            raise ValueError("r_0..r_19 and dv_0..dv_19 are required for sequence models")
        r_idx = [feature_names.index(c) for c in r_cols]
        dv_idx = [feature_names.index(c) for c in dv_cols]
        f_idx = [feature_names.index(c) for c in feature_names if c.startswith("f_")]
        risk_cols = [c for c in feature_names if c.startswith("industry_")] + [c for c in ["beta", "indbeta"] if c in feature_names]
        risk_idx = [feature_names.index(c) for c in risk_cols]

        model = SequenceHybridRegressor(
            r_idx=r_idx,
            dv_idx=dv_idx,
            f_idx=f_idx,
            risk_idx=risk_idx,
            backbone_type=model_type,
            dropout=float(kwargs.get("dropout", 0.1)),
            batch_norm=bool(kwargs.get("batch_norm", True)),
            residual=bool(kwargs.get("residual", True)),
            f_hidden_dim=int(kwargs.get("f_hidden_dim", 32)),
            f_layers=int(kwargs.get("f_layers", 1)),
            alpha_hidden_dim=int(kwargs.get("alpha_hidden_dim", 64)),
            alpha_layers=int(kwargs.get("alpha_layers", 2)),
            cnn_channels=kwargs.get("cnn_channels", (32, 64)),
            kernel_size=int(kwargs.get("kernel_size", 3)),
            rnn_hidden=int(kwargs.get("rnn_hidden", 64)),
            rnn_layers=int(kwargs.get("rnn_layers", 1)),
            bidirectional=bool(kwargs.get("bidirectional", False)),
        )

    model = model.to(dev)
    if debug:
        print(model)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float("inf")
    best_state = None
    best_epoch = -1
    patience_count = 0
    history = []

    n_train = Xtr.shape[0]
    n_valid = Xva.shape[0]
    pin_memory = dev.type == "cuda"
    train_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(Xtr), torch.from_numpy(ytr), torch.from_numpy(wtr)
    )
    valid_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(Xva), torch.from_numpy(yva), torch.from_numpy(wva)
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    epoch_iter = range(max_epochs)
    if show_progress:
        if grid_idx is not None and grid_total is not None:
            desc = f"Epochs[{model_type}][{grid_idx + 1}/{grid_total}]"
        else:
            desc = f"Epochs[{model_type}]"
        bar_format = "{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}"
        epoch_iter = tqdm(epoch_iter, desc=desc, ncols=100, leave=False, bar_format=bar_format)
    for epoch in epoch_iter:
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        train_iter = train_loader
        if show_progress:
            train_iter = tqdm(
                train_loader,
                desc=f"Batch[{model_type}] {epoch + 1}/{max_epochs}",
                ncols=100,
                leave=False,
            )
        for xb, yb, wb in train_iter:
            xb = xb.to(dev)
            yb = yb.view(-1, 1).to(dev)
            wb = wb.view(-1, 1).to(dev)
            pred = model(xb)
            loss = _weighted_mse(pred, yb, wb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss_sum += float(loss.item()) * len(xb)
            train_count += len(xb)
        train_loss = train_loss_sum / max(1, train_count)

        model.eval()
        with torch.no_grad():
            vloss_sum = 0.0
            vcount = 0
            preds_v = []
            yv_all = []
            wv_all = []
            for xv, yv, wv in valid_loader:
                xv = xv.to(dev)
                yv = yv.view(-1, 1).to(dev)
                wv = wv.view(-1, 1).to(dev)
                pv = model(xv)
                vloss_sum += float(_weighted_mse(pv, yv, wv).item()) * len(xv)
                vcount += len(xv)
                preds_v.append(pv.view(-1).cpu().numpy())
                yv_all.append(yv.view(-1).cpu().numpy())
                wv_all.append(wv.view(-1).cpu().numpy())
            vloss = vloss_sum / max(1, vcount)

        # compute weighted corr only when logging (can be expensive)
        do_log = log_every > 0 and (epoch + 1) % log_every == 0
        train_corr = float("nan")
        valid_corr = float("nan")
        if do_log:
            def _predict_all(loader):
                preds = []
                ys = []
                ws = []
                with torch.no_grad():
                    for xb, yb, wb in loader:
                        xb = xb.to(dev)
                        pv = model(xb).view(-1).detach().cpu().numpy()
                        preds.append(pv)
                        ys.append(yb.view(-1).cpu().numpy())
                        ws.append(wb.view(-1).cpu().numpy())
                return np.concatenate(preds), np.concatenate(ys), np.concatenate(ws)

            preds_tr, ytr_all, wtr_all = _predict_all(train_loader)
            preds_v = np.concatenate(preds_v) if preds_v else np.array([])
            yv_all = np.concatenate(yv_all) if yv_all else np.array([])
            wv_all = np.concatenate(wv_all) if wv_all else np.array([])
            train_corr = weighted_corr(ytr_all, preds_tr, wtr_all)
            valid_corr = weighted_corr(yv_all, preds_v, wv_all)

            msg = (
                f"[Epoch {epoch + 1}/{max_epochs}] "
                f"train_loss={train_loss:.4f}, train_ic={train_corr:.4f}, "
                f"val_loss={vloss:.4f}, val_ic={valid_corr:.4f}"
            )
            print(msg)

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "train_corr": float(train_corr),
                "valid_loss": float(vloss),
                "valid_corr": float(valid_corr),
            }
        )
        if vloss < best_loss - early_stop_min_delta:
            best_loss = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            patience_count = 0
        else:
            if early_stop_patience > 0:
                patience_count += 1
                if patience_count >= early_stop_patience:
                    msg = f"Early stop at epoch {epoch + 1} (best={best_epoch}, best_loss={best_loss:.4f})"
                    if show_progress:
                        tqdm.write(msg)
                    else:
                        print(msg)
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    if log_path:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_corr", "valid_loss", "valid_corr"])
            writer.writeheader()
            writer.writerows(history)

    def predict_fn(Xnew):
        model.eval()
        with torch.no_grad():
            xn = torch.from_numpy(_prepare_X(Xnew)).to(dev)
            return model(xn).view(-1).cpu().numpy()

    return model, predict_fn

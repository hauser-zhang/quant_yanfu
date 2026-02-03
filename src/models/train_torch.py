from __future__ import annotations

from typing import Callable, Tuple
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from src.eval.metrics import weighted_corr, daily_weighted_mean_ic


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


def _assert_finite_after_fill(X, name: str):
    """Assert feature matrix has finite values after fillna(0)."""
    arr = X.fillna(0).to_numpy(dtype=np.float32)
    if np.isfinite(arr).all():
        return arr
    bad_cols = []
    for c in X.columns:
        v = X[c].fillna(0).to_numpy(dtype=np.float32)
        if not np.isfinite(v).all():
            bad_cols.append(c)
    raise ValueError(f"Non-finite values found in {name} after fillna(0). offending_cols={bad_cols[:20]}")


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
        # Accept int/tuple/list for channels from param grid.
        if isinstance(channels, (int, np.integer)):
            channels = [int(channels)]
        elif not isinstance(channels, (list, tuple)):
            channels = [int(channels)]
        prev = in_channels
        for ch in channels:
            ch = int(ch)
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
        id_emb_dim=0,
    ):
        super().__init__()
        torch = _try_import_torch()

        self.register_buffer("r_idx", torch.tensor(r_idx, dtype=torch.long), persistent=False)
        self.register_buffer("dv_idx", torch.tensor(dv_idx, dtype=torch.long), persistent=False)
        self.register_buffer("f_idx", torch.tensor(f_idx, dtype=torch.long), persistent=False)
        self.register_buffer("risk_idx", torch.tensor(risk_idx, dtype=torch.long), persistent=False)
        self.has_f = len(f_idx) > 0
        self.has_risk = len(risk_idx) > 0

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
        self.f_blocks, f_last = _mlp_blocks(len(f_idx), f_hidden_dims, dropout, batch_norm) if self.has_f else (None, 0)
        self.f_out = torch.nn.Linear(f_last, f_hidden_dim) if self.has_f else None

        # alpha head
        self.id_emb_dim = int(id_emb_dim)
        alpha_in_dim = self.backbone.out_dim + (f_hidden_dim if self.has_f else 0) + self.id_emb_dim
        alpha_hidden_dims = [alpha_hidden_dim] * max(0, int(alpha_layers))
        self.alpha = MLPRegressor(
            alpha_in_dim,
            alpha_hidden_dims,
            dropout=dropout,
            batch_norm=batch_norm,
            residual=residual,
        )

        # risk head: linear on industry/beta/indbeta
        self.risk = torch.nn.Linear(len(risk_idx), 1, bias=False) if self.has_risk else None

    def forward(self, x, id_emb=None):
        torch = _try_import_torch()
        r = x.index_select(1, self.r_idx)
        dv = x.index_select(1, self.dv_idx)
        seq = torch.stack([r, dv], dim=-1)  # (B, T, 2)
        seq_emb = self.backbone(seq)

        if self.has_f:
            f = x.index_select(1, self.f_idx)
            for block in self.f_blocks:
                f = block(f)
            f = self.f_out(f)
            alpha_in = torch.cat([seq_emb, f], dim=1)
        else:
            alpha_in = seq_emb

        if self.id_emb_dim > 0 and id_emb is not None:
            alpha_in = torch.cat([alpha_in, id_emb], dim=1)
        alpha = self.alpha(alpha_in)
        if self.risk is not None:
            risk = self.risk(x.index_select(1, self.risk_idx))
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
    scheduler_type = str(kwargs.get("scheduler_type", "plateau_ic"))
    plateau_factor = float(kwargs.get("plateau_factor", 0.8))
    plateau_patience = int(kwargs.get("plateau_patience", 8))
    plateau_min_lr = float(kwargs.get("plateau_min_lr", 1e-5))
    best_metric = str(kwargs.get("best_metric", "ic")).lower()
    early_stop_metric = str(kwargs.get("early_stop_metric", "ic")).lower()
    if best_metric not in {"ic", "loss"}:
        raise ValueError("best_metric must be one of: ic, loss")
    if early_stop_metric not in {"ic", "loss"}:
        raise ValueError("early_stop_metric must be one of: ic, loss")
    valid_dates = kwargs.get("valid_dates", None)
    if valid_dates is not None:
        valid_dates = np.asarray(valid_dates)
    use_id_embedding = bool(kwargs.get("use_id_embedding", False))
    id_emb_dim = int(kwargs.get("id_emb_dim", 8))
    id_dropout_p = float(kwargs.get("id_dropout_p", 0.1))
    id_train = kwargs.get("id_train", None)
    id_valid = kwargs.get("id_valid", None)
    id_predict = kwargs.get("id_predict", None)

    if hasattr(X_train, "columns"):
        Xtr = _assert_finite_after_fill(X_train, "X_train")
    else:
        Xtr = _prepare_X(X_train)
    if hasattr(X_valid, "columns"):
        Xva = _assert_finite_after_fill(X_valid, "X_valid")
    else:
        Xva = _prepare_X(X_valid)
    ytr = y_train.astype(np.float32)
    yva = y_valid.astype(np.float32)
    wtr = w_train.astype(np.float32)
    wva = w_valid.astype(np.float32)

    torch.manual_seed(42)
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    lr = float(kwargs.get("lr", 3e-4))
    weight_decay = float(kwargs.get("weight_decay", 1e-4))
    grad_clip_norm = float(kwargs.get("grad_clip_norm", 1.0))
    optim_name = str(kwargs.get("optim", "adamw")).lower()
    if optim_name not in {"adamw", "adam"}:
        raise ValueError("optim must be one of: adamw, adam")
    max_epochs = kwargs.get("max_epochs", max_epochs)

    if model_type in {"linear", "mlp"}:
        num_layers = int(kwargs.get("num_layers", 2))
        hidden_dim = int(kwargs.get("hidden_dim", 128))
        dropout = float(kwargs.get("dropout", 0.0))
        batch_norm = bool(kwargs.get("batch_norm", True))
        residual = bool(kwargs.get("residual", False))
        hidden_dims = [] if model_type == "linear" or num_layers <= 0 else [hidden_dim] * num_layers
        in_dim = Xtr.shape[1] + (id_emb_dim if use_id_embedding and id_train is not None else 0)
        model = MLPRegressor(in_dim, hidden_dims, dropout=dropout, batch_norm=batch_norm, residual=residual)
    else:
        if feature_names is None or len(feature_names) != Xtr.shape[1]:
            if hasattr(X_train, "columns"):
                feature_names = list(X_train.columns)
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
        max_idx = max(r_idx + dv_idx + f_idx + risk_idx)
        if max_idx >= Xtr.shape[1]:
            raise ValueError(
                f"Sequence index out of bounds: max_idx={max_idx}, X_dim={Xtr.shape[1]}. "
                "Check feature_names vs X columns."
            )

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
            id_emb_dim=id_emb_dim if use_id_embedding and id_train is not None else 0,
        )

    model = model.to(dev)
    if debug:
        print(model)
    if use_id_embedding and id_train is not None:
        id_vocab_size = int(max(np.max(id_train), np.max(id_valid) if id_valid is not None else 0) + 1)
        id_embed = torch.nn.Embedding(id_vocab_size, id_emb_dim).to(dev)
        params_all = list(model.parameters()) + list(id_embed.parameters())
    else:
        id_embed = None
        params_all = list(model.parameters())
    if optim_name == "adamw":
        opt = torch.optim.AdamW(params_all, lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.Adam(params_all, lr=lr, weight_decay=weight_decay)
    scheduler = None
    if scheduler_type == "plateau_ic":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=plateau_factor, patience=plateau_patience, min_lr=plateau_min_lr
        )
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=lr * 0.05)
    if best_metric == "ic":
        best_score = float("-inf")
    else:
        best_score = float("inf")
    if early_stop_metric == "ic":
        early_best_score = float("-inf")
    else:
        early_best_score = float("inf")
    best_state = None
    best_id_state = None
    best_epoch = -1
    patience_count = 0
    history = []
    history_fields = [
        "epoch",
        "train_loss",
        "train_corr",
        "valid_loss",
        "valid_corr",
        "valid_ic",
        "lr",
        "weight_decay",
        "grad_clip_norm",
        "optim",
        "best_epoch",
        "best_metric_value",
    ]

    def _flush_history() -> None:
        """Write history CSV and a quick trend plot so training progress is visible during run."""
        if not log_path or not history:
            return
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history_fields)
            writer.writeheader()
            writer.writerows(history)
        # Best effort: plot should not break training.
        try:
            import matplotlib.pyplot as plt

            dfh = pd.DataFrame(history).sort_values("epoch")
            fig, axes = plt.subplots(2, 1, figsize=(9, 6), dpi=150, sharex=True)
            axes[0].plot(dfh["epoch"], dfh["train_loss"], label="train_loss", linewidth=1.4)
            axes[0].plot(dfh["epoch"], dfh["valid_loss"], label="valid_loss", linewidth=1.4)
            axes[0].set_ylabel("Loss")
            axes[0].legend(frameon=False)
            axes[1].plot(dfh["epoch"], dfh["train_corr"], label="train_ic", linewidth=1.4)
            axes[1].plot(dfh["epoch"], dfh["valid_ic"], label="valid_ic", linewidth=1.4)
            axes[1].set_ylabel("IC")
            axes[1].set_xlabel("Epoch")
            axes[1].legend(frameon=False)
            fig.tight_layout()
            fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
            plt.close(fig)
        except Exception:
            pass

    n_train = Xtr.shape[0]
    n_valid = Xva.shape[0]
    pin_memory = dev.type == "cuda"
    if use_id_embedding and id_train is not None:
        train_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(Xtr),
            torch.from_numpy(ytr),
            torch.from_numpy(wtr),
            torch.from_numpy(np.asarray(id_train, dtype=np.int64)),
        )
        valid_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(Xva),
            torch.from_numpy(yva),
            torch.from_numpy(wva),
            torch.from_numpy(np.asarray(id_valid, dtype=np.int64)),
        )
    else:
        train_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr), torch.from_numpy(wtr))
        valid_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva), torch.from_numpy(wva))
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
        for batch in train_iter:
            if use_id_embedding and id_embed is not None:
                xb, yb, wb, ib = batch
            else:
                xb, yb, wb = batch
                ib = None
            xb = xb.to(dev)
            yb = yb.view(-1, 1).to(dev)
            wb = wb.view(-1, 1).to(dev)
            if ib is not None:
                ib = ib.to(dev)
                if id_dropout_p > 0:
                    drop_mask = torch.rand_like(ib.float()) < id_dropout_p
                    ib = torch.where(drop_mask, torch.zeros_like(ib), ib)
                e = id_embed(ib)
                pred = model(xb, e) if model_type in {"cnn", "rnn", "lstm", "gru"} else model(torch.cat([xb, e], dim=1))
            else:
                pred = model(xb)
            loss = _weighted_mse(pred, yb, wb)
            opt.zero_grad()
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(params_all, grad_clip_norm)
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
            for batch in valid_loader:
                if use_id_embedding and id_embed is not None:
                    xv, yv, wv, iv = batch
                else:
                    xv, yv, wv = batch
                    iv = None
                xv = xv.to(dev)
                yv = yv.view(-1, 1).to(dev)
                wv = wv.view(-1, 1).to(dev)
                if iv is not None:
                    iv = iv.to(dev)
                    e = id_embed(iv)
                    pv = model(xv, e) if model_type in {"cnn", "rnn", "lstm", "gru"} else model(torch.cat([xv, e], dim=1))
                else:
                    pv = model(xv)
                vloss_sum += float(_weighted_mse(pv, yv, wv).item()) * len(xv)
                vcount += len(xv)
                preds_v.append(pv.view(-1).cpu().numpy())
                yv_all.append(yv.view(-1).cpu().numpy())
                wv_all.append(wv.view(-1).cpu().numpy())
            vloss = vloss_sum / max(1, vcount)

        preds_v = np.concatenate(preds_v) if preds_v else np.array([])
        yv_all = np.concatenate(yv_all) if yv_all else np.array([])
        wv_all = np.concatenate(wv_all) if wv_all else np.array([])
        if valid_dates is not None and len(valid_dates) == len(yv_all):
            df_ic = pd.DataFrame({"date": valid_dates, "pred": preds_v, "y_score": yv_all, "weight": wv_all})
            valid_ic = daily_weighted_mean_ic(df_ic, "pred", "y_score", "weight", "date")
            valid_corr = valid_ic
        else:
            valid_corr = weighted_corr(yv_all, preds_v, wv_all)
            valid_ic = valid_corr

        # Print/flush cadence.
        do_log = (
            log_every <= 1
            or (log_every > 0 and (epoch + 1) % log_every == 0)
            or epoch == 0
            or (epoch + 1) == int(max_epochs)
        )
        def _predict_all(loader):
            preds = []
            ys = []
            ws = []
            with torch.no_grad():
                for batch in loader:
                    if use_id_embedding and id_embed is not None:
                        xb, yb, wb, ib = batch
                    else:
                        xb, yb, wb = batch
                        ib = None
                    xb = xb.to(dev)
                    if ib is not None:
                        ib = ib.to(dev)
                        e = id_embed(ib)
                        pv_t = model(xb, e) if model_type in {"cnn", "rnn", "lstm", "gru"} else model(torch.cat([xb, e], dim=1))
                    else:
                        pv_t = model(xb)
                    pv = pv_t.view(-1).detach().cpu().numpy()
                    preds.append(pv)
                    ys.append(yb.view(-1).cpu().numpy())
                    ws.append(wb.view(-1).cpu().numpy())
            return np.concatenate(preds), np.concatenate(ys), np.concatenate(ws)

        # Always compute train IC each epoch so history plots show full train curve.
        preds_tr, ytr_all, wtr_all = _predict_all(train_loader)
        train_corr = weighted_corr(ytr_all, preds_tr, wtr_all)

        if do_log:
            msg = (
                f"[Epoch {epoch + 1}/{max_epochs}] "
                f"train_loss={train_loss:.4f}, train_ic={train_corr:.4f}, "
                f"valid_loss={vloss:.4f}, valid_ic={valid_corr:.4f}, "
                f"best_epoch={best_epoch}, best_metric_value={(best_score if np.isfinite(best_score) else float('nan')):.6f}, "
                f"lr={opt.param_groups[0]['lr']:.2e}"
            )
            print(msg)
            print("")

        if scheduler is not None:
            if scheduler_type == "plateau_ic":
                if np.isfinite(valid_ic):
                    scheduler.step(valid_ic)
            else:
                scheduler.step()

        best_value = valid_ic if best_metric == "ic" else vloss
        if best_metric == "ic":
            improved_best = np.isfinite(best_value) and (best_value > best_score + early_stop_min_delta)
        else:
            improved_best = np.isfinite(best_value) and (best_value < best_score - early_stop_min_delta)
        if improved_best:
            best_score = float(best_value)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_id_state = {k: v.detach().clone() for k, v in id_embed.state_dict().items()} if id_embed is not None else None
            best_epoch = epoch + 1

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "train_corr": float(train_corr),
                "valid_loss": float(vloss),
                "valid_corr": float(valid_corr),
                "valid_ic": float(valid_ic),
                "lr": float(opt.param_groups[0]["lr"]),
                "weight_decay": float(weight_decay),
                "grad_clip_norm": float(grad_clip_norm),
                "optim": optim_name,
                "best_epoch": int(best_epoch),
                "best_metric_value": float(best_score) if np.isfinite(best_score) else np.nan,
            }
        )
        if do_log:
            _flush_history()
        early_value = valid_ic if early_stop_metric == "ic" else vloss
        if early_stop_metric == "ic":
            improved_early = np.isfinite(early_value) and (early_value > early_best_score + early_stop_min_delta)
        else:
            improved_early = np.isfinite(early_value) and (early_value < early_best_score - early_stop_min_delta)
        if improved_early:
            early_best_score = float(early_value)
            patience_count = 0
        else:
            if early_stop_patience > 0:
                patience_count += 1
                if patience_count >= early_stop_patience:
                    msg = (
                        f"Early stop at epoch {epoch + 1} by {early_stop_metric} "
                        f"(best_epoch={best_epoch}, best_metric={best_metric}, best_value={best_score:.6f}, "
                        f"curr_valid_ic={valid_ic:.6f}, curr_valid_loss={vloss:.6f})"
                    )
                    if show_progress:
                        tqdm.write(msg)
                    else:
                        print(msg)
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    if id_embed is not None and best_id_state is not None:
        id_embed.load_state_dict(best_id_state)

    print(f"Best epoch by {best_metric}: {best_epoch}, best_value={best_score:.6f}")

    _flush_history()

    def predict_fn(Xnew, id_new=None):
        model.eval()
        with torch.no_grad():
            xn = torch.from_numpy(_prepare_X(Xnew)).to(dev)
            if use_id_embedding and id_embed is not None:
                ids_src = id_new if id_new is not None else id_predict
                if ids_src is None or len(ids_src) != len(Xnew):
                    ids = np.zeros(len(Xnew), dtype=np.int64)
                else:
                    ids = np.asarray(ids_src, dtype=np.int64)
                it = torch.from_numpy(ids).to(dev)
                emb = id_embed(it)
                pred = model(xn, emb) if model_type in {"cnn", "rnn", "lstm", "gru"} else model(torch.cat([xn, emb], dim=1))
                return pred.view(-1).cpu().numpy()
            return model(xn).view(-1).cpu().numpy()

    return model, predict_fn

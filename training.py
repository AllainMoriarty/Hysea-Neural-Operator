"""
Training loops, Optuna objective factories, evaluation and visualisation.

No top-level execution — all public names are pure functions.
No MLflow calls — logging is the caller's responsibility (task scripts).

Memory optimisations applied
-----------------------------
• AMP (BF16)               : autocast + GradScaler on all training loops
• Gradient accumulation    : GRAD_ACCUM_STEPS micro-batches per optimizer step
• CPU best-state snapshot  : model checkpoint stored in CPU RAM, not VRAM
• Cache flush              : gc.collect() + cuda.empty_cache() between stages
• non_blocking transfers   : overlaps H2D copy with computation
• DataLoader               : pin_memory=False, spawn-safe num_workers
• val_rmse / evaluate      : AMP autocast during inference
• PINO physics loss        : reuses pre-computed pred (no double forward)
"""
from __future__ import annotations
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.amp import autocast, GradScaler
from config import (
    device,
    USE_AMP, GRAD_ACCUM_STEPS, NUM_WORKERS, PIN_MEMORY, PREFETCH_FACTOR,
)
from models import MFDeepONet, MFFno, MFPino


# ── Data helpers ──────────────────────────────────────────────────────────────

class _PairDataset(Dataset):
    """Dataset wrapper for lazy HDF5 views (or any indexable pair of arrays)."""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return min(len(self.X), len(self.y))

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y


def make_loader(X, y, batch_size: int, shuffle: bool = True) -> DataLoader:
    """Build a DataLoader from either H5FieldViews or plain numpy arrays."""
    pf = PREFETCH_FACTOR if NUM_WORKERS > 0 else None
    pw = NUM_WORKERS > 0

    if getattr(X, "is_h5_view", False) or getattr(y, "is_h5_view", False):
        ds = _PairDataset(X, y)
    else:
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=PIN_MEMORY,
        num_workers=NUM_WORKERS,
        prefetch_factor=pf,
        persistent_workers=pw,
    )


def _ensure_aux_device(aux, model_type: str):
    """Move DeepONet query tensors to the compute device if needed."""
    if model_type == "deeponet" and isinstance(aux, torch.Tensor):
        return aux.to(device, non_blocking=True)
    return aux


def _infer_out_channels(y) -> int:
    """Infer FNO/PINO output channels from target sample shape."""
    sample = y[0]
    if isinstance(sample, torch.Tensor):
        sample = sample.detach().cpu().numpy()
    else:
        sample = np.asarray(sample)

    # y sample shapes:
    #   (H, W)       -> scalar field
    #   (C, H, W)    -> multi-channel field (e.g. eta with C=NTIME)
    if sample.ndim == 2:
        return 1
    if sample.ndim == 3:
        return int(sample.shape[0])
    return 1


def val_rmse(model, model_type: str, X_val, y_val, aux, fidelity: str,
             batch: int = 32) -> float:
    model.eval()
    aux = _ensure_aux_device(aux, model_type)
    losses = []
    loader = make_loader(X_val, y_val, batch_size=batch, shuffle=False)
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with autocast("cuda", dtype=torch.bfloat16, enabled=USE_AMP):
                pred = model(Xb, aux, fidelity=fidelity)
                if pred.dim() == 4:
                    pred = pred.squeeze(1).reshape(len(Xb), -1)
                    yb   = yb.squeeze(1).reshape(len(Xb), -1) if yb.dim() == 4 else yb
            # MSE in float32 for stability
            losses.append(F.mse_loss(pred.float(), yb.float()).item())
    return float(np.mean(losses)) ** 0.5


# ── Optuna objective factories ────────────────────────────────────────────────
# Each factory returns a closure that captures its dataset arguments.

def make_objective_deeponet(X_tr, y_tr, X_va, y_va, q, trunk_dim: int = 2,
                             epochs: int = 20):
    """Return an Optuna objective closure for MFDeepONet."""
    def objective(trial):
        p      = trial.suggest_categorical("p",      [64, 128, 256])
        hidden = trial.suggest_categorical("hidden", [128, 256, 512])
        lr     = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        wd     = trial.suggest_float("wd", 1e-6, 1e-3, log=True)
        batch  = trial.suggest_categorical("batch",  [16, 32, 64])

        model  = MFDeepONet(p=p, hidden=hidden, trunk_dim=trunk_dim).to(device)
        q_dev  = _ensure_aux_device(q, "deeponet")
        opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        scaler = GradScaler(enabled=USE_AMP)
        loader = make_loader(X_tr, y_tr, batch)

        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            for i, (Xb, yb) in enumerate(loader):
                Xb, yb = Xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                with autocast("cuda", dtype=torch.bfloat16, enabled=USE_AMP):
                    loss = F.mse_loss(model(Xb, q_dev, fidelity="lf"), yb) / GRAD_ACCUM_STEPS
                scaler.scale(loss).backward()
                if (i + 1) % GRAD_ACCUM_STEPS == 0 or (i + 1) == len(loader):
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad()

        result = val_rmse(model, "deeponet", X_va, y_va, q_dev, "lf")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return result
    return objective


def make_objective_fno(X_tr, y_tr, X_va, y_va, bathy_t, nlat: int, nlon: int,
                        epochs: int = 15):
    """Return an Optuna objective closure for MFFno."""
    def objective(trial):
        # Flush fragmented CUDA cache from previous trial before allocating.
        gc.collect()
        torch.cuda.empty_cache()

        # Search space capped for 349x780 grid:
        #   decoder (B * NLAT * NLON, dec_sz) at B=4, dec_sz=128 → ~540 MB/layer
        #   latent=64, modes=16, batch=8 → OOM confirmed on 31 GiB GPU
        latent = trial.suggest_categorical("latent_channels",    [16, 32])
        layers = trial.suggest_int("num_fno_layers",              2, 4)
        modes  = trial.suggest_categorical("num_fno_modes",      [8, 12])
        dec_sz = trial.suggest_categorical("decoder_layer_size", [64, 128])
        lr     = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        wd     = trial.suggest_float("wd", 1e-6, 1e-3, log=True)
        batch  = trial.suggest_categorical("batch", [2, 4])
        out_channels = _infer_out_channels(y_tr)

        model  = MFFno(nlat=nlat, nlon=nlon,
                       latent_channels=latent, num_fno_layers=layers,
                   num_fno_modes=modes, decoder_layer_size=dec_sz,
                   out_channels=out_channels).to(device)
        opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        scaler = GradScaler(enabled=USE_AMP)
        loader = make_loader(X_tr, y_tr, batch)

        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            for i, (Xb, yb) in enumerate(loader):
                Xb, yb = Xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                with autocast("cuda", dtype=torch.bfloat16, enabled=USE_AMP):
                    loss = F.mse_loss(model(Xb, bathy_t, fidelity="lf"), yb) / GRAD_ACCUM_STEPS
                scaler.scale(loss).backward()
                if (i + 1) % GRAD_ACCUM_STEPS == 0 or (i + 1) == len(loader):
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad()

        result = val_rmse(model, "fno", X_va, y_va, bathy_t, "lf")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return result
    return objective


def make_objective_pino(X_tr, y_tr, X_va, y_va, bathy_t, nlat: int, nlon: int,
                         epochs: int = 15):
    """Return an Optuna objective closure for MFPino (tunes lambda_pde too)."""
    def objective(trial):
        # Flush fragmented CUDA cache from previous trial before allocating.
        gc.collect()
        torch.cuda.empty_cache()

        # Same conservative search space as FNO — PINO has identical backbone.
        latent     = trial.suggest_categorical("latent_channels",    [16, 32])
        layers     = trial.suggest_int("num_fno_layers",              2, 4)
        modes      = trial.suggest_categorical("num_fno_modes",      [8, 12])
        dec_sz     = trial.suggest_categorical("decoder_layer_size", [64, 128])
        lr         = trial.suggest_float("lr",          1e-4, 5e-3, log=True)
        wd         = trial.suggest_float("wd",          1e-6, 1e-3, log=True)
        batch      = trial.suggest_categorical("batch", [2, 4])
        lambda_pde = trial.suggest_float("lambda_pde",  1e-4, 1e-1, log=True)
        out_channels = _infer_out_channels(y_tr)

        model  = MFPino(nlat=nlat, nlon=nlon,
                        latent_channels=latent, num_fno_layers=layers,
                num_fno_modes=modes, decoder_layer_size=dec_sz,
                out_channels=out_channels).to(device)
        opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        scaler = GradScaler(enabled=USE_AMP)
        loader = make_loader(X_tr, y_tr, batch)

        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            for i, (Xb, yb) in enumerate(loader):
                Xb, yb = Xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                with autocast("cuda", dtype=torch.bfloat16, enabled=USE_AMP):
                    pred      = model(Xb, bathy_t, fidelity="lf")
                    data_loss = F.mse_loss(pred, yb)
                    phys_loss = model.pino_physics_loss(Xb, bathy_t, fidelity="lf", pred=pred)
                    loss      = (data_loss + lambda_pde * phys_loss) / GRAD_ACCUM_STEPS
                scaler.scale(loss).backward()
                if (i + 1) % GRAD_ACCUM_STEPS == 0 or (i + 1) == len(loader):
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad()

        result = val_rmse(model, "fno", X_va, y_va, bathy_t, "lf")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return result
    return objective


# ── Training loops ────────────────────────────────────────────────────────────

def train_mf(model, model_type: str, lf_data, mf_data, hf_data, aux,
             best_params: dict, epochs_lf: int = 300, epochs_mf: int = 150,
             epochs_hf: int = 75) -> dict:
    """
    3-stage multi-fidelity training for MFDeepONet or MFFno.

    Parameters
    ----------
    *_data      : (X_tr, X_va, y_tr, y_va) tuple per fidelity
    aux         : q_sp/q_spt for DeepONet, bathy_t for FNO
    best_params : dict with at least 'lr', 'wd', 'batch'

    Returns
    -------
    history dict with per-epoch RMSE lists for each stage
    """
    lr    = best_params.get("lr",    1e-3)
    wd    = best_params.get("wd",    1e-5)
    batch = best_params.get("batch", 4)
    aux = _ensure_aux_device(aux, model_type)

    history = {k: [] for k in
               ["lf_train", "lf_val", "mf_train", "mf_val", "hf_train", "hf_val"]}

    def _run_stage(params, loader, val_X, val_y, fidelity, epochs, tag):
        opt    = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        scaler = GradScaler(enabled=USE_AMP)
        best_val, best_state = float("inf"), None

        for ep in range(epochs):
            model.train()
            ep_loss = []
            opt.zero_grad()

            for i, (Xb, yb) in enumerate(loader):
                Xb = Xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                with autocast("cuda", dtype=torch.bfloat16, enabled=USE_AMP):
                    pred = model(Xb, aux, fidelity=fidelity)
                    if pred.dim() == 4:
                        pred = pred.squeeze(1)
                        yb   = yb.squeeze(1) if yb.dim() == 4 else yb
                    loss = F.mse_loss(pred, yb) / GRAD_ACCUM_STEPS

                # Track unscaled RMSE for logging
                ep_loss.append((loss.item() * GRAD_ACCUM_STEPS) ** 0.5)
                scaler.scale(loss).backward()

                if (i + 1) % GRAD_ACCUM_STEPS == 0 or (i + 1) == len(loader):
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad()

            sched.step()

            vl = val_rmse(model, model_type, val_X, val_y, aux, fidelity)
            history[f"{tag}_train"].append(float(np.mean(ep_loss)))
            history[f"{tag}_val"].append(vl)

            if vl < best_val:
                best_val  = vl
                # Store best checkpoint in CPU RAM — not GPU VRAM
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if (ep + 1) % max(1, epochs // 5) == 0:
                print(f"  [{tag.upper()}] Ep {ep+1:3d} | "
                      f"Train RMSE: {np.mean(ep_loss):.4f} | Val RMSE: {vl:.4f}")

        if best_state:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        del best_state
        gc.collect()
        torch.cuda.empty_cache()

    X_lf_tr, X_lf_va, y_lf_tr, y_lf_va = lf_data
    X_mf_tr, X_mf_va, y_mf_tr, y_mf_va = mf_data
    X_hf_tr, X_hf_va, y_hf_tr, y_hf_va = hf_data

    # Stage 1: LF
    print("\nStage 1: LF Training")
    lf_params = (
        list(model.lf.parameters()) if model_type == "deeponet"
        else list(model.fno.parameters()) + list(model.feat_embed.parameters())
    )
    _run_stage(lf_params, make_loader(X_lf_tr, y_lf_tr, batch),
               X_lf_va, y_lf_va, "lf", epochs_lf, "lf")

    for p_ in (model.lf if model_type == "deeponet" else model.fno).parameters():
        p_.requires_grad = False

    # Stage 2: MF correction
    print("\nStage 2: MF Correction")
    mf_params = (
        list(model.mf.parameters()) + [model.alpha_mf] if model_type == "deeponet"
        else list(model.mf_correction.parameters()) + [model.alpha_mf]
    )
    _run_stage(mf_params, make_loader(X_mf_tr, y_mf_tr, batch),
               X_mf_va, y_mf_va, "mf", epochs_mf, "mf")

    for p_ in (model.mf if model_type == "deeponet"
               else model.mf_correction).parameters():
        p_.requires_grad = False
    model.alpha_mf.requires_grad = False

    # Stage 3: HF correction
    print("\nStage 3: HF Correction")
    hf_params = (
        list(model.hf.parameters()) + [model.alpha_hf] if model_type == "deeponet"
        else list(model.hf_correction.parameters()) + [model.alpha_hf]
    )
    _run_stage(hf_params, make_loader(X_hf_tr, y_hf_tr, batch),
               X_hf_va, y_hf_va, "hf", epochs_hf, "hf")

    return history


def train_mf_pino(model: MFPino, lf_data, mf_data, hf_data, aux,
                  best_params: dict, epochs_lf: int = 300, epochs_mf: int = 150,
                  epochs_hf: int = 75) -> dict:
    """
    3-stage MF training for MFPino.

    Stage 1 (LF) : L = L_data + lambda_pde * L_physics  ← PINO active
    Stages 2 & 3 : data loss only (correction nets, LF backbone frozen)

    Physics loss reuses the pre-computed pred to avoid a second forward pass.
    """
    lr         = best_params.get("lr",         1e-3)
    wd         = best_params.get("wd",         1e-5)
    batch      = best_params.get("batch",      4)
    lambda_pde = best_params.get("lambda_pde", 1e-2)

    history = {k: [] for k in
               ["lf_train", "lf_val", "mf_train", "mf_val", "hf_train", "hf_val"]}

    def _run_stage(params, loader, val_X, val_y, fidelity, epochs, tag,
                   physics: bool = False):
        opt    = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        scaler = GradScaler(enabled=USE_AMP)
        best_val, best_state = float("inf"), None

        for ep in range(epochs):
            model.train()
            ep_loss = []
            opt.zero_grad()

            for i, (Xb, yb) in enumerate(loader):
                Xb = Xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                with autocast("cuda", dtype=torch.bfloat16, enabled=USE_AMP):
                    pred_full = model(Xb, aux, fidelity=fidelity)
                    pred_loss = pred_full
                    yb_loss = yb

                    if pred_loss.dim() == 4 and pred_loss.shape[1] == 1:
                        pred_loss = pred_loss.squeeze(1)
                        yb_loss = yb_loss.squeeze(1) if yb_loss.dim() == 4 else yb_loss

                    data_loss = F.mse_loss(pred_loss, yb_loss)
                    if physics:
                        # Pass pre-computed pred to avoid a second full forward pass
                        phys_loss = model.pino_physics_loss(
                            Xb, aux, fidelity=fidelity, pred=pred_full
                        )
                        loss = (data_loss + lambda_pde * phys_loss) / GRAD_ACCUM_STEPS
                    else:
                        loss = data_loss / GRAD_ACCUM_STEPS

                ep_loss.append(data_loss.item() ** 0.5)   # track data RMSE only
                scaler.scale(loss).backward()

                if (i + 1) % GRAD_ACCUM_STEPS == 0 or (i + 1) == len(loader):
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad()

            sched.step()

            vl = val_rmse(model, "fno", val_X, val_y, aux, fidelity)
            history[f"{tag}_train"].append(float(np.mean(ep_loss)))
            history[f"{tag}_val"].append(vl)

            if vl < best_val:
                best_val  = vl
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if (ep + 1) % max(1, epochs // 5) == 0:
                print(f"  [PINO-{tag.upper()}] Ep {ep+1:3d} | "
                      f"Data RMSE: {np.mean(ep_loss):.4f} | Val RMSE: {vl:.4f}")

        if best_state:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        del best_state
        gc.collect()
        torch.cuda.empty_cache()

    X_lf_tr, X_lf_va, y_lf_tr, y_lf_va = lf_data
    X_mf_tr, X_mf_va, y_mf_tr, y_mf_va = mf_data
    X_hf_tr, X_hf_va, y_hf_tr, y_hf_va = hf_data

    print("\nStage 1 (PINO): LF Training with physics residual")
    _run_stage(
        list(model.fno.parameters()) + list(model.feat_embed.parameters()),
        make_loader(X_lf_tr, y_lf_tr, batch),
        X_lf_va, y_lf_va, "lf", epochs_lf, "lf", physics=True,
    )
    for p_ in model.fno.parameters():
        p_.requires_grad = False

    print("\nStage 2 (PINO): MF Correction")
    _run_stage(
        list(model.mf_correction.parameters()) + [model.alpha_mf],
        make_loader(X_mf_tr, y_mf_tr, batch),
        X_mf_va, y_mf_va, "mf", epochs_mf, "mf", physics=False,
    )
    for p_ in model.mf_correction.parameters():
        p_.requires_grad = False
    model.alpha_mf.requires_grad = False

    print("\nStage 3 (PINO): HF Correction")
    _run_stage(
        list(model.hf_correction.parameters()) + [model.alpha_hf],
        make_loader(X_hf_tr, y_hf_tr, batch),
        X_hf_va, y_hf_va, "hf", epochs_hf, "hf", physics=False,
    )

    return history


# ── Evaluation & visualisation ────────────────────────────────────────────────

def evaluate(model, model_type: str, X_te, y_te_flat, aux, name: str,
             fidelity: str = "hf", inverse_fn=None, batch: int = 16) -> dict:
    """Run inference on the test set and compute metrics."""
    model.eval()
    aux = _ensure_aux_device(aux, model_type)
    preds = []
    trues = []
    loader = make_loader(X_te, y_te_flat, batch_size=batch, shuffle=False)
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device, non_blocking=True)
            with autocast("cuda", dtype=torch.bfloat16, enabled=USE_AMP):
                pred = model(Xb, aux, fidelity=fidelity)
                if pred.dim() == 4:
                    pred = pred.reshape(len(Xb), -1)
            preds.append(pred.float().cpu().numpy())
            if yb.dim() > 2:
                trues.append(yb.reshape(len(yb), -1).cpu().numpy())
            else:
                trues.append(yb.cpu().numpy())

    pred_flat = np.concatenate(preds, axis=0)
    true_flat = np.concatenate(trues, axis=0)

    if inverse_fn:
        pred_flat = inverse_fn(pred_flat)
        true_flat = inverse_fn(true_flat)

    pr, tr = pred_flat.ravel(), true_flat.ravel()
    eps    = 1e-3

    mae   = mean_absolute_error(tr, pr)
    rmse  = root_mean_squared_error(tr, pr)
    r2    = r2_score(tr, pr)
    mask  = np.abs(tr) > eps
    rel   = np.mean(np.abs(pr[mask] - tr[mask]) / np.abs(tr[mask]))
    nrmse = rmse / (np.std(tr) + 1e-8)

    print(f"\n── {name} [{fidelity.upper()}] ")
    print(f"  MAE:            {mae:.4f}")
    print(f"  RMSE:           {rmse:.4f}")
    print(f"  NRMSE:          {nrmse:.4f}")
    print(f"  R²:             {r2:.4f}")
    print(f"  Relative error: {rel*100:.2f}%")

    return dict(model=name, fidelity=fidelity,
                mae=mae, rmse=rmse, nrmse=nrmse, r2=r2,
                rel_err=rel * 100, pred=pred_flat, true=true_flat)


def plot_maps(result: dict, lon, lat, nlat: int, nlon: int,
              out_path: str = None, n: int = 3):
    """Plot true / predicted / error spatial maps for up to n test samples."""
    p2d = result["pred"].reshape(-1, nlat, nlon)
    t2d = result["true"].reshape(-1, nlat, nlon)

    n_avail = min(len(p2d), len(t2d))
    n_plot = min(n, n_avail)
    if n_plot == 0:
        return None

    fig, axes = plt.subplots(n_plot, 3, figsize=(18, 5 * n_plot))
    axes = np.atleast_2d(axes)

    for i in range(n_plot):
        vmax = max(float(t2d[i].max()), float(p2d[i].max()))
        for ax, data, title in zip(
            axes[i],
            [t2d[i], p2d[i], np.abs(p2d[i] - t2d[i])],
            ["True", "Predicted", "|Error|"],
        ):
            cmap  = "jet" if title != "|Error|" else "Reds"
            vmax_ = vmax if title != "|Error|" else None
            im    = ax.pcolormesh(lon, lat, data, cmap=cmap, vmin=0, vmax=vmax_)
            ax.set_title(
                f"{title} — {result['model']} [{result['fidelity'].upper()}]"
            )
            plt.colorbar(im, ax=ax)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return fig

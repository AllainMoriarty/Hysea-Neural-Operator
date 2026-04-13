"""
Training loops, Optuna objective factories, evaluation and visualisation.

No top-level execution — all public names are pure functions.
No MLflow calls — logging is the caller's responsibility (task scripts).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from config import device
from models import MFDeepONet, MFFno, MFPino


# Data helpers 
def make_loader(X, y, batch_size: int, shuffle: bool = True) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


def val_rmse(model, model_type: str, X_val, y_val, aux, fidelity: str, batch: int = 32) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(0, len(X_val), batch):
            Xb = torch.tensor(X_val[i:i+batch], dtype=torch.float32).to(device)
            yb = torch.tensor(y_val[i:i+batch], dtype=torch.float32).to(device)
            pred = model(Xb, aux, fidelity=fidelity)
            if pred.dim() == 4:
                pred = pred.squeeze(1).reshape(len(Xb), -1)
                yb   = yb.squeeze(1).reshape(len(Xb), -1) if yb.dim() == 4 else yb
            losses.append(F.mse_loss(pred, yb).item())
    return float(np.mean(losses)) ** 0.5


# Optuna objective factories
# Each factory returns a closure that captures its dataset arguments.
def make_objective_deeponet(X_tr, y_tr, X_va, y_va, q, trunk_dim: int = 2, epochs: int = 20):
    """Return an Optuna objective closure for MFDeepONet."""
    def objective(trial):
        p      = trial.suggest_categorical("p",      [64, 128, 256])
        hidden = trial.suggest_categorical("hidden", [128, 256, 512])
        lr     = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        wd     = trial.suggest_float("wd", 1e-6, 1e-3, log=True)
        batch  = trial.suggest_categorical("batch",  [16, 32, 64])

        model  = MFDeepONet(p=p, hidden=hidden, trunk_dim=trunk_dim).to(device)
        opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        loader = make_loader(X_tr, y_tr, batch)

        model.train()
        for _ in range(epochs):
            for Xb, yb in loader:
                Xb, yb = Xb.to(device), yb.to(device)
                opt.zero_grad()
                F.mse_loss(model(Xb, q, fidelity="lf"), yb).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        return val_rmse(model, "deeponet", X_va, y_va, q, "lf")
    return objective


def make_objective_fno(X_tr, y_tr, X_va, y_va, bathy_t, nlat: int, nlon: int, epochs: int = 15):
    """Return an Optuna objective closure for MFFno."""
    def objective(trial):
        latent = trial.suggest_categorical("latent_channels",    [16, 32, 64])
        layers = trial.suggest_int("num_fno_layers",              2, 6)
        modes  = trial.suggest_categorical("num_fno_modes",      [8, 16, 24])
        dec_sz = trial.suggest_categorical("decoder_layer_size", [64, 128, 256])
        lr     = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        wd     = trial.suggest_float("wd", 1e-6, 1e-3, log=True)
        batch  = trial.suggest_categorical("batch", [8, 16, 32])

        model  = MFFno(nlat=nlat, nlon=nlon,
                       latent_channels=latent, num_fno_layers=layers,
                       num_fno_modes=modes, decoder_layer_size=dec_sz).to(device)
        opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        loader = make_loader(X_tr, y_tr, batch)

        model.train()
        for _ in range(epochs):
            for Xb, yb in loader:
                Xb, yb = Xb.to(device), yb.to(device)
                opt.zero_grad()
                F.mse_loss(model(Xb, bathy_t, fidelity="lf"), yb).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        return val_rmse(model, "fno", X_va, y_va, bathy_t, "lf")
    return objective


def make_objective_pino(X_tr, y_tr, X_va, y_va, bathy_t, nlat: int, nlon: int, epochs: int = 15):
    """Return an Optuna objective closure for MFPino (tunes lambda_pde too)."""
    def objective(trial):
        latent     = trial.suggest_categorical("latent_channels",    [16, 32, 64])
        layers     = trial.suggest_int("num_fno_layers",              2, 6)
        modes      = trial.suggest_categorical("num_fno_modes",      [8, 16, 24])
        dec_sz     = trial.suggest_categorical("decoder_layer_size", [64, 128, 256])
        lr         = trial.suggest_float("lr",          1e-4, 5e-3, log=True)
        wd         = trial.suggest_float("wd",          1e-6, 1e-3, log=True)
        batch      = trial.suggest_categorical("batch", [8, 16, 32])
        lambda_pde = trial.suggest_float("lambda_pde",  1e-4, 1e-1, log=True)

        model  = MFPino(nlat=nlat, nlon=nlon,
                        latent_channels=latent, num_fno_layers=layers,
                        num_fno_modes=modes, decoder_layer_size=dec_sz).to(device)
        opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        loader = make_loader(X_tr, y_tr, batch)

        model.train()
        for _ in range(epochs):
            for Xb, yb in loader:
                Xb, yb = Xb.to(device), yb.to(device)
                opt.zero_grad()
                pred      = model(Xb, bathy_t, fidelity="lf")
                data_loss = F.mse_loss(pred, yb)
                phys_loss = model.pino_physics_loss(Xb, bathy_t, fidelity="lf")
                (data_loss + lambda_pde * phys_loss).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        return val_rmse(model, "fno", X_va, y_va, bathy_t, "lf")
    return objective


# Training loops
def train_mf(model, model_type: str, lf_data, mf_data, hf_data, aux, best_params: dict, epochs_lf: int = 300, epochs_mf: int = 150, epochs_hf: int = 75) -> dict:
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
    batch = best_params.get("batch", 32)

    history = {k: [] for k in
               ["lf_train", "lf_val", "mf_train", "mf_val", "hf_train", "hf_val"]}

    def _run_stage(params, loader, val_X, val_y, fidelity, epochs, tag):
        opt   = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        best_val, best_state = float("inf"), None

        for ep in range(epochs):
            model.train()
            ep_loss = []
            for Xb, yb in loader:
                Xb, yb = Xb.to(device), yb.to(device)
                opt.zero_grad()
                pred = model(Xb, aux, fidelity=fidelity)
                if pred.dim() == 4:
                    pred = pred.squeeze(1)
                    yb   = yb.squeeze(1)
                loss = F.mse_loss(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_loss.append(loss.item() ** 0.5)
            sched.step()

            vl = val_rmse(model, model_type, val_X, val_y, aux, fidelity)
            history[f"{tag}_train"].append(float(np.mean(ep_loss)))
            history[f"{tag}_val"].append(vl)

            if vl < best_val:
                best_val   = vl
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            if (ep + 1) % max(1, epochs // 5) == 0:
                print(f"  [{tag.upper()}] Ep {ep+1:3d} | "
                      f"Train RMSE: {np.mean(ep_loss):.4f} | Val RMSE: {vl:.4f}")

        if best_state:
            model.load_state_dict(best_state)

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


def train_mf_pino(model: MFPino, lf_data, mf_data, hf_data, aux, best_params: dict, epochs_lf: int = 300, epochs_mf: int = 150, epochs_hf: int = 75) -> dict:
    """
    3-stage MF training for MFPino.

    Stage 1 (LF) : L = L_data + lambda_pde * L_physics  ← PINO active
    Stages 2 & 3 : data loss only (correction nets, LF backbone frozen)
    """
    lr         = best_params.get("lr",         1e-3)
    wd         = best_params.get("wd",         1e-5)
    batch      = best_params.get("batch",      32)
    lambda_pde = best_params.get("lambda_pde", 1e-2)

    history = {k: [] for k in
               ["lf_train", "lf_val", "mf_train", "mf_val", "hf_train", "hf_val"]}

    def _run_stage(params, loader, val_X, val_y, fidelity, epochs, tag, physics: bool = False):
        opt   = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        best_val, best_state = float("inf"), None

        for ep in range(epochs):
            model.train()
            ep_loss = []
            for Xb, yb in loader:
                Xb, yb = Xb.to(device), yb.to(device)
                opt.zero_grad()
                pred      = model(Xb, aux, fidelity=fidelity)
                if pred.dim() == 4:
                    pred = pred.squeeze(1)
                    yb   = yb.squeeze(1)
                data_loss = F.mse_loss(pred, yb)
                if physics:
                    phys_loss = model.pino_physics_loss(Xb, aux, fidelity=fidelity)
                    loss = data_loss + lambda_pde * phys_loss
                else:
                    loss = data_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_loss.append(data_loss.item() ** 0.5)  # track data RMSE only
            sched.step()

            vl = val_rmse(model, "fno", val_X, val_y, aux, fidelity)
            history[f"{tag}_train"].append(float(np.mean(ep_loss)))
            history[f"{tag}_val"].append(vl)

            if vl < best_val:
                best_val   = vl
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            if (ep + 1) % max(1, epochs // 5) == 0:
                print(f"  [PINO-{tag.upper()}] Ep {ep+1:3d} | "
                      f"Data RMSE: {np.mean(ep_loss):.4f} | Val RMSE: {vl:.4f}")

        if best_state:
            model.load_state_dict(best_state)

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

# Evaluation & visualisation
def evaluate(model, model_type: str, X_te, y_te_flat, aux, name: str, fidelity: str = "hf", inverse_fn=None, batch: int = 16) -> dict:
    """Run inference on the test set and compute metrics."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_te), batch):
            Xb   = torch.tensor(X_te[i:i+batch], dtype=torch.float32).to(device)
            pred = model(Xb, aux, fidelity=fidelity)
            if pred.dim() == 4:
                pred = pred.reshape(len(Xb), -1)
            preds.append(pred.cpu().numpy())

    pred_flat = np.concatenate(preds, axis=0)
    true_flat = (y_te_flat.reshape(len(X_te), -1)
                 if y_te_flat.ndim > 2 else y_te_flat)

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


def plot_maps(result: dict, lon, lat, nlat: int, nlon: int, out_path: str = None, n: int = 3):
    """Plot true / predicted / error spatial maps for n test samples."""
    p2d = result["pred"].reshape(-1, nlat, nlon)
    t2d = result["true"].reshape(-1, nlat, nlon)
    fig, axes = plt.subplots(n, 3, figsize=(18, 5 * n))
    for i in range(n):
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

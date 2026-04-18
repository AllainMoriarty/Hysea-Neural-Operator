import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import optuna
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# ── PhysicsNeMo imports ──────────────────────────────────────────────────
from physicsnemo.models.fno import FNO
from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.models.fourier_net import FourierNetArch
from physicsnemo.sym.models.deeponet import DeepONetArch
from physicsnemo.sym.key import Key

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device:          {device}")
print(f"PhysicsNeMo FNO: physicsnemo.models.fno.FNO")
print(f"DeepONet:        physicsnemo.sym.models.deeponet.DeepONetArch")


# ============================================================
# 1. LOAD DATASETS
# ============================================================
def load_h5(path):
    with h5py.File(path, 'r') as hf:
        return {k: hf[k][:] for k in hf.keys()}

print("\nLoading datasets...")
lf      = load_h5('/kaggle/input/datasets/dataset_5km_full.h5')
mf      = load_h5('/kaggle/input/datasets/dataset_2km_full.h5')
hf_data = load_h5('/kaggle/input/datasets/dataset_1km_full.h5')

N_LF, NLAT, NLON = lf['max_height'].shape
NTIME = lf['eta'].shape[1]
NPTS  = NLAT * NLON
print(f"LF:{N_LF}  MF:{mf['features'].shape[0]}  HF:{hf_data['features'].shape[0]}")
print(f"Grid: ({NLAT}×{NLON})  Time steps: {NTIME}")


# ============================================================
# 2. PREPROCESSING
# ============================================================
def preprocess(data, feat_scaler=None, fit=False):
    X = data['features'].astype(np.float32)
    if fit:
        feat_scaler = StandardScaler()
        X_s = feat_scaler.fit_transform(X)
    else:
        X_s = feat_scaler.transform(X)

    at_max  = float(data['arrival_times'].max())
    eta_max = float(np.abs(data['eta']).max())

    # Flat targets for DeepONet
    y_mh  = np.log1p(np.clip(data['max_height'], 0, 50)
                     .reshape(len(X), -1)).astype(np.float32)
    y_at  = (data['arrival_times'].reshape(len(X), -1)
             / (at_max + 1e-8)).astype(np.float32)
    y_eta = (data['eta'].reshape(len(X), -1)
             / (eta_max + 1e-8)).astype(np.float32)

    # 2D targets for FNO — shape (N, C, NLAT, NLON)
    y_mh_2d  = np.log1p(np.clip(data['max_height'], 0, 50)
                        ).astype(np.float32)[:, None]          # (N,1,NLAT,NLON)
    y_at_2d  = (data['arrival_times'] / (at_max + 1e-8)
                ).astype(np.float32)[:, None]                  # (N,1,NLAT,NLON)
    y_eta_2d = (data['eta'] / (eta_max + 1e-8)
                ).astype(np.float32)                           # (N,NTIME,NLAT,NLON)

    return (X_s, y_mh, y_at, y_eta,
            y_mh_2d, y_at_2d, y_eta_2d,
            feat_scaler, at_max, eta_max)

(X_lf, ymh_lf, yat_lf, yeta_lf,
 ymh2d_lf, yat2d_lf, yeta2d_lf,
 feat_scaler, at_lf, eta_lf) = preprocess(lf, fit=True)

(X_mf, ymh_mf, yat_mf, yeta_mf,
 ymh2d_mf, yat2d_mf, yeta2d_mf,
 _, at_mf, eta_mf) = preprocess(mf, feat_scaler)

(X_hf, ymh_hf, yat_hf, yeta_hf,
 ymh2d_hf, yat2d_hf, yeta2d_hf,
 _, at_hf, eta_hf) = preprocess(hf_data, feat_scaler)

# Query points for DeepONet trunk
lon_grid, lat_grid = np.meshgrid(lf['lon'], lf['lat'])
spatial_pts = np.stack([lon_grid.ravel(), lat_grid.ravel()], axis=1).astype(np.float32)
t_norm = (lf['time'] / lf['time'].max()).astype(np.float32)
lon_t, lat_t, t_t = np.meshgrid(lf['lon'], lf['lat'], t_norm)
spt_pts = np.stack([lon_t.ravel(), lat_t.ravel(), t_t.ravel()], axis=1).astype(np.float32)

sp_scaler  = StandardScaler(); spatial_scaled = sp_scaler.fit_transform(spatial_pts)
spt_scaler = StandardScaler(); spt_scaled     = spt_scaler.fit_transform(spt_pts)

q_sp  = torch.tensor(spatial_scaled, dtype=torch.float32).to(device)  # (NPTS, 2)
q_spt = torch.tensor(spt_scaled,     dtype=torch.float32).to(device)  # (NPTS*NTIME, 3)

# Bathymetry for FNO input
bathy_np   = lf['bathymetry'].astype(np.float32)
bathy_norm = (bathy_np - bathy_np.mean()) / (bathy_np.std() + 1e-8)
bathy_t    = torch.tensor(bathy_norm, dtype=torch.float32).to(device)  # (NLAT, NLON)


# ============================================================
# 3. TRAIN/VAL/TEST SPLITS
# ============================================================
def make_splits(*arrays, test=0.10, val=0.12, seed=42):
    idx = np.arange(len(arrays[0]))
    tv, te = train_test_split(idx, test_size=test, random_state=seed)
    tr, va = train_test_split(tv,  test_size=val,  random_state=seed)
    return [(a[tr], a[va], a[te]) for a in arrays]

(X_lf_tr,  X_lf_va,  X_lf_te),  \
(ymh_lf_tr, ymh_lf_va, ymh_lf_te), \
(yat_lf_tr, yat_lf_va, yat_lf_te), \
(yeta_lf_tr,yeta_lf_va,yeta_lf_te),\
(ymh2d_lf_tr,ymh2d_lf_va,ymh2d_lf_te), \
(yat2d_lf_tr,yat2d_lf_va,yat2d_lf_te), \
(yeta2d_lf_tr,yeta2d_lf_va,yeta2d_lf_te) = make_splits(
    X_lf, ymh_lf, yat_lf, yeta_lf, ymh2d_lf, yat2d_lf, yeta2d_lf)

(X_mf_tr,  X_mf_va,  X_mf_te),  \
(ymh_mf_tr, ymh_mf_va, ymh_mf_te), \
(yat_mf_tr, yat_mf_va, yat_mf_te), \
(yeta_mf_tr,yeta_mf_va,yeta_mf_te),\
(ymh2d_mf_tr,ymh2d_mf_va,ymh2d_mf_te), \
(yat2d_mf_tr,yat2d_mf_va,yat2d_mf_te), \
(yeta2d_mf_tr,yeta2d_mf_va,yeta2d_mf_te) = make_splits(
    X_mf, ymh_mf, yat_mf, yeta_mf, ymh2d_mf, yat2d_mf, yeta2d_mf)

(X_hf_tr,  X_hf_va,  X_hf_te),  \
(ymh_hf_tr, ymh_hf_va, ymh_hf_te), \
(yat_hf_tr, yat_hf_va, yat_hf_te), \
(yeta_hf_tr,yeta_hf_va,yeta_hf_te),\
(ymh2d_hf_tr,ymh2d_hf_va,ymh2d_hf_te), \
(yat2d_hf_tr,yat2d_hf_va,yat2d_hf_te), \
(yeta2d_hf_tr,yeta2d_hf_va,yeta2d_hf_te) = make_splits(
    X_hf, ymh_hf, yat_hf, yeta_hf, ymh2d_hf, yat2d_hf, yeta2d_hf)

print(f"LF  train:{len(X_lf_tr)}  val:{len(X_lf_va)}  test:{len(X_lf_te)}")
print(f"MF  train:{len(X_mf_tr)}  val:{len(X_mf_va)}  test:{len(X_mf_te)}")
print(f"HF  train:{len(X_hf_tr)}  val:{len(X_hf_va)}  test:{len(X_hf_te)}")


# ============================================================
# 4. DEEPONET — PhysicsNeMo Sym DeepONetArch
# ============================================================
# Based on docs: DeepONetArch takes branch_net + trunk_net
# Both built with FullyConnectedArch or FourierNetArch

def build_deeponet(p=128, hidden=256, trunk_dim=2, n_fault_params=9):
    """
    Build DeepONet using PhysicsNeMo DeepONetArch.
    Branch: fault_params (9,) → encoding (p,)
    Trunk:  query coords (trunk_dim,) → basis functions (p,)
    """
    branch_net = FullyConnectedArch(
        input_keys=[Key("fault_params", size=n_fault_params)],
        output_keys=[Key("branch", size=p)],
        layer_size=hidden,
        nr_layers=4,
        activation_fn="gelu",
    )

    trunk_net = FourierNetArch(
        input_keys=[Key("query_coords", size=trunk_dim)],
        output_keys=[Key("trunk", size=p)],
        layer_size=hidden,
        nr_layers=4,
        frequencies=("axis", [i for i in range(10)]),
    )

    deeponet = DeepONetArch(
        output_keys=[Key("field_output")],
        branch_net=branch_net,
        trunk_net=trunk_net,
        branch_dim=p,
        trunk_dim=p,
    )
    return deeponet


class MFDeepONet(nn.Module):
    """
    Multi-Fidelity DeepONet wrapper.
    3 levels: LF → MF correction → HF correction
    output = f_LF + sigmoid(α_MF)*δ_MF + sigmoid(α_HF)*δ_HF
    """
    def __init__(self, p=128, hidden=256, trunk_dim=2):
        super().__init__()
        self.lf  = build_deeponet(p, hidden, trunk_dim)
        self.mf  = build_deeponet(p, hidden, trunk_dim)
        self.hf  = build_deeponet(p, hidden, trunk_dim)
        self.alpha_mf = nn.Parameter(torch.tensor(0.0))
        self.alpha_hf = nn.Parameter(torch.tensor(0.0))
        self.trunk_dim = trunk_dim

    def _forward_one(self, net, fault_params, query_coords):
        """Call a DeepONetArch with dict inputs."""
        out = net({
            "fault_params":  fault_params,   # (B, 9)
            "query_coords":  query_coords,   # (NPTS, trunk_dim)
        })
        return out["field_output"]           # (B, NPTS)

    def forward(self, fault_params, query_coords, fidelity='hf'):
        lf_out = self._forward_one(self.lf, fault_params, query_coords)
        if fidelity == 'lf':
            return lf_out
        mf_out = lf_out + torch.sigmoid(self.alpha_mf) * \
                 self._forward_one(self.mf, fault_params, query_coords)
        if fidelity == 'mf':
            return mf_out
        hf_out = mf_out + torch.sigmoid(self.alpha_hf) * \
                 self._forward_one(self.hf, fault_params, query_coords)
        return hf_out


# ============================================================
# 5. FNO — PhysicsNeMo physicsnemo.models.fno.FNO
# ============================================================
# Based on docs: FNO(in_channels, out_channels, dimension=2,
#                    latent_channels, num_fno_layers, num_fno_modes,
#                    decoder_layers, decoder_layer_size, padding)
# Input:  (N, in_channels, NLAT, NLON)
# Output: (N, out_channels, NLAT, NLON)

class MFFno(nn.Module):
    """
    Multi-Fidelity FNO wrapper.
    Input channels: fault_params embedded into spatial field
                    + bathymetry + coord_grid (handled by coord_features=True)
    """
    def __init__(self, latent_channels=32, num_fno_layers=4,
                 num_fno_modes=16, decoder_layers=2,
                 decoder_layer_size=128, n_fault_params=9,
                 nlat=NLAT, nlon=NLON, out_channels=1):
        super().__init__()
        self.nlat = nlat
        self.nlon = nlon
        self.n_feat = n_fault_params

        # Embed fault params → spatial field (1 channel)
        self.feat_embed = nn.Sequential(
            nn.Linear(n_fault_params, 256), nn.GELU(),
            nn.Linear(256, nlat * nlon),
        )

        # in_channels = feat_embed(1) + bathymetry(1) = 2
        # coord_features=True adds 2 more → effective 4 channels internally
        self.fno = FNO(
            in_channels=2,
            out_channels=out_channels,
            decoder_layers=decoder_layers,
            decoder_layer_size=decoder_layer_size,
            dimension=2,
            latent_channels=latent_channels,
            num_fno_layers=num_fno_layers,
            num_fno_modes=num_fno_modes,
            padding=8,
            padding_type="constant",
            activation_fn="gelu",
            coord_features=True,     # adds grid coords automatically
        ).to(device)

        # LF/MF/HF correction FNOs (smaller)
        correction_kwargs = dict(
            in_channels=2, out_channels=out_channels,
            decoder_layers=1, decoder_layer_size=64,
            dimension=2, latent_channels=16,
            num_fno_layers=2, num_fno_modes=8,
            padding=4, coord_features=True,
        )
        self.mf_correction = FNO(**correction_kwargs).to(device)
        self.hf_correction = FNO(**correction_kwargs).to(device)
        self.alpha_mf = nn.Parameter(torch.tensor(0.0))
        self.alpha_hf = nn.Parameter(torch.tensor(0.0))

    def _build_input(self, fault_params, bathy_t):
        """Embed fault params + bathy into (B, 2, NLAT, NLON)."""
        B = fault_params.shape[0]
        feat = self.feat_embed(fault_params).view(B, 1, self.nlat, self.nlon)
        bathy = bathy_t.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
        return torch.cat([feat, bathy], dim=1)          # (B, 2, NLAT, NLON)

    def forward(self, fault_params, bathy_t, fidelity='hf'):
        x = self._build_input(fault_params, bathy_t)
        lf_out = self.fno(x)                            # (B, 1, NLAT, NLON)
        if fidelity == 'lf':
            return lf_out
        mf_out = lf_out + torch.sigmoid(self.alpha_mf) * self.mf_correction(x)
        if fidelity == 'mf':
            return mf_out
        hf_out = mf_out + torch.sigmoid(self.alpha_hf) * self.hf_correction(x)
        return hf_out


# ============================================================
# 5b. PINO — Physics-Informed FNO
# ============================================================
# PINO = FNO backbone + physics-informed loss term (PDE residual).
# Following the PhysicsNeMo PINO pattern (darcy_pino tutorial), the
# physics constraint is baked into the *training loss*, not the
# architecture.  The backbone is identical to MFFno.
#
# Physics residual chosen here: spectral Laplacian smoothness penalty
#   R(u) = || ∇²u ||²  (mean over batch & pixels)
# This regularises the predicted field to satisfy the elliptic
# equation  ∇²u ≈ 0  that smooth solutions of many geophysical PDEs
# obey far from the source.  The weight λ_pde controls the trade-off.

def spectral_laplacian_residual(u: torch.Tensor) -> torch.Tensor:
    """
    Compute the Laplacian of a 2D spatial field using spectral derivatives
    (FFT), as used in the PhysicsNeMo PINO tutorial.

    Args:
        u: Tensor of shape (B, C, H, W) — model output field.

    Returns:
        Scalar — mean squared Laplacian over batch / channels / spatial dims.
    """
    B, C, H, W = u.shape
    # Build wavenumber grids (normalised so the period is 2π)
    kx = torch.fft.fftfreq(W, d=1.0 / W).to(u.device)  # (W,)
    ky = torch.fft.fftfreq(H, d=1.0 / H).to(u.device)  # (H,)
    ky_grid, kx_grid = torch.meshgrid(ky, kx, indexing='ij')   # (H, W)
    # -(k_x² + k_y²) is the Fourier multiplier for the Laplacian
    laplacian_mult = -(kx_grid ** 2 + ky_grid ** 2)             # (H, W)

    u_hat    = torch.fft.fft2(u)                                 # (B, C, H, W) complex
    lap_hat  = laplacian_mult.unsqueeze(0).unsqueeze(0) * u_hat  # broadcast
    lap_u    = torch.fft.ifft2(lap_hat).real                    # (B, C, H, W)
    return (lap_u ** 2).mean()


class MFPino(nn.Module):
    """
    Multi-Fidelity PINO wrapper.

    Architecture is identical to MFFno.  The physics-informed loss
    (spectral Laplacian residual) is applied externally during training
    via `pino_physics_loss()`, following the PhysicsNeMo pattern where
    the physics constraint is part of the loss, not the forward pass.

    multi-fidelity correction:
        output = f_LF + sigmoid(α_MF)*δ_MF + sigmoid(α_HF)*δ_HF
    """
    def __init__(self, latent_channels=32, num_fno_layers=4,
                 num_fno_modes=16, decoder_layers=2,
                 decoder_layer_size=128, n_fault_params=9,
                 nlat=NLAT, nlon=NLON, out_channels=1):
        super().__init__()
        self.nlat = nlat
        self.nlon = nlon

        # Fault-parameter embedding (shared across LF/MF/HF FNOs)
        self.feat_embed = nn.Sequential(
            nn.Linear(n_fault_params, 256), nn.GELU(),
            nn.Linear(256, nlat * nlon),
        )

        # Primary LF FNO (this is the "PINO" backbone)
        self.fno = FNO(
            in_channels=2,
            out_channels=out_channels,
            decoder_layers=decoder_layers,
            decoder_layer_size=decoder_layer_size,
            dimension=2,
            latent_channels=latent_channels,
            num_fno_layers=num_fno_layers,
            num_fno_modes=num_fno_modes,
            padding=8,
            padding_type="constant",
            activation_fn="gelu",
            coord_features=True,
        ).to(device)

        # Smaller correction networks for MF / HF stages
        _corr = dict(
            in_channels=2, out_channels=out_channels,
            decoder_layers=1, decoder_layer_size=64,
            dimension=2, latent_channels=16,
            num_fno_layers=2, num_fno_modes=8,
            padding=4, coord_features=True,
        )
        self.mf_correction = FNO(**_corr).to(device)
        self.hf_correction = FNO(**_corr).to(device)
        self.alpha_mf = nn.Parameter(torch.tensor(0.0))
        self.alpha_hf = nn.Parameter(torch.tensor(0.0))

    def _build_input(self, fault_params, bathy_t):
        """Embed fault params + bathy → (B, 2, NLAT, NLON)."""
        B    = fault_params.shape[0]
        feat = self.feat_embed(fault_params).view(B, 1, self.nlat, self.nlon)
        bathy = bathy_t.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
        return torch.cat([feat, bathy], dim=1)

    def forward(self, fault_params, bathy_t, fidelity='hf'):
        x      = self._build_input(fault_params, bathy_t)
        lf_out = self.fno(x)                                     # (B,1,H,W)
        if fidelity == 'lf':
            return lf_out
        mf_out = lf_out + torch.sigmoid(self.alpha_mf) * self.mf_correction(x)
        if fidelity == 'mf':
            return mf_out
        hf_out = mf_out + torch.sigmoid(self.alpha_hf) * self.hf_correction(x)
        return hf_out

    def pino_physics_loss(self, fault_params, bathy_t, fidelity='lf'):
        """
        Physics-informed loss term (PINO residual).
        Returns the mean squared spectral Laplacian of the predicted field.
        A well-resolved tsunami max-height field satisfies ∇²u ≈ 0
        away from the source, so this acts as a soft PDE constraint.
        """
        pred = self.forward(fault_params, bathy_t, fidelity=fidelity)
        return spectral_laplacian_residual(pred)


# ============================================================
# 6. HELPERS
# ============================================================
def make_loader(X, y, batch_size, shuffle=True):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size,
                      shuffle=shuffle, pin_memory=True)

def val_rmse(model, model_type, X_val, y_val, aux, fidelity, batch=32):
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(0, len(X_val), batch):
            Xb = torch.tensor(X_val[i:i+batch], dtype=torch.float32).to(device)
            yb = torch.tensor(y_val[i:i+batch], dtype=torch.float32).to(device)
            if model_type == 'deeponet':
                pred = model(Xb, aux, fidelity=fidelity)
            else:
                pred = model(Xb, aux, fidelity=fidelity)
                if pred.dim() == 4:
                    pred = pred.squeeze(1).reshape(len(Xb), -1)
                    yb   = yb.squeeze(1).reshape(len(Xb), -1) if yb.dim()==4 else yb
            losses.append(F.mse_loss(pred, yb).item())
    return float(np.mean(losses)) ** 0.5


# ============================================================
# 7. HYPERPARAMETER TUNING — OPTUNA
# ============================================================
def objective_deeponet(trial):
    p       = trial.suggest_categorical('p',       [64, 128, 256])
    hidden  = trial.suggest_categorical('hidden',  [128, 256, 512])
    lr      = trial.suggest_float('lr',            1e-4, 5e-3, log=True)
    wd      = trial.suggest_float('wd',            1e-6, 1e-3, log=True)
    batch   = trial.suggest_categorical('batch',   [16, 32, 64])
    epochs  = 20

    model = MFDeepONet(p=p, hidden=hidden, trunk_dim=2).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loader = make_loader(X_lf_tr, ymh_lf_tr, batch)

    model.train()
    for _ in range(epochs):
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            F.mse_loss(model(Xb, q_sp, fidelity='lf'), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    return val_rmse(model, 'deeponet', X_lf_va, ymh_lf_va, q_sp, 'lf')


def objective_fno(trial):
    latent  = trial.suggest_categorical('latent_channels', [16, 32, 64])
    layers  = trial.suggest_int('num_fno_layers',           2, 6)
    modes   = trial.suggest_categorical('num_fno_modes',   [8, 16, 24])
    dec_sz  = trial.suggest_categorical('decoder_layer_size', [64, 128, 256])
    lr      = trial.suggest_float('lr',                     1e-4, 5e-3, log=True)
    wd      = trial.suggest_float('wd',                     1e-6, 1e-3, log=True)
    batch   = trial.suggest_categorical('batch',            [8, 16, 32])
    epochs  = 15

    model = MFFno(latent_channels=latent, num_fno_layers=layers,
                  num_fno_modes=modes, decoder_layer_size=dec_sz).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loader = make_loader(X_lf_tr, ymh2d_lf_tr, batch)

    model.train()
    for _ in range(epochs):
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(Xb, bathy_t, fidelity='lf')
            F.mse_loss(pred, yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    return val_rmse(model, 'fno', X_lf_va, ymh2d_lf_va, bathy_t, 'lf')


def objective_pino(trial):
    """
    Optuna objective for MFPino.
    Tunes all FNO hyper-params PLUS the physics-loss weight (lambda_pde).
    """
    latent     = trial.suggest_categorical('latent_channels',    [16, 32, 64])
    layers     = trial.suggest_int('num_fno_layers',              2, 6)
    modes      = trial.suggest_categorical('num_fno_modes',      [8, 16, 24])
    dec_sz     = trial.suggest_categorical('decoder_layer_size', [64, 128, 256])
    lr         = trial.suggest_float('lr',          1e-4, 5e-3, log=True)
    wd         = trial.suggest_float('wd',          1e-6, 1e-3, log=True)
    batch      = trial.suggest_categorical('batch', [8, 16, 32])
    lambda_pde = trial.suggest_float('lambda_pde',  1e-4, 1e-1, log=True)
    epochs     = 15

    model  = MFPino(latent_channels=latent, num_fno_layers=layers,
                    num_fno_modes=modes, decoder_layer_size=dec_sz).to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loader = make_loader(X_lf_tr, ymh2d_lf_tr, batch)

    model.train()
    for _ in range(epochs):
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            pred       = model(Xb, bathy_t, fidelity='lf')
            data_loss  = F.mse_loss(pred, yb)
            phys_loss  = model.pino_physics_loss(Xb, bathy_t, fidelity='lf')
            loss       = data_loss + lambda_pde * phys_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    return val_rmse(model, 'fno', X_lf_va, ymh2d_lf_va, bathy_t, 'lf')


print("\n" + "="*60)
print("Optuna Tuning — DeepONet (20 trials)")
study_don = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42))
study_don.optimize(objective_deeponet, n_trials=20, show_progress_bar=True)
best_don = study_don.best_params
print(f"Best DeepONet: {best_don}")

print("\n" + "="*60)
print("Optuna Tuning — FNO (20 trials)")
study_fno = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42))
study_fno.optimize(objective_fno, n_trials=20, show_progress_bar=True)
best_fno = study_fno.best_params
print(f"Best FNO: {best_fno}")

print("\n" + "="*60)
print("Optuna Tuning — PINO (20 trials)")
study_pino = optuna.create_study(direction='minimize',
                                 sampler=optuna.samplers.TPESampler(seed=42))
study_pino.optimize(objective_pino, n_trials=20, show_progress_bar=True)
best_pino = study_pino.best_params
print(f"Best PINO: {best_pino}")


# ============================================================
# 8. FULL MULTI-FIDELITY TRAINING
# ============================================================
def train_mf(model, model_type,
             lf_data, mf_data, hf_data_splits,
             aux, best_params,
             epochs_lf=300, epochs_mf=150, epochs_hf=75):
    """
    3-stage multi-fidelity training.
    lf_data/mf_data/hf_data_splits = (X_tr, X_va, y_tr, y_va)
    """
    lr    = best_params.get('lr',    1e-3)
    wd    = best_params.get('wd',    1e-5)
    batch = best_params.get('batch', 32)

    history = {'lf_train': [], 'lf_val': [],
               'mf_train': [], 'mf_val': [],
               'hf_train': [], 'hf_val': []}

    def run_stage(params, loader, val_X, val_y, fidelity, epochs, tag):
        opt  = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        best_val, best_state = float('inf'), None

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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_loss.append(loss.item() ** 0.5)
            sched.step()

            vl = val_rmse(model, model_type, val_X, val_y, aux, fidelity)
            history[f'{tag}_train'].append(np.mean(ep_loss))
            history[f'{tag}_val'].append(vl)

            if vl < best_val:
                best_val = vl
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            if (ep+1) % (epochs // 5) == 0:
                print(f"  [{tag.upper()}] Ep {ep+1:3d} | "
                      f"Train RMSE: {np.mean(ep_loss):.4f} | Val RMSE: {vl:.4f}")

        if best_state:
            model.load_state_dict(best_state)

    X_lf_tr_, X_lf_va_, y_lf_tr_, y_lf_va_ = lf_data
    X_mf_tr_, X_mf_va_, y_mf_tr_, y_mf_va_ = mf_data
    X_hf_tr_, X_hf_va_, y_hf_tr_, y_hf_va_ = hf_data_splits

    # ── Stage 1: LF ─────────────────────────────────────────
    print("\n── Stage 1: LF Training ──")
    lf_params = (list(model.lf.parameters())
                 if model_type == 'deeponet'
                 else list(model.fno.parameters()) +
                      list(model.feat_embed.parameters()))
    run_stage(lf_params,
              make_loader(X_lf_tr_, y_lf_tr_, batch),
              X_lf_va_, y_lf_va_, 'lf', epochs_lf, 'lf')

    # Freeze LF
    for p_ in (model.lf if model_type == 'deeponet' else model.fno).parameters():
        p_.requires_grad = False

    # ── Stage 2: MF correction ───────────────────────────────
    print("\n── Stage 2: MF Correction ──")
    mf_params = (list(model.mf.parameters()) + [model.alpha_mf]
                 if model_type == 'deeponet'
                 else list(model.mf_correction.parameters()) + [model.alpha_mf])
    run_stage(mf_params,
              make_loader(X_mf_tr_, y_mf_tr_, batch),
              X_mf_va_, y_mf_va_, 'mf', epochs_mf, 'mf')

    # Freeze MF
    for p_ in (model.mf if model_type == 'deeponet'
               else model.mf_correction).parameters():
        p_.requires_grad = False
    model.alpha_mf.requires_grad = False

    # ── Stage 3: HF correction ───────────────────────────────
    print("\n── Stage 3: HF Correction ──")
    hf_params = (list(model.hf.parameters()) + [model.alpha_hf]
                 if model_type == 'deeponet'
                 else list(model.hf_correction.parameters()) + [model.alpha_hf])
    run_stage(hf_params,
              make_loader(X_hf_tr_, y_hf_tr_, batch),
              X_hf_va_, y_hf_va_, 'hf', epochs_hf, 'hf')

    return history


def train_mf_pino(model,
                  lf_data, mf_data, hf_data_splits,
                  aux, best_params,
                  epochs_lf=300, epochs_mf=150, epochs_hf=75):
    """
    3-stage multi-fidelity training for MFPino.

    Identical to train_mf but the LF stage adds a physics-informed
    loss term (PDE residual) scaled by `lambda_pde` from best_params.
    Following the PhysicsNeMo PINO pattern, the physics constraint is
    encoded in the loss: L = L_data + λ * L_physics.
    """
    lr         = best_params.get('lr',         1e-3)
    wd         = best_params.get('wd',         1e-5)
    batch      = best_params.get('batch',      32)
    lambda_pde = best_params.get('lambda_pde', 1e-2)

    history = {'lf_train': [], 'lf_val': [],
               'mf_train': [], 'mf_val': [],
               'hf_train': [], 'hf_val': []}

    def run_stage(params, loader, val_X, val_y, fidelity, epochs, tag,
                  physics=False):
        opt   = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        best_val, best_state = float('inf'), None

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
                    # PINO physics residual — spectral Laplacian penalty
                    phys_loss = model.pino_physics_loss(Xb, aux,
                                                        fidelity=fidelity)
                    loss = data_loss + lambda_pde * phys_loss
                else:
                    loss = data_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_loss.append(data_loss.item() ** 0.5)  # track data RMSE
            sched.step()

            vl = val_rmse(model, 'fno', val_X, val_y, aux, fidelity)
            history[f'{tag}_train'].append(np.mean(ep_loss))
            history[f'{tag}_val'].append(vl)

            if vl < best_val:
                best_val = vl
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            if (ep+1) % (epochs // 5) == 0:
                print(f"  [PINO-{tag.upper()}] Ep {ep+1:3d} | "
                      f"Data RMSE: {np.mean(ep_loss):.4f} | Val RMSE: {vl:.4f}")

        if best_state:
            model.load_state_dict(best_state)

    X_lf_tr_, X_lf_va_, y_lf_tr_, y_lf_va_ = lf_data
    X_mf_tr_, X_mf_va_, y_mf_tr_, y_mf_va_ = mf_data
    X_hf_tr_, X_hf_va_, y_hf_tr_, y_hf_va_ = hf_data_splits

    # ── Stage 1: LF (with PINO physics loss) ────────────────
    print("\n── Stage 1 (PINO): LF Training with physics residual ──")
    lf_params = (list(model.fno.parameters()) +
                 list(model.feat_embed.parameters()))
    run_stage(lf_params,
              make_loader(X_lf_tr_, y_lf_tr_, batch),
              X_lf_va_, y_lf_va_, 'lf', epochs_lf, 'lf',
              physics=True)          # ← physics constraint active here

    # Freeze LF backbone
    for p_ in model.fno.parameters():
        p_.requires_grad = False

    # ── Stage 2: MF correction (data-only; LF frozen) ────────
    print("\n── Stage 2 (PINO): MF Correction ──")
    mf_params = list(model.mf_correction.parameters()) + [model.alpha_mf]
    run_stage(mf_params,
              make_loader(X_mf_tr_, y_mf_tr_, batch),
              X_mf_va_, y_mf_va_, 'mf', epochs_mf, 'mf',
              physics=False)

    for p_ in model.mf_correction.parameters():
        p_.requires_grad = False
    model.alpha_mf.requires_grad = False

    # ── Stage 3: HF correction (data-only) ──────────────────
    print("\n── Stage 3 (PINO): HF Correction ──")
    hf_params = list(model.hf_correction.parameters()) + [model.alpha_hf]
    run_stage(hf_params,
              make_loader(X_hf_tr_, y_hf_tr_, batch),
              X_hf_va_, y_hf_va_, 'hf', epochs_hf, 'hf',
              physics=False)

    return history


# ── Train DeepONet: max_height ───────────────────────────────
print("\n" + "="*60)
print("Training MF-DeepONet: max_height")
don_mh = MFDeepONet(p=best_don['p'], hidden=best_don['hidden'],
                    trunk_dim=2).to(device)
hist_don_mh = train_mf(
    don_mh, 'deeponet',
    (X_lf_tr, X_lf_va, ymh_lf_tr, ymh_lf_va),
    (X_mf_tr, X_mf_va, ymh_mf_tr, ymh_mf_va),
    (X_hf_tr, X_hf_va, ymh_hf_tr, ymh_hf_va),
    q_sp, best_don
)
torch.save(don_mh.state_dict(), 'don_maxheight.pt')

# ── Train DeepONet: arrival_times ────────────────────────────
print("\n" + "="*60)
print("Training MF-DeepONet: arrival_times")
don_at = MFDeepONet(p=best_don['p'], hidden=best_don['hidden'],
                    trunk_dim=2).to(device)
hist_don_at = train_mf(
    don_at, 'deeponet',
    (X_lf_tr, X_lf_va, yat_lf_tr, yat_lf_va),
    (X_mf_tr, X_mf_va, yat_mf_tr, yat_mf_va),
    (X_hf_tr, X_hf_va, yat_hf_tr, yat_hf_va),
    q_sp, best_don
)
torch.save(don_at.state_dict(), 'don_arrival.pt')

# ── Train DeepONet: eta time series ─────────────────────────
print("\n" + "="*60)
print("Training MF-DeepONet: eta timeseries")
don_eta = MFDeepONet(p=best_don['p'], hidden=best_don['hidden'],
                     trunk_dim=3).to(device)
hist_don_eta = train_mf(
    don_eta, 'deeponet',
    (X_lf_tr, X_lf_va, yeta_lf_tr, yeta_lf_va),
    (X_mf_tr, X_mf_va, yeta_mf_tr, yeta_mf_va),
    (X_hf_tr, X_hf_va, yeta_hf_tr, yeta_hf_va),
    q_spt, best_don
)
torch.save(don_eta.state_dict(), 'don_eta.pt')

# ── Train FNO: max_height ────────────────────────────────────
print("\n" + "="*60)
print("Training MF-FNO: max_height")
fno_mh = MFFno(
    latent_channels=best_fno['latent_channels'],
    num_fno_layers=best_fno['num_fno_layers'],
    num_fno_modes=best_fno['num_fno_modes'],
    decoder_layer_size=best_fno['decoder_layer_size'],
    out_channels=1,
).to(device)
hist_fno_mh = train_mf(
    fno_mh, 'fno',
    (X_lf_tr, X_lf_va, ymh2d_lf_tr, ymh2d_lf_va),
    (X_mf_tr, X_mf_va, ymh2d_mf_tr, ymh2d_mf_va),
    (X_hf_tr, X_hf_va, ymh2d_hf_tr, ymh2d_hf_va),
    bathy_t, best_fno
)
torch.save(fno_mh.state_dict(), 'fno_maxheight.pt')

# ── Train FNO: eta time series ────────────────────────────────
print("\n" + "="*60)
print("Training MF-FNO: eta timeseries")
fno_eta = MFFno(
    latent_channels=best_fno['latent_channels'],
    num_fno_layers=best_fno['num_fno_layers'],
    num_fno_modes=best_fno['num_fno_modes'],
    decoder_layer_size=best_fno['decoder_layer_size'],
    out_channels=NTIME,
).to(device)
hist_fno_eta = train_mf(
    fno_eta, 'fno',
    (X_lf_tr, X_lf_va, yeta2d_lf_tr, yeta2d_lf_va),
    (X_mf_tr, X_mf_va, yeta2d_mf_tr, yeta2d_mf_va),
    (X_hf_tr, X_hf_va, yeta2d_hf_tr, yeta2d_hf_va),
    bathy_t, best_fno
)
torch.save(fno_eta.state_dict(), 'fno_eta.pt')

# ── Train PINO: max_height ────────────────────────────────────
print("\n" + "="*60)
print("Training MF-PINO: max_height")
pino_mh = MFPino(
    latent_channels=best_pino['latent_channels'],
    num_fno_layers=best_pino['num_fno_layers'],
    num_fno_modes=best_pino['num_fno_modes'],
    decoder_layer_size=best_pino['decoder_layer_size'],
    out_channels=1,
).to(device)
hist_pino_mh = train_mf_pino(
    pino_mh,
    (X_lf_tr, X_lf_va, ymh2d_lf_tr, ymh2d_lf_va),
    (X_mf_tr, X_mf_va, ymh2d_mf_tr, ymh2d_mf_va),
    (X_hf_tr, X_hf_va, ymh2d_hf_tr, ymh2d_hf_va),
    bathy_t, best_pino
)
torch.save(pino_mh.state_dict(), 'pino_maxheight.pt')

# ── Train PINO: eta time series ───────────────────────────────
print("\n" + "="*60)
print("Training MF-PINO: eta timeseries")
pino_eta = MFPino(
    latent_channels=best_pino['latent_channels'],
    num_fno_layers=best_pino['num_fno_layers'],
    num_fno_modes=best_pino['num_fno_modes'],
    decoder_layer_size=best_pino['decoder_layer_size'],
    out_channels=NTIME,
).to(device)
hist_pino_eta = train_mf_pino(
    pino_eta,
    (X_lf_tr, X_lf_va, yeta2d_lf_tr, yeta2d_lf_va),
    (X_mf_tr, X_mf_va, yeta2d_mf_tr, yeta2d_mf_va),
    (X_hf_tr, X_hf_va, yeta2d_hf_tr, yeta2d_hf_va),
    bathy_t, best_pino
)
torch.save(pino_eta.state_dict(), 'pino_eta.pt')


# ============================================================
# 9. EVALUATION
# ============================================================
def evaluate(model, model_type, X_te, y_te_flat,
             aux, name, fidelity='hf',
             inverse_fn=None, batch=16):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_te), batch):
            Xb = torch.tensor(X_te[i:i+batch], dtype=torch.float32).to(device)
            pred = model(Xb, aux, fidelity=fidelity)
            if pred.dim() == 4:
                pred = pred.reshape(len(Xb), -1)
            preds.append(pred.cpu().numpy())

    pred_flat = np.concatenate(preds, axis=0)
    true_flat = y_te_flat.reshape(len(X_te), -1) if y_te_flat.ndim > 2 else y_te_flat

    if inverse_fn:
        pred_flat = inverse_fn(pred_flat)
        true_flat = inverse_fn(true_flat)

    pr  = pred_flat.ravel()
    tr  = true_flat.ravel()
    eps = 1e-3

    mae   = mean_absolute_error(tr, pr)
    rmse  = root_mean_squared_error(tr, pr)
    r2    = r2_score(tr, pr)
    mask  = np.abs(tr) > eps
    rel   = np.mean(np.abs(pr[mask] - tr[mask]) / np.abs(tr[mask]))
    nrmse = rmse / (np.std(tr) + 1e-8)

    print(f"\n── {name} [{fidelity.upper()}] ──────────────────")
    print(f"  MAE:            {mae:.4f} m")
    print(f"  RMSE:           {rmse:.4f} m")
    print(f"  NRMSE:          {nrmse:.4f}")
    print(f"  R²:             {r2:.4f}")
    print(f"  Relative error: {rel*100:.2f}%")

    return dict(model=name, fidelity=fidelity,
                mae=mae, rmse=rmse, nrmse=nrmse, r2=r2,
                rel_err=rel*100,
                pred=pred_flat, true=true_flat)


results = []
for fid in ['lf', 'mf', 'hf']:
    Xte = {'lf': X_lf_te, 'mf': X_mf_te, 'hf': X_hf_te}[fid]
    yte = {'lf': ymh_lf_te, 'mf': ymh_mf_te, 'hf': ymh_hf_te}[fid]
    results.append(evaluate(don_mh, 'deeponet', Xte, yte,
                            q_sp, 'MF-DeepONet max_height', fid, np.expm1))

for fid in ['lf', 'mf', 'hf']:
    Xte  = {'lf': X_lf_te, 'mf': X_mf_te, 'hf': X_hf_te}[fid]
    yte  = {'lf': ymh2d_lf_te, 'mf': ymh2d_mf_te, 'hf': ymh2d_hf_te}[fid]
    results.append(evaluate(fno_mh, 'fno', Xte, yte,
                            bathy_t, 'MF-FNO max_height', fid, np.expm1))

for fid in ['lf', 'mf', 'hf']:
    Xte  = {'lf': X_lf_te, 'mf': X_mf_te, 'hf': X_hf_te}[fid]
    yte  = {'lf': ymh2d_lf_te, 'mf': ymh2d_mf_te, 'hf': ymh2d_hf_te}[fid]
    results.append(evaluate(pino_mh, 'fno', Xte, yte,
                            bathy_t, 'MF-PINO max_height', fid, np.expm1))


# ============================================================
# 10. COMPARISON TABLE + PLOTS
# ============================================================
df_res = pd.DataFrame([{k: v for k, v in r.items()
                         if k not in ('pred', 'true')}
                        for r in results])
df_res['mae']   = df_res['mae'].round(4)
df_res['rmse']  = df_res['rmse'].round(4)
df_res['nrmse'] = df_res['nrmse'].round(4)
df_res['r2']    = df_res['r2'].round(4)
df_res['rel_err'] = df_res['rel_err'].round(2)

print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)
print(df_res.to_string(index=False))
df_res.to_csv('results.csv', index=False)

# Loss curves
fig, axes = plt.subplots(1, 3, figsize=(21, 5))
for hist, label, ax in [(hist_don_mh,  'MF-DeepONet', axes[0]),
                         (hist_fno_mh,  'MF-FNO',      axes[1]),
                         (hist_pino_mh, 'MF-PINO',     axes[2])]:
    for stage in ['lf', 'mf', 'hf']:
        if f'{stage}_train' in hist:
            ax.semilogy(hist[f'{stage}_train'], label=f'{stage.upper()} train')
            ax.semilogy(hist[f'{stage}_val'],   label=f'{stage.upper()} val', ls='--')
    ax.set_title(f'{label} — max_height loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('RMSE'); ax.legend()
plt.tight_layout()
plt.savefig('loss_curves.png', dpi=150)
plt.show()

# Spatial maps
def plot_maps(result, lon, lat, n=3):
    p2d = result['pred'].reshape(-1, NLAT, NLON)
    t2d = result['true'].reshape(-1, NLAT, NLON)
    fig, axes = plt.subplots(n, 3, figsize=(18, 5*n))
    for i in range(n):
        vmax = max(t2d[i].max(), p2d[i].max())
        for ax, data, title in zip(axes[i],
                [t2d[i], p2d[i], np.abs(p2d[i]-t2d[i])],
                ['True', 'Predicted', '|Error|']):
            im = ax.pcolormesh(lon, lat, data, cmap='jet' if title!='|Error|' else 'Reds',
                               vmin=0, vmax=vmax if title!='|Error|' else None)
            ax.set_title(f"{title} — {result['model']} [{result['fidelity'].upper()}]")
            plt.colorbar(im, ax=ax, label='m')
    plt.tight_layout()
    plt.savefig(f"map_{result['model'].replace(' ','_')}_{result['fidelity']}.png", dpi=150)
    plt.show()

for r in results:
    plot_maps(r, lf['lon'], lf['lat'])

print("\n✓ Done! Saved: results.csv, loss_curves.png, map_*.png")
print("        Models: don_maxheight.pt, fno_maxheight.pt, pino_maxheight.pt, ...")
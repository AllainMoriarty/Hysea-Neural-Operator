from __future__ import annotations
import numpy as np
import torch
import h5py
from dataclasses import dataclass
from typing import Any, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from config import DATA_PATHS, device, RANDOM_SEED


# Low-level helpers
def load_h5(path: str) -> dict:
    """Read an HDF5 file and return all datasets as a dict of numpy arrays."""
    with h5py.File(path, "r") as hf:
        return {k: hf[k][:] for k in hf.keys()}


def preprocess(data: dict, feat_scaler=None, fit: bool = False):
    """
    Scale features and normalise targets.

    Returns
    -------
    X_s, y_mh, y_at, y_eta,
    y_mh_2d, y_at_2d, y_eta_2d,
    feat_scaler, at_max, eta_max
    """
    X = data["features"].astype(np.float32)
    if fit:
        feat_scaler = StandardScaler()
        X_s = feat_scaler.fit_transform(X)
    else:
        X_s = feat_scaler.transform(X)

    at_max  = float(data["arrival_times"].max())
    eta_max = float(np.abs(data["eta"]).max())
    N       = len(X)

    # Flat targets (N, NPTS or N, NPTS*NTIME) — for DeepONet
    y_mh  = np.log1p(np.clip(data["max_height"], 0, 50)
                     .reshape(N, -1)).astype(np.float32)
    y_at  = (data["arrival_times"].reshape(N, -1) / (at_max  + 1e-8)).astype(np.float32)
    y_eta = (data["eta"].reshape(N, -1)           / (eta_max + 1e-8)).astype(np.float32)

    # 2-D targets (N, C, NLAT, NLON) — for FNO / PINO
    y_mh_2d  = np.log1p(np.clip(data["max_height"], 0, 50)).astype(np.float32)[:, None]
    y_at_2d  = (data["arrival_times"] / (at_max  + 1e-8)).astype(np.float32)[:, None]
    y_eta_2d = (data["eta"]           / (eta_max + 1e-8)).astype(np.float32)

    return (X_s, y_mh, y_at, y_eta,
            y_mh_2d, y_at_2d, y_eta_2d,
            feat_scaler, at_max, eta_max)


def make_splits(*arrays, test: float = 0.10, val: float = 0.12, seed: int = RANDOM_SEED):
    """
    Split arrays into train / val / test.
    Returns a list of (arr_tr, arr_va, arr_te) tuples, one per input array.
    """
    idx = np.arange(len(arrays[0]))
    tv, te = train_test_split(idx, test_size=test, random_state=seed)
    tr, va = train_test_split(tv,  test_size=val,  random_state=seed)
    return [(a[tr], a[va], a[te]) for a in arrays]


def build_query_points(lf_raw: dict):
    """
    Build spatial and spatio-temporal query-point tensors for DeepONet trunk.
    Returns
    -------
    q_sp  : torch.Tensor  (NPTS, 2)        — 2-D spatial grid
    q_spt : torch.Tensor  (NPTS*NTIME, 3)  — spatio-temporal grid
    """
    lon_grid, lat_grid = np.meshgrid(lf_raw["lon"], lf_raw["lat"])
    spatial_pts = np.stack(
        [lon_grid.ravel(), lat_grid.ravel()], axis=1
    ).astype(np.float32)

    t_norm = (lf_raw["time"] / lf_raw["time"].max()).astype(np.float32)
    lon_t, lat_t, t_t = np.meshgrid(lf_raw["lon"], lf_raw["lat"], t_norm)
    spt_pts = np.stack(
        [lon_t.ravel(), lat_t.ravel(), t_t.ravel()], axis=1
    ).astype(np.float32)

    sp_scaler  = StandardScaler()
    spt_scaler = StandardScaler()
    spatial_scaled = sp_scaler.fit_transform(spatial_pts)
    spt_scaled     = spt_scaler.fit_transform(spt_pts)

    q_sp  = torch.tensor(spatial_scaled, dtype=torch.float32).to(device)
    q_spt = torch.tensor(spt_scaled,     dtype=torch.float32).to(device)
    return q_sp, q_spt


def build_bathy(lf_raw: dict) -> torch.Tensor:
    """Return normalised bathymetry tensor (NLAT, NLON) on the compute device."""
    bathy      = lf_raw["bathymetry"].astype(np.float32)
    bathy_norm = (bathy - bathy.mean()) / (bathy.std() + 1e-8)
    return torch.tensor(bathy_norm, dtype=torch.float32).to(device)


# DataBundle — holds everything a task script needs
def _to_dict(split_tuple) -> dict:
    """Convert a (tr, va, te) 3-tuple → {'tr': …, 'va': …, 'te': …}."""
    return {"tr": split_tuple[0], "va": split_tuple[1], "te": split_tuple[2]}


@dataclass
class DataBundle:
    # Grid dimensions
    NLAT:  int
    NLON:  int
    NTIME: int

    # Feature splits — access as db.X_lf["tr"], db.X_lf["va"], db.X_lf["te"]
    X_lf: Dict[str, np.ndarray]
    X_mf: Dict[str, np.ndarray]
    X_hf: Dict[str, np.ndarray]

    # Flat targets (DeepONet) — shape (N, NPTS) or (N, NPTS*NTIME)
    ymh_lf:  Dict[str, np.ndarray]
    ymh_mf:  Dict[str, np.ndarray]
    ymh_hf:  Dict[str, np.ndarray]
    yat_lf:  Dict[str, np.ndarray]
    yat_mf:  Dict[str, np.ndarray]
    yat_hf:  Dict[str, np.ndarray]
    yeta_lf: Dict[str, np.ndarray]
    yeta_mf: Dict[str, np.ndarray]
    yeta_hf: Dict[str, np.ndarray]

    # 2-D targets (FNO / PINO) — shape (N, C, NLAT, NLON)
    ymh2d_lf:  Dict[str, np.ndarray]
    ymh2d_mf:  Dict[str, np.ndarray]
    ymh2d_hf:  Dict[str, np.ndarray]
    yat2d_lf:  Dict[str, np.ndarray]
    yat2d_mf:  Dict[str, np.ndarray]
    yat2d_hf:  Dict[str, np.ndarray]
    yeta2d_lf: Dict[str, np.ndarray]
    yeta2d_mf: Dict[str, np.ndarray]
    yeta2d_hf: Dict[str, np.ndarray]

    # Auxiliary tensors
    q_sp:    torch.Tensor   # (NPTS, 2)        spatial query coords for DeepONet
    q_spt:   torch.Tensor   # (NPTS*NTIME, 3)  spatio-temporal query coords
    bathy_t: torch.Tensor   # (NLAT, NLON)     normalised bathymetry for FNO/PINO

    # Normalisation constants (used to inverse-transform predictions)
    at_lf:  float;  at_mf:  float;  at_hf:  float   # arrival_times scale
    eta_lf: float;  eta_mf: float;  eta_hf: float   # eta scale

    # Grid coordinates (used for spatial plots)
    lon: np.ndarray
    lat: np.ndarray

    # Feature scaler (may be needed for external inference)
    feat_scaler: Any


def load_dataset() -> DataBundle:
    """
    Load all three fidelity datasets, preprocess, split, and return a DataBundle.
    This is the only function in data.py that reads from disk.
    """
    print("Loading datasets…")
    lf_raw = load_h5(DATA_PATHS["lf"])
    mf_raw = load_h5(DATA_PATHS["mf"])
    hf_raw = load_h5(DATA_PATHS["hf"])

    _, NLAT, NLON = lf_raw["max_height"].shape
    NTIME         = lf_raw["eta"].shape[1]
    print(f"LF:{lf_raw['features'].shape[0]}  "
          f"MF:{mf_raw['features'].shape[0]}  "
          f"HF:{hf_raw['features'].shape[0]}")
    print(f"Grid: ({NLAT}×{NLON})  Time steps: {NTIME}")

    # Preprocess each fidelity
    (X_lf, ymh_lf, yat_lf, yeta_lf, ymh2d_lf, yat2d_lf, yeta2d_lf,
     feat_scaler, at_lf, eta_lf) = preprocess(lf_raw, fit=True)

    (X_mf, ymh_mf, yat_mf, yeta_mf, ymh2d_mf, yat2d_mf, yeta2d_mf,
     _, at_mf, eta_mf) = preprocess(mf_raw, feat_scaler=feat_scaler)

    (X_hf, ymh_hf, yat_hf, yeta_hf, ymh2d_hf, yat2d_hf, yeta2d_hf,
     _, at_hf, eta_hf) = preprocess(hf_raw, feat_scaler=feat_scaler)

    # Train / val / test splits
    # make_splits returns [(arr_tr, arr_va, arr_te), …] for each input array
    # Index map: 0→X, 1→ymh, 2→yat, 3→yeta, 4→ymh2d, 5→yat2d, 6→yeta2d
    s_lf = make_splits(X_lf, ymh_lf, yat_lf, yeta_lf, ymh2d_lf, yat2d_lf, yeta2d_lf)
    s_mf = make_splits(X_mf, ymh_mf, yat_mf, yeta_mf, ymh2d_mf, yat2d_mf, yeta2d_mf)
    s_hf = make_splits(X_hf, ymh_hf, yat_hf, yeta_hf, ymh2d_hf, yat2d_hf, yeta2d_hf)

    print(f"LF  train:{len(s_lf[0][0])}  val:{len(s_lf[0][1])}  test:{len(s_lf[0][2])}")
    print(f"MF  train:{len(s_mf[0][0])}  val:{len(s_mf[0][1])}  test:{len(s_mf[0][2])}")
    print(f"HF  train:{len(s_hf[0][0])}  val:{len(s_hf[0][1])}  test:{len(s_hf[0][2])}")

    # Auxiliary tensors
    q_sp, q_spt = build_query_points(lf_raw)
    bathy_t     = build_bathy(lf_raw)

    _d = _to_dict
    return DataBundle(
        NLAT=NLAT, NLON=NLON, NTIME=NTIME,

        X_lf=_d(s_lf[0]),  X_mf=_d(s_mf[0]),  X_hf=_d(s_hf[0]),

        ymh_lf=_d(s_lf[1]),   ymh_mf=_d(s_mf[1]),   ymh_hf=_d(s_hf[1]),
        yat_lf=_d(s_lf[2]),   yat_mf=_d(s_mf[2]),   yat_hf=_d(s_hf[2]),
        yeta_lf=_d(s_lf[3]),  yeta_mf=_d(s_mf[3]),  yeta_hf=_d(s_hf[3]),

        ymh2d_lf=_d(s_lf[4]),  ymh2d_mf=_d(s_mf[4]),  ymh2d_hf=_d(s_hf[4]),
        yat2d_lf=_d(s_lf[5]),  yat2d_mf=_d(s_mf[5]),  yat2d_hf=_d(s_hf[5]),
        yeta2d_lf=_d(s_lf[6]), yeta2d_mf=_d(s_mf[6]), yeta2d_hf=_d(s_hf[6]),

        q_sp=q_sp, q_spt=q_spt, bathy_t=bathy_t,

        at_lf=at_lf,   at_mf=at_mf,   at_hf=at_hf,
        eta_lf=eta_lf, eta_mf=eta_mf, eta_hf=eta_hf,

        lon=lf_raw["lon"], lat=lf_raw["lat"],
        feat_scaler=feat_scaler,
    )


def build_grid_info(lf_raw: dict, db: DataBundle) -> tuple:
    """
    Build the grid_info dict and raw-bathymetry tensor for MFPino's SWE physics loss.

    Must be called AFTER load_dataset() since it needs db.at_lf for T_max.

    Parameters
    ----------
    lf_raw : raw LF HDF5 dict (from load_h5)
    db     : DataBundle returned by load_dataset()

    Returns
    -------
    grid_info : dict with keys lat_rad, dlon, dlat, T_max
    H_raw     : torch.Tensor (NLAT, NLON) — raw bathymetry in metres on compute device
    """
    lon_rad = np.deg2rad(lf_raw["lon"].astype(np.float32))
    lat_rad = np.deg2rad(lf_raw["lat"].astype(np.float32))

    grid_info = {
        "lat_rad": torch.tensor(lat_rad, dtype=torch.float32).to(device),
        "dlon":    float(np.mean(np.diff(lon_rad))),
        "dlat":    float(np.abs(np.mean(np.diff(lat_rad)))),
        "T_max":   db.at_lf,   # arrival_times denorm factor [s]
    }

    bathy_raw = lf_raw["bathymetry"].astype(np.float32)
    H_raw = torch.tensor(bathy_raw, dtype=torch.float32).to(device)

    return grid_info, H_raw

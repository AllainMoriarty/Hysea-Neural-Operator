from __future__ import annotations
import gc
import numpy as np
import torch
import torch.nn.functional as F
import h5py
from dataclasses import dataclass
from typing import Any, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from config import DATA_PATHS, device, RANDOM_SEED


# Low-level helpers
def load_h5(path: str, keys: list[str] | None = None) -> dict:
    """Read selected HDF5 datasets into memory as numpy arrays."""
    with h5py.File(path, "r") as hf:
        if keys is None:
            keys = list(hf.keys())
        return {k: hf[k][:] for k in keys}


def preprocess(data: dict, feat_scaler=None, fit: bool = False, target: str = "all",
               mh_out_hw: tuple[int, int] | None = None,
               at_out_hw: tuple[int, int] | None = None,
               eta_out_hw: tuple[int, int] | None = None):
    """
    Scale features and normalise targets.

    The returned dict always contains keys:
    X, ymh, yat, yeta, ymh2d, yat2d, yeta2d,
    feat_scaler, at_max, eta_max.
    Unused targets are set to None.
    """
    X = data["features"].astype(np.float32)
    if fit:
        feat_scaler = StandardScaler()
        X_s = feat_scaler.fit_transform(X)
    else:
        X_s = feat_scaler.transform(X)

    out = {
        "X": X_s,
        "ymh": None,
        "yat": None,
        "yeta": None,
        "ymh2d": None,
        "yat2d": None,
        "yeta2d": None,
        "feat_scaler": feat_scaler,
        "at_max": 1.0,
        "eta_max": 1.0,
    }
    N = len(X)

    if target in ("all", "max_height"):
        mh = _resize_2d_fields(data["max_height"], mh_out_hw)
        out["ymh"]   = np.log1p(np.clip(mh, 0, 50).reshape(N, -1)).astype(np.float32)
        out["ymh2d"] = np.log1p(np.clip(mh, 0, 50)).astype(np.float32)[:, None]

    if target in ("all", "arrival_times"):
        at = _resize_2d_fields(data["arrival_times"], at_out_hw)
        at_max = float(at.max())
        out["at_max"] = at_max
        out["yat"]   = (at.reshape(N, -1) / (at_max + 1e-8)).astype(np.float32)
        out["yat2d"] = (at / (at_max + 1e-8)).astype(np.float32)[:, None]

    if target in ("all", "eta"):
        eta = _resize_3d_fields(data["eta"], eta_out_hw)
        eta_max = float(np.abs(eta).max())
        out["eta_max"] = eta_max
        out["yeta"]   = (eta.reshape(N, -1) / (eta_max + 1e-8)).astype(np.float32)
        out["yeta2d"] = (eta / (eta_max + 1e-8)).astype(np.float32)

    return out


def make_splits(*arrays, test: float = 0.10, val: float = 0.12, seed: int = RANDOM_SEED):
    """
    Split arrays into train / val / test.
    Returns a list of (arr_tr, arr_va, arr_te) tuples, one per input array.
    """
    idx = np.arange(len(arrays[0]))
    tv, te = train_test_split(idx, test_size=test, random_state=seed)
    tr, va = train_test_split(tv,  test_size=val,  random_state=seed)
    return [(a[tr], a[va], a[te]) for a in arrays]


def make_index_splits(n: int, test: float = 0.10, val: float = 0.12, seed: int = RANDOM_SEED):
    """Return train/val/test index arrays without materialising data arrays."""
    idx = np.arange(n)
    tv, te = train_test_split(idx, test_size=test, random_state=seed)
    tr, va = train_test_split(tv,  test_size=val,  random_state=seed)
    return tr, va, te


class H5FieldView:
    """Lightweight view over one HDF5 dataset with sample indices + transform."""

    def __init__(self, path: str, key: str, indices: np.ndarray, transform=None):
        self.path = path
        self.key = key
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform
        self.is_h5_view = True
        self._hf = None
        self._ds = None

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def _ensure_open(self):
        if self._hf is None:
            self._hf = h5py.File(self.path, "r")
            self._ds = self._hf[self.key]

    def _read_many(self, global_idx: np.ndarray):
        self._ensure_open()
        global_idx = np.asarray(global_idx, dtype=np.int64)
        order = np.argsort(global_idx)
        sorted_idx = global_idx[order]
        sorted_data = self._ds[sorted_idx]
        inv = np.empty_like(order)
        inv[order] = np.arange(order.size)
        return sorted_data[inv]

    def __getitem__(self, idx):
        self._ensure_open()
        gidx = self.indices[idx]
        if np.isscalar(gidx):
            out = self._ds[int(gidx)]
        else:
            out = self._read_many(gidx)
        if self.transform is not None:
            out = self.transform(out)
        return out

    def close(self):
        if self._hf is not None:
            self._hf.close()
            self._hf = None
            self._ds = None

    def __del__(self):
        self.close()

    # Windows DataLoader uses 'spawn' which pickles H5FieldView objects.
    # We must close the HDF5 handle before pickling and reopen lazily after.
    def __getstate__(self):
        self.close()
        state = self.__dict__.copy()
        state["_hf"] = None
        state["_ds"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)  # _hf / _ds remain None; reopened on demand


def _fit_feature_stats(path: str, chunk_size: int = 2048):
    """Compute mean/std for 'features' using a streaming pass over HDF5."""
    with h5py.File(path, "r") as hf:
        ds = hf["features"]
        n, nfeat = ds.shape
        sum_x = np.zeros(nfeat, dtype=np.float64)
        sum_x2 = np.zeros(nfeat, dtype=np.float64)
        for i in range(0, n, chunk_size):
            block = ds[i:i + chunk_size].astype(np.float64)
            sum_x += block.sum(axis=0)
            sum_x2 += (block * block).sum(axis=0)

    mean = sum_x / float(n)
    var = np.maximum(sum_x2 / float(n) - mean * mean, 1e-12)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def _transform_features_factory(mean: np.ndarray, std: np.ndarray):
    def _transform(arr):
        x = np.asarray(arr, dtype=np.float32)
        return ((x - mean) / (std + 1e-8)).astype(np.float32)
    return _transform


def _resize_2d_fields(arr, out_hw: tuple[int, int] | None, chunk_size: int = 64):
    """
    Resize 2-D fields to (H, W) while preserving sample axis when present.

    Accepted shapes:
      (H, W)      single sample
      (N, H, W)   batched samples
    """
    y = np.asarray(arr, dtype=np.float32)
    if out_hw is None:
        return np.ascontiguousarray(y, dtype=np.float32)

    out_h, out_w = out_hw
    if y.ndim == 2:
        if y.shape == (out_h, out_w):
            return np.ascontiguousarray(y, dtype=np.float32)
        t = torch.from_numpy(y).unsqueeze(0).unsqueeze(0)
        r = F.interpolate(t, size=out_hw, mode="bilinear", align_corners=False)
        return np.ascontiguousarray(r.squeeze(0).squeeze(0).cpu().numpy(), dtype=np.float32)

    if y.ndim == 3:
        if y.shape[-2:] == (out_h, out_w):
            return np.ascontiguousarray(y, dtype=np.float32)
        out = np.empty((y.shape[0], out_h, out_w), dtype=np.float32)
        for i in range(0, y.shape[0], chunk_size):
            blk = torch.from_numpy(y[i:i + chunk_size]).unsqueeze(1)
            rb = F.interpolate(blk, size=out_hw, mode="bilinear", align_corners=False)
            out[i:i + chunk_size] = rb.squeeze(1).cpu().numpy().astype(np.float32, copy=False)
        return np.ascontiguousarray(out, dtype=np.float32)

    raise ValueError(f"Expected 2D or 3D max_height array, got shape {y.shape}")


def _resize_3d_fields(arr, out_hw: tuple[int, int] | None, chunk_size: int = 16):
    """
    Resize 3-D fields over spatial dimensions (H, W), preserving channels.

    Accepted shapes:
      (C, H, W)      single sample with channels (e.g., NTIME)
      (N, C, H, W)   batched samples
    """
    y = np.asarray(arr, dtype=np.float32)
    if out_hw is None:
        return np.ascontiguousarray(y, dtype=np.float32)

    out_h, out_w = out_hw
    if y.ndim == 3:
        if y.shape[-2:] == (out_h, out_w):
            return np.ascontiguousarray(y, dtype=np.float32)
        t = torch.from_numpy(y).unsqueeze(0)  # (1, C, H, W)
        r = F.interpolate(t, size=out_hw, mode="bilinear", align_corners=False)
        return np.ascontiguousarray(r.squeeze(0).cpu().numpy(), dtype=np.float32)

    if y.ndim == 4:
        if y.shape[-2:] == (out_h, out_w):
            return np.ascontiguousarray(y, dtype=np.float32)
        out = np.empty((y.shape[0], y.shape[1], out_h, out_w), dtype=np.float32)
        for i in range(0, y.shape[0], chunk_size):
            blk = torch.from_numpy(y[i:i + chunk_size])
            rb = F.interpolate(blk, size=out_hw, mode="bilinear", align_corners=False)
            out[i:i + chunk_size] = rb.cpu().numpy().astype(np.float32, copy=False)
        return np.ascontiguousarray(out, dtype=np.float32)

    raise ValueError(f"Expected 3D or 4D eta array, got shape {y.shape}")


def _transform_mh_flat_factory(out_hw: tuple[int, int] | None = None):
    def _transform(arr):
        y = _resize_2d_fields(arr, out_hw)
        y = np.log1p(np.clip(y, 0, 50)).astype(np.float32)
        if y.ndim == 2:
            return np.ascontiguousarray(y.reshape(-1), dtype=np.float32)
        return np.ascontiguousarray(y.reshape(y.shape[0], -1), dtype=np.float32)
    return _transform


def _transform_mh_2d_factory(out_hw: tuple[int, int] | None = None):
    def _transform(arr):
        y = _resize_2d_fields(arr, out_hw)
        y = np.log1p(np.clip(y, 0, 50)).astype(np.float32)
        if y.ndim == 2:
            return np.ascontiguousarray(y[None, ...], dtype=np.float32)
        return np.ascontiguousarray(y[:, None, ...], dtype=np.float32)
    return _transform


def _transform_at_flat_factory(scale: float, out_hw: tuple[int, int] | None = None):
    def _transform(arr):
        y = _resize_2d_fields(arr, out_hw)
        if y.ndim == 2:
            return (y.reshape(-1) / (scale + 1e-8)).astype(np.float32)
        return (y.reshape(y.shape[0], -1) / (scale + 1e-8)).astype(np.float32)
    return _transform


def _transform_at_2d_factory(scale: float, out_hw: tuple[int, int] | None = None):
    def _transform(arr):
        y = _resize_2d_fields(arr, out_hw)
        y = (y / (scale + 1e-8)).astype(np.float32)
        if y.ndim == 2:
            return y[None, ...]
        return y[:, None, ...]
    return _transform


def _transform_eta_flat_factory(scale: float, out_hw: tuple[int, int] | None = None):
    def _transform(arr):
        y = _resize_3d_fields(arr, out_hw)
        if y.ndim == 3:
            return (y.reshape(-1) / (scale + 1e-8)).astype(np.float32)
        return (y.reshape(y.shape[0], -1) / (scale + 1e-8)).astype(np.float32)
    return _transform


def _transform_eta_2d_factory(scale: float, out_hw: tuple[int, int] | None = None):
    def _transform(arr):
        y = _resize_3d_fields(arr, out_hw)
        return (y / (scale + 1e-8)).astype(np.float32)
    return _transform


def _stream_dataset_max(path: str, key: str, abs_value: bool = False, chunk_size: int = 64) -> float:
    """Compute dataset max (or abs max) by streaming over sample dimension."""
    with h5py.File(path, "r") as hf:
        ds = hf[key]
        n = int(ds.shape[0])
        m = 0.0
        for i in range(0, n, chunk_size):
            block = ds[i:i + chunk_size]
            if abs_value:
                cur = float(np.abs(block).max())
            else:
                cur = float(block.max())
            if cur > m:
                m = cur
    return m


def build_query_points(lf_raw: dict, include_spt: bool = True):
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

    sp_scaler  = StandardScaler()
    spatial_scaled = sp_scaler.fit_transform(spatial_pts)

    q_sp  = torch.tensor(spatial_scaled, dtype=torch.float32).to(device)
    if include_spt:
        t_norm = (lf_raw["time"] / lf_raw["time"].max()).astype(np.float32)
        lon_t, lat_t, t_t = np.meshgrid(lf_raw["lon"], lf_raw["lat"], t_norm)
        spt_pts = np.stack(
            [lon_t.ravel(), lat_t.ravel(), t_t.ravel()], axis=1
        ).astype(np.float32)
        spt_scaler = StandardScaler()
        spt_scaled = spt_scaler.fit_transform(spt_pts)
        # q_spt stays on CPU: at 1 km grid with many timesteps the full
        # spatio-temporal tensor (NLAT*NLON*NTIME, 3) is too large for GPU.
        # Move slices to device inside the forward pass if needed.
        q_spt = torch.tensor(spt_scaled, dtype=torch.float32)  # CPU
    else:
        q_spt = torch.empty((0, 3), dtype=torch.float32)  # CPU; no-op for non-eta tasks
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


def _empty_split() -> dict:
    """Return an empty train/val/test split dict for unavailable targets."""
    e = np.empty((0,), dtype=np.float32)
    return {"tr": e, "va": e, "te": e}


def _canonical_target(target: str) -> str:
    """Map user target aliases to canonical keys used by preprocessing."""
    t = target.strip().lower()
    aliases = {
        "max_height": "max_height",
        "arrival_times": "arrival_times",
        "eta": "eta",
        "eta_timeseries": "eta",
        "all": "all",
    }
    if t not in aliases:
        raise ValueError(
            f"Unknown target '{target}'. Use one of: {', '.join(sorted(aliases))}."
        )
    return aliases[t]


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


def load_dataset(target: str = "all", lazy: bool = True) -> DataBundle:
    """
    Load three-fidelity datasets, preprocess, split, and return a DataBundle.
    This is the only function in data.py that reads from disk.

    Parameters
    ----------
    target : str
        "all" | "max_height" | "arrival_times" | "eta" | "eta_timeseries"
        When a specific target is provided, only that target is materialised,
        which significantly lowers peak RAM usage.
    lazy : bool
        If True, return HDF5-backed split views so training reads samples on demand.
        This avoids loading full datasets into RAM at startup.
    """
    target = _canonical_target(target)
    if lazy:
        with h5py.File(DATA_PATHS["lf"], "r") as hf_lf, \
             h5py.File(DATA_PATHS["mf"], "r") as hf_mf, \
             h5py.File(DATA_PATHS["hf"], "r") as hf_hf:
            n_lf = int(hf_lf["features"].shape[0])
            n_mf = int(hf_mf["features"].shape[0])
            n_hf = int(hf_hf["features"].shape[0])
            if target == "eta":
                _, NTIME, NLAT, NLON = hf_lf["eta"].shape
            elif target == "arrival_times":
                _, NLAT, NLON = hf_lf["arrival_times"].shape
                NTIME = int(hf_lf["time"].shape[0])
            else:
                _, NLAT, NLON = hf_lf["max_height"].shape
                NTIME = int(hf_lf["time"].shape[0])
            if target == "max_height":
                mf_nlat, mf_nlon = hf_mf["max_height"].shape[-2:]
                hf_nlat, hf_nlon = hf_hf["max_height"].shape[-2:]
            elif target == "arrival_times":
                mf_nlat, mf_nlon = hf_mf["arrival_times"].shape[-2:]
                hf_nlat, hf_nlon = hf_hf["arrival_times"].shape[-2:]
            elif target == "eta":
                mf_nlat, mf_nlon = hf_mf["eta"].shape[-2:]
                hf_nlat, hf_nlon = hf_hf["eta"].shape[-2:]
            else:
                mf_nlat = hf_nlat = NLAT
                mf_nlon = hf_nlon = NLON
            NTIME = int(hf_lf["time"].shape[0])
            lon = hf_lf["lon"][:]
            lat = hf_lf["lat"][:]
            bathy_raw = hf_lf["bathymetry"][:].astype(np.float32)
            time = hf_lf["time"][:]

        lf_tr, lf_va, lf_te = make_index_splits(n_lf)
        mf_tr, mf_va, mf_te = make_index_splits(n_mf)
        hf_tr, hf_va, hf_te = make_index_splits(n_hf)

        mean, std = _fit_feature_stats(DATA_PATHS["lf"])
        feat_tf = _transform_features_factory(mean, std)

        def _pack_views(path: str, tr: np.ndarray, va: np.ndarray, te: np.ndarray, key: str, tf):
            return {
                "tr": H5FieldView(path, key, tr, tf),
                "va": H5FieldView(path, key, va, tf),
                "te": H5FieldView(path, key, te, tf),
            }

        q_sp, q_spt = build_query_points(
            {"lon": lon, "lat": lat, "time": time},
            include_spt=(target == "eta"),
        )
        bathy_norm = (bathy_raw - bathy_raw.mean()) / (bathy_raw.std() + 1e-8)
        bathy_t = torch.tensor(bathy_norm, dtype=torch.float32).to(device)

        at_lf = at_mf = at_hf = 1.0
        eta_lf = eta_mf = eta_hf = 1.0

        if target == "arrival_times":
            at_lf = _stream_dataset_max(DATA_PATHS["lf"], "arrival_times")
            at_mf = _stream_dataset_max(DATA_PATHS["mf"], "arrival_times")
            at_hf = _stream_dataset_max(DATA_PATHS["hf"], "arrival_times")
        elif target == "eta":
            eta_lf = _stream_dataset_max(DATA_PATHS["lf"], "eta", abs_value=True)
            eta_mf = _stream_dataset_max(DATA_PATHS["mf"], "eta", abs_value=True)
            eta_hf = _stream_dataset_max(DATA_PATHS["hf"], "eta", abs_value=True)

        print("Loading datasets…")
        print(f"LF:{n_lf}  MF:{n_mf}  HF:{n_hf}")
        print(f"Grid: ({NLAT}×{NLON})  Time steps: {NTIME}")
        if target in ("max_height", "arrival_times", "eta"):
            label_map = {
                "max_height": "max_height",
                "arrival_times": "arrival_times",
                "eta": "eta",
            }
            label = label_map[target]
            print(f"Native {label} grids: "
              f"LF({NLAT}×{NLON})  MF({mf_nlat}×{mf_nlon})  HF({hf_nlat}×{hf_nlon})")
            print(f"Canonical {label} grid: LF ({NLAT}×{NLON})")
        print(f"LF  train:{len(lf_tr)}  val:{len(lf_va)}  test:{len(lf_te)}")
        print(f"MF  train:{len(mf_tr)}  val:{len(mf_va)}  test:{len(mf_te)}")
        print(f"HF  train:{len(hf_tr)}  val:{len(hf_va)}  test:{len(hf_te)}")

        e = _empty_split
        scaler_stub = {"mean": mean, "std": std}

        if target == "max_height":
            canonical_hw = (NLAT, NLON)
            mh_tf_flat = _transform_mh_flat_factory(canonical_hw)
            mh_tf_2d = _transform_mh_2d_factory(canonical_hw)

            ymh_lf = _pack_views(DATA_PATHS["lf"], lf_tr, lf_va, lf_te, "max_height", mh_tf_flat)
            ymh_mf = _pack_views(DATA_PATHS["mf"], mf_tr, mf_va, mf_te, "max_height", mh_tf_flat)
            ymh_hf = _pack_views(DATA_PATHS["hf"], hf_tr, hf_va, hf_te, "max_height", mh_tf_flat)
            ymh2d_lf = _pack_views(DATA_PATHS["lf"], lf_tr, lf_va, lf_te, "max_height", mh_tf_2d)
            ymh2d_mf = _pack_views(DATA_PATHS["mf"], mf_tr, mf_va, mf_te, "max_height", mh_tf_2d)
            ymh2d_hf = _pack_views(DATA_PATHS["hf"], hf_tr, hf_va, hf_te, "max_height", mh_tf_2d)
            yat_lf = yat_mf = yat_hf = e()
            yat2d_lf = yat2d_mf = yat2d_hf = e()
            yeta_lf = yeta_mf = yeta_hf = e()
            yeta2d_lf = yeta2d_mf = yeta2d_hf = e()
        elif target == "arrival_times":
            canonical_hw = (NLAT, NLON)
            at_tf_lf_f = _transform_at_flat_factory(at_lf, canonical_hw)
            at_tf_mf_f = _transform_at_flat_factory(at_mf, canonical_hw)
            at_tf_hf_f = _transform_at_flat_factory(at_hf, canonical_hw)
            at_tf_lf_2 = _transform_at_2d_factory(at_lf, canonical_hw)
            at_tf_mf_2 = _transform_at_2d_factory(at_mf, canonical_hw)
            at_tf_hf_2 = _transform_at_2d_factory(at_hf, canonical_hw)
            yat_lf = _pack_views(DATA_PATHS["lf"], lf_tr, lf_va, lf_te, "arrival_times", at_tf_lf_f)
            yat_mf = _pack_views(DATA_PATHS["mf"], mf_tr, mf_va, mf_te, "arrival_times", at_tf_mf_f)
            yat_hf = _pack_views(DATA_PATHS["hf"], hf_tr, hf_va, hf_te, "arrival_times", at_tf_hf_f)
            yat2d_lf = _pack_views(DATA_PATHS["lf"], lf_tr, lf_va, lf_te, "arrival_times", at_tf_lf_2)
            yat2d_mf = _pack_views(DATA_PATHS["mf"], mf_tr, mf_va, mf_te, "arrival_times", at_tf_mf_2)
            yat2d_hf = _pack_views(DATA_PATHS["hf"], hf_tr, hf_va, hf_te, "arrival_times", at_tf_hf_2)
            ymh_lf = ymh_mf = ymh_hf = e()
            ymh2d_lf = ymh2d_mf = ymh2d_hf = e()
            yeta_lf = yeta_mf = yeta_hf = e()
            yeta2d_lf = yeta2d_mf = yeta2d_hf = e()
        else:
            canonical_hw = (NLAT, NLON)
            eta_tf_lf_f = _transform_eta_flat_factory(eta_lf, canonical_hw)
            eta_tf_mf_f = _transform_eta_flat_factory(eta_mf, canonical_hw)
            eta_tf_hf_f = _transform_eta_flat_factory(eta_hf, canonical_hw)
            eta_tf_lf_2 = _transform_eta_2d_factory(eta_lf, canonical_hw)
            eta_tf_mf_2 = _transform_eta_2d_factory(eta_mf, canonical_hw)
            eta_tf_hf_2 = _transform_eta_2d_factory(eta_hf, canonical_hw)
            yeta_lf = _pack_views(DATA_PATHS["lf"], lf_tr, lf_va, lf_te, "eta", eta_tf_lf_f)
            yeta_mf = _pack_views(DATA_PATHS["mf"], mf_tr, mf_va, mf_te, "eta", eta_tf_mf_f)
            yeta_hf = _pack_views(DATA_PATHS["hf"], hf_tr, hf_va, hf_te, "eta", eta_tf_hf_f)
            yeta2d_lf = _pack_views(DATA_PATHS["lf"], lf_tr, lf_va, lf_te, "eta", eta_tf_lf_2)
            yeta2d_mf = _pack_views(DATA_PATHS["mf"], mf_tr, mf_va, mf_te, "eta", eta_tf_mf_2)
            yeta2d_hf = _pack_views(DATA_PATHS["hf"], hf_tr, hf_va, hf_te, "eta", eta_tf_hf_2)
            ymh_lf = ymh_mf = ymh_hf = e()
            ymh2d_lf = ymh2d_mf = ymh2d_hf = e()
            yat_lf = yat_mf = yat_hf = e()
            yat2d_lf = yat2d_mf = yat2d_hf = e()

        return DataBundle(
            NLAT=NLAT, NLON=NLON, NTIME=NTIME,
            X_lf=_pack_views(DATA_PATHS["lf"], lf_tr, lf_va, lf_te, "features", feat_tf),
            X_mf=_pack_views(DATA_PATHS["mf"], mf_tr, mf_va, mf_te, "features", feat_tf),
            X_hf=_pack_views(DATA_PATHS["hf"], hf_tr, hf_va, hf_te, "features", feat_tf),

            ymh_lf=ymh_lf, ymh_mf=ymh_mf, ymh_hf=ymh_hf,

            yat_lf=yat_lf, yat_mf=yat_mf, yat_hf=yat_hf,
            yeta_lf=yeta_lf, yeta_mf=yeta_mf, yeta_hf=yeta_hf,

            ymh2d_lf=ymh2d_lf, ymh2d_mf=ymh2d_mf, ymh2d_hf=ymh2d_hf,

            yat2d_lf=yat2d_lf, yat2d_mf=yat2d_mf, yat2d_hf=yat2d_hf,
            yeta2d_lf=yeta2d_lf, yeta2d_mf=yeta2d_mf, yeta2d_hf=yeta2d_hf,

            q_sp=q_sp, q_spt=q_spt, bathy_t=bathy_t,

            at_lf=at_lf, at_mf=at_mf, at_hf=at_hf,
            eta_lf=eta_lf, eta_mf=eta_mf, eta_hf=eta_hf,

            lon=lon, lat=lat,
            feat_scaler=scaler_stub,
        )

    target_keys = {
        "all": ["max_height", "arrival_times", "eta"],
        "max_height": ["max_height"],
        "arrival_times": ["arrival_times"],
        "eta": ["eta"],
    }[target]

    def _split_processed(proc: dict) -> dict:
        names = ["X"]
        arrays = [proc["X"]]
        for name in ["ymh", "yat", "yeta", "ymh2d", "yat2d", "yeta2d"]:
            if proc[name] is not None:
                names.append(name)
                arrays.append(proc[name])
        splits = make_splits(*arrays)
        return {n: _to_dict(splits[i]) for i, n in enumerate(names)}

    print("Loading datasets…")
    # LF: includes spatial/time metadata used to build auxiliary tensors.
    lf_keys = ["features", "lon", "lat", "time", "bathymetry", *target_keys]
    lf_raw = load_h5(DATA_PATHS["lf"], keys=lf_keys)

    if "max_height" in lf_raw:
        _, NLAT, NLON = lf_raw["max_height"].shape
    elif "arrival_times" in lf_raw:
        _, NLAT, NLON = lf_raw["arrival_times"].shape
    else:
        _, _, NLAT, NLON = lf_raw["eta"].shape
    NTIME = int(len(lf_raw["time"]))
    mh_out_hw = (NLAT, NLON) if target in ("all", "max_height") else None
    at_out_hw = (NLAT, NLON) if target in ("all", "arrival_times") else None
    eta_out_hw = (NLAT, NLON) if target in ("all", "eta") else None

    lf_proc   = preprocess(
        lf_raw, fit=True, target=target,
        mh_out_hw=mh_out_hw, at_out_hw=at_out_hw, eta_out_hw=eta_out_hw
    )
    s_lf_map  = _split_processed(lf_proc)
    feat_scaler = lf_proc["feat_scaler"]
    at_lf, eta_lf = float(lf_proc["at_max"]), float(lf_proc["eta_max"])

    q_sp, q_spt = build_query_points(lf_raw, include_spt=(target == "eta"))
    bathy_t = build_bathy(lf_raw)
    lon = lf_raw["lon"]
    lat = lf_raw["lat"]
    n_lf = lf_proc["X"].shape[0]

    del lf_raw, lf_proc
    gc.collect()

    # MF
    mf_raw = load_h5(DATA_PATHS["mf"], keys=["features", *target_keys])
    mf_hw = tuple(mf_raw["max_height"].shape[-2:]) if "max_height" in mf_raw else None
    mf_eta_hw = tuple(mf_raw["eta"].shape[-2:]) if "eta" in mf_raw else None
    mf_proc  = preprocess(
        mf_raw, feat_scaler=feat_scaler, target=target,
        mh_out_hw=mh_out_hw, at_out_hw=at_out_hw, eta_out_hw=eta_out_hw
    )
    s_mf_map = _split_processed(mf_proc)
    at_mf, eta_mf = float(mf_proc["at_max"]), float(mf_proc["eta_max"])
    n_mf = mf_proc["X"].shape[0]

    del mf_raw, mf_proc
    gc.collect()

    # HF
    hf_raw = load_h5(DATA_PATHS["hf"], keys=["features", *target_keys])
    hf_hw = tuple(hf_raw["max_height"].shape[-2:]) if "max_height" in hf_raw else None
    hf_eta_hw = tuple(hf_raw["eta"].shape[-2:]) if "eta" in hf_raw else None
    hf_proc  = preprocess(
        hf_raw, feat_scaler=feat_scaler, target=target,
        mh_out_hw=mh_out_hw, at_out_hw=at_out_hw, eta_out_hw=eta_out_hw
    )
    s_hf_map = _split_processed(hf_proc)
    at_hf, eta_hf = float(hf_proc["at_max"]), float(hf_proc["eta_max"])
    n_hf = hf_proc["X"].shape[0]

    del hf_raw, hf_proc
    gc.collect()

    print(f"LF:{n_lf}  MF:{n_mf}  HF:{n_hf}")
    print(f"Grid: ({NLAT}×{NLON})  Time steps: {NTIME}")
    if target in ("all", "max_height") and mf_hw is not None and hf_hw is not None:
        print("Native max_height grids: "
              f"LF({NLAT}×{NLON})  MF({mf_hw[0]}×{mf_hw[1]})  HF({hf_hw[0]}×{hf_hw[1]})")
        print(f"Canonical max_height grid: LF ({NLAT}×{NLON})")
    if target in ("all", "eta") and mf_eta_hw is not None and hf_eta_hw is not None:
        print("Native eta grids: "
              f"LF({NLAT}×{NLON})  MF({mf_eta_hw[0]}×{mf_eta_hw[1]})  HF({hf_eta_hw[0]}×{hf_eta_hw[1]})")
        print(f"Canonical eta grid: LF ({NLAT}×{NLON})")
    print(f"LF  train:{len(s_lf_map['X']['tr'])}  val:{len(s_lf_map['X']['va'])}  test:{len(s_lf_map['X']['te'])}")
    print(f"MF  train:{len(s_mf_map['X']['tr'])}  val:{len(s_mf_map['X']['va'])}  test:{len(s_mf_map['X']['te'])}")
    print(f"HF  train:{len(s_hf_map['X']['tr'])}  val:{len(s_hf_map['X']['va'])}  test:{len(s_hf_map['X']['te'])}")

    e = _empty_split
    return DataBundle(
        NLAT=NLAT, NLON=NLON, NTIME=NTIME,

        X_lf=s_lf_map["X"],  X_mf=s_mf_map["X"],  X_hf=s_hf_map["X"],

        ymh_lf=s_lf_map.get("ymh", e()),   ymh_mf=s_mf_map.get("ymh", e()),   ymh_hf=s_hf_map.get("ymh", e()),
        yat_lf=s_lf_map.get("yat", e()),   yat_mf=s_mf_map.get("yat", e()),   yat_hf=s_hf_map.get("yat", e()),
        yeta_lf=s_lf_map.get("yeta", e()), yeta_mf=s_mf_map.get("yeta", e()), yeta_hf=s_hf_map.get("yeta", e()),

        ymh2d_lf=s_lf_map.get("ymh2d", e()),  ymh2d_mf=s_mf_map.get("ymh2d", e()),  ymh2d_hf=s_hf_map.get("ymh2d", e()),
        yat2d_lf=s_lf_map.get("yat2d", e()),  yat2d_mf=s_mf_map.get("yat2d", e()),  yat2d_hf=s_hf_map.get("yat2d", e()),
        yeta2d_lf=s_lf_map.get("yeta2d", e()), yeta2d_mf=s_mf_map.get("yeta2d", e()), yeta2d_hf=s_hf_map.get("yeta2d", e()),

        q_sp=q_sp, q_spt=q_spt, bathy_t=bathy_t,

        at_lf=at_lf,   at_mf=at_mf,   at_hf=at_hf,
        eta_lf=eta_lf, eta_mf=eta_mf, eta_hf=eta_hf,

        lon=lon, lat=lat,
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

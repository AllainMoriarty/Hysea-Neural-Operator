# SWE Physics Loss for MFPino — Complete Sketch

> **Status**: Implemented.  
> **Dataset**: Synthetic output from Tsunami-HySEA v1.3.0 — 2D nonlinear SWE, spherical coordinates, GPU FVM.  
> **Approach**: Hybrid physics loss — two functions, one `if-else` dispatch inside `MFPino`.

---

## 1. Physical Variables

| Symbol | Meaning | Units | In model |
|--------|---------|-------|----------|
| `η(λ,φ,t)` | Sea-surface elevation | m | PINO output (eta task) |
| `H(λ,φ)` | Still-water depth (bathymetry) | m | `H_raw` buffer |
| `h = η + H` | Total water depth | m | — |
| `qx = h·u` | Longitudinal discharge | m²/s | Not predicted |
| `qy = h·v` | Latitudinal discharge | m²/s | Not predicted |
| `λ, φ` | Longitude, latitude | rad | `dlon`, `dlat`, `lat_rad` |
| `R` | Earth radius | 6,378,136.6 m | `EARTH_RADIUS` |
| `g` | Gravity | 9.81 m/s² | `G` |

---

## 2. Tsunami-HySEA SWE (Spherical Coordinates)

HySEA integrates the **nonlinear 2D SWE** ([src/GPU/Arista_kernel.cu](https://github.com/edanya-uma/Tsunami-HySEA/blob/main/src/GPU/Arista_kernel.cu)):

### Continuity
```
∂h/∂t  +  1/(R·cosφ)·∂qx/∂λ  +  1/R·∂(qy·cosφ)/∂φ  =  0
```

### X-Momentum (longitude)
```
∂qx/∂t  +  1/(R·cosφ)·∂(qx²/h + ½g·h²)/∂λ  +  1/R·∂(qx·qy·cosφ/h)/∂φ
         =  (qx·qy·tanφ)/(R·h)  −  g·h/(R·cosφ)·∂H/∂λ  −  τx
```

### Y-Momentum (latitude)
```
∂qy/∂t  +  1/(R·cosφ)·∂(qx·qy/h)/∂λ  +  1/R·∂(qy²·cosφ/h + ½g·h²·cosφ)/∂φ
         =  −(qx²·tanφ)/(R·h)  −  g·h/R·∂H/∂φ  −  τy
```

### Manning Friction
```
τx = g·n²·qx·√(qx²+qy²) / h^(7/3)
τy = g·n²·qy·√(qx²+qy²) / h^(7/3)
```

> [!NOTE]
> The PINO predicts only `η` (or `T`, or `η_max`) — not `qx`, `qy`. So we cannot evaluate
> the full system directly. Each physics loss below is derived from a reduced form that
> involves only the predicted quantity.

---

## 3. Hybrid Physics Loss Design

```
task == "arrival_times" ──→  eikonal_loss()        |∇T|² = 1/(gH)    [EXACT]
task == "max_height"    ──→  swe_spatial_loss()    ∇·(gH·∇u) = 0    [linearised SWE]
task == "eta"           ──→  swe_spatial_loss()    ∇·(gH·∇u) = 0    [linearised SWE]
```

| | `eikonal_loss` | `swe_spatial_loss` |
|-|----------------|-------------------|
| Derivation | Long-wave kinematic wavefront | Linearised SWE, qx/qy eliminated |
| Validity | Exact across entire ocean | Deep ocean only (`η << H`) |
| Masking | Ocean + sponge + source-region | Ocean + sponge |
| Output shape | `(B, 1, NLAT, NLON)` | `(B, C, NLAT, NLON)` any C |

---

## 4. `swe_spatial_loss` — max_height & eta

### Derivation

Linearise HySEA's SWE around `η≈0`, `qx≈0`, `qy≈0`. Drop nonlinear advection and Manning friction (O(η²)):

**Linearised continuity:**
```
∂η/∂t = −1/(R·cosφ)·∂qx/∂λ − 1/R·∂(qy·cosφ)/∂φ
```

**Linearised momentum:**
```
∂qx/∂t = −gH/(R·cosφ)·∂η/∂λ
∂qy/∂t = −gH/R·∂η/∂φ
```

Differentiate continuity by `t`, substitute linearised momentum → eliminate `qx`, `qy`:

```
∂²η/∂t²  =  ∇·(gH·∇η)                       [variable-coefficient wave equation]
```

In spherical coordinates:

```
∇·(gH·∇η) =  g/(R²·cos²φ) · ∂(H·∂η/∂λ)/∂λ
           +  g/(R²·cosφ)  · ∂(H·cosφ·∂η/∂φ)/∂φ
```

> [!IMPORTANT]
> This is **not** the same as `gH·∇²η`. The correct form `∇·(gH·∇η)` includes the extra
> term `g·(∇H)·(∇η)` from the product rule, which represents wave shoaling — the dominant
> mechanism driving amplitude growth as tsunamis approach shore.

As a spatial regulariser (ignoring `∂²η/∂t²` or applying per time channel):

```
R(u) = || ∇·(gH·∇u) ||²  ≈  0
```

### Validity
- ✅ Deep ocean, slow bathymetry variation, `η << H`
- ❌ Near-shore cells → masked out (`H < 10 m`)
- ❌ Sponge boundary (4 cells) → masked out

### Code (`models/swe_residuals.py`)

```python
def swe_spatial_loss(
    u: torch.Tensor,           # (B, C, NLAT, NLON)
    H_raw: torch.Tensor,       # (NLAT, NLON)  — raw bathymetry [m]
    lat_rad: torch.Tensor,     # (NLAT,)       — latitudes [rad]
    dlon: float,               # Δλ [rad]
    dlat: float,               # Δφ [rad]
    R: float = EARTH_RADIUS,   # 6_378_136.6 m
    g: float = G,              # 9.81 m/s²
    min_depth: float = 10.0,
    sponge: int = 4,           # HySEA SPONGE_SIZE = 4
) -> torch.Tensor:
    B, C, NLAT, NLON = u.shape
    H  = H_raw.view(1,1,NLAT,NLON).expand(B,C,-1,-1)
    Hs = torch.clamp(H, min=min_depth)
    cos = _cos_phi_grid(lat_rad, B, C, NLAT, NLON, u.device)

    # ∂u/∂λ,  ∂u/∂φ
    du_dlon = _cd(u, dim=3, dx=dlon)
    du_dlat = _cd(u, dim=2, dx=dlat)

    # Weighted flux: F_λ = H·∂u/∂λ,  F_φ = H·cosφ·∂u/∂φ
    F_lon = Hs * du_dlon
    F_lat = Hs * cos * du_dlat

    # Divergence
    dFlon_dlon = _cd(F_lon, dim=3, dx=dlon)
    dFlat_dlat = _cd(F_lat, dim=2, dx=dlat)

    residual = (g * dFlon_dlon / (R**2 * cos**2)
              + g * dFlat_dlat / (R**2 * cos))

    mask = _masks(H, NLAT, NLON, min_depth, sponge)
    return (mask * residual**2).mean()
```

---

## 5. `eikonal_loss` — arrival_times

### Derivation

The first-arrival time `T(λ,φ) = min{t : η(λ,φ,t) > ε_h}` is recorded by HySEA when
the surface exceeds threshold `dif_at` ([ShallowWater.cxx L111](https://github.com/edanya-uma/Tsunami-HySEA/blob/main/src/ShallowWater.cxx)).

The tsunami wavefront propagates at the long-wave phase speed `c = √(gH)`.
The first-arrival time satisfies the **eikonal equation** exactly:

```
|∇T|²  =  1/c²  =  1/(gH)
```

In spherical coordinates:

```
(∂T/∂λ)²/(R·cosφ)²  +  (∂T/∂φ)²/R²  =  1/(g·H)
```

> [!NOTE]
> This is the **only exact constraint** among the three. The long-wave approximation
> (`λ_wave >> H`) holds for tsunamis across the entire ocean basin — typical tsunami
> wavelengths are 100–500 km vs ocean depths of 1–6 km.

### Masking
- Ocean + sponge boundary (same as above)
- **Source region**: cells where `T < 1%·T_max` are masked — the eikonal breaks at the wavefront origin where `|∇T| → ∞`

### Code (`models/swe_residuals.py`)

```python
def eikonal_loss(
    T_pred: torch.Tensor,      # (B, 1, NLAT, NLON) — normalised ∈ [0,1]
    H_raw: torch.Tensor,       # (NLAT, NLON) [m]
    lat_rad: torch.Tensor,     # (NLAT,) [rad]
    dlon: float,
    dlat: float,
    T_max: float,              # T_physical = T_pred * T_max  [s]
    R: float = EARTH_RADIUS,
    g: float = G,
    min_depth: float = 10.0,
    sponge: int = 4,
    min_T_frac: float = 0.01,  # mask source region
) -> torch.Tensor:
    B, _, NLAT, NLON = T_pred.shape
    T_phys = T_pred * T_max    # denormalise [s]

    H  = H_raw.view(1,1,NLAT,NLON).expand(B,1,-1,-1)
    Hs = torch.clamp(H, min=min_depth)
    cos = _cos_phi_grid(lat_rad, B, 1, NLAT, NLON, T_pred.device)

    # |∇T|² in spherical coords [s²/m²]
    dT_dlon = _cd(T_phys, dim=3, dx=dlon)
    dT_dlat = _cd(T_phys, dim=2, dx=dlat)
    grad_T_sq = (dT_dlon / (R * cos))**2 + (dT_dlat / R)**2

    # Slowness 1/c² = 1/(gH) [s²/m²]
    slowness_sq = 1.0 / (g * Hs)

    residual = grad_T_sq - slowness_sq

    # Mask: ocean + sponge + arrived cells only
    mask = _masks(H, NLAT, NLON, min_depth, sponge) * (T_pred > min_T_frac).float()
    return (mask * residual**2).mean()
```

---

## 6. Shared Utilities (`models/swe_residuals.py`)

```python
EARTH_RADIUS = 6_378_136.6   # from HySEA Constantes.hxx
G            = 9.81
_EPS         = 1e-7

def _cd(u, dim, dx):
    """2nd-order central difference, circular roll padding."""
    return (torch.roll(u,-1,dims=dim) - torch.roll(u,1,dims=dim)) / (2*dx)

def _cos_phi_grid(lat_rad, B, C, NLAT, NLON, device):
    """cosφ broadcast to (B,C,NLAT,NLON), clamped ≥ EPS."""
    cos = torch.cos(lat_rad.to(device)).view(1,1,NLAT,1).expand(B,C,-1,NLON)
    return torch.clamp(cos, min=_EPS)

def _masks(H, NLAT, NLON, min_depth, sponge):
    """Ocean mask * sponge boundary mask → 1 where physics enforced."""
    ocean = (H > min_depth).float()
    boundary = torch.zeros_like(ocean)
    if NLAT > 2*sponge and NLON > 2*sponge:
        boundary[:,:,sponge:NLAT-sponge,sponge:NLON-sponge] = 1.0
    return ocean * boundary
```

---

## 7. `MFPino` Dispatch (`models/pino.py`)

```python
def pino_physics_loss(self, fault_params, bathy_t, fidelity="lf"):
    pred = self.forward(fault_params, bathy_t, fidelity=fidelity)

    gi = self.grid_info
    if self.H_raw is None or not gi:
        return spectral_laplacian_residual(pred)   # graceful fallback

    lat_rad = gi["lat_rad"]
    dlon, dlat = gi["dlon"], gi["dlat"]

    if self.task == "arrival_times":
        return eikonal_loss(pred, self.H_raw, lat_rad, dlon, dlat,
                            T_max=gi.get("T_max", 1.0))

    return swe_spatial_loss(pred, self.H_raw, lat_rad, dlon, dlat)
```

`H_raw` is stored via `self.register_buffer("H_raw", H_raw)` — automatically moves to GPU with `.to(device)` and is saved in checkpoints.

---

## 8. Grid Info Builder (`data.py`)

```python
def build_grid_info(lf_raw: dict, db: DataBundle) -> tuple:
    lon_rad = np.deg2rad(lf_raw["lon"].astype(np.float32))
    lat_rad = np.deg2rad(lf_raw["lat"].astype(np.float32))

    grid_info = {
        "lat_rad": torch.tensor(lat_rad, dtype=torch.float32).to(device),
        "dlon":    float(np.mean(np.diff(lon_rad))),
        "dlat":    float(np.abs(np.mean(np.diff(lat_rad)))),
        "T_max":   db.at_lf,    # arrival_times denorm factor [s]
    }
    H_raw = torch.tensor(
        lf_raw["bathymetry"].astype(np.float32), dtype=torch.float32
    ).to(device)

    return grid_info, H_raw
```

---

## 9. Task Script Pattern (identical for all three)

```python
# All three task scripts: max_height.py, arrival_times.py, eta_timeseries.py
from data import load_dataset, load_h5, build_grid_info

db               = load_dataset()
lf_raw           = load_h5(DATA_PATHS["lf"])
grid_info, H_raw = build_grid_info(lf_raw, db)

pino = MFPino(
    nlat=nlat, nlon=nlon,
    latent_channels=..., num_fno_layers=..., num_fno_modes=...,
    out_channels=1,           # or ntime for eta
    task="max_height",        # "max_height" | "arrival_times" | "eta"
    H_raw=H_raw,
    grid_info=grid_info,
).to(device)
```

`training.py` / `train_mf_pino()` — **zero changes** required.

---

## 10. Total Loss

```
L_total = MSE(pred, HySEA_label) + λ_pde · physics_loss(pred)
```

`λ_pde` is tuned by Optuna (`suggest_float("lambda_pde", 1e-4, 1e-1, log=True)`).

Suggested starting values for first trial:

| Task | `λ_pde` | Reason |
|------|---------|--------|
| `max_height` | `5e-3` | Energy-weighted spatial residual, smooth |
| `arrival_times` | `1e-2` | Eikonal well-conditioned in deep ocean |
| `eta` | `1e-3` | Applied to all NTIME channels simultaneously |

---

## 11. File Summary

| File | Role |
|------|------|
| [`models/swe_residuals.py`](file:///f:/Researchs/neural-operator/models/swe_residuals.py) | `swe_spatial_loss` + `eikonal_loss` + shared helpers |
| [`models/pino.py`](file:///f:/Researchs/neural-operator/models/pino.py) | `MFPino` with hybrid dispatch + fallback |
| [`models/__init__.py`](file:///f:/Researchs/neural-operator/models/__init__.py) | Exports `swe_spatial_loss`, `eikonal_loss` |
| [`data.py`](file:///f:/Researchs/neural-operator/data.py) | `build_grid_info(lf_raw, db)` |
| [`max_height.py`](file:///f:/Researchs/neural-operator/max_height.py) | `task="max_height"` → `swe_spatial_loss` |
| [`arrival_times.py`](file:///f:/Researchs/neural-operator/arrival_times.py) | `task="arrival_times"` → `eikonal_loss` |
| [`eta_timeseries.py`](file:///f:/Researchs/neural-operator/eta_timeseries.py) | `task="eta"` → `swe_spatial_loss` per channel |
| `training.py` | **Unchanged** |

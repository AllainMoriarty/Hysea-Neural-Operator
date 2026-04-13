"""
SWE Physics Residuals for MFPino.

Two functions cover all three prediction targets (hybrid design):

  swe_spatial_loss(u, ...)
      ∇·(gH·∇u) = 0  — unified for max_height and eta timeseries
      Derived from linearised SWE (continuity + momentum combined).
      Applies to any (B, C, NLAT, NLON) field; for eta each time
      channel is treated as an independent spatial field.

  eikonal_loss(T_pred, ...)
      |∇T|² = 1/(gH)  — exact for arrival_times
      Long-wave kinematic wavefront equation.

Both use:
  - spherical-coordinate metric (Earth radius from HySEA Constantes.hxx)
  - ocean + sponge-boundary masking (SPONGE_SIZE = 4 from HySEA)
  - depth-floor guard (min_depth=10 m) for land / dry cells

No top-level execution — functions only.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F   # noqa: F401 (kept for optional future use)

# Physical constants matching HySEA src/Constantes.hxx
EARTH_RADIUS: float = 6_378_136.6   # metres
G: float = 9.81                      # m s⁻²
_EPS: float = 1e-7                   # numerical guard


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _cd(u: torch.Tensor, dim: int, dx: float) -> torch.Tensor:
    """2nd-order central difference with circular (roll) padding."""
    return (torch.roll(u, -1, dims=dim) - torch.roll(u, 1, dims=dim)) / (2.0 * dx)


def _cos_phi_grid(lat_rad: torch.Tensor, B: int, C: int, NLAT: int, NLON: int,
                  device: torch.device) -> torch.Tensor:
    """Broadcast cosφ to (B, C, NLAT, NLON), clamped to avoid pole singularity."""
    cos = torch.cos(lat_rad).to(device)                  # (NLAT,)
    cos = cos.view(1, 1, NLAT, 1).expand(B, C, -1, NLON)
    return torch.clamp(cos, min=_EPS)


def _masks(H_expanded: torch.Tensor, NLAT: int, NLON: int,
           min_depth: float, sponge: int) -> torch.Tensor:
    """
    Combined ocean + sponge mask: 1 where physics is enforced, 0 elsewhere.
    H_expanded shape: (B, C, NLAT, NLON)
    """
    ocean = (H_expanded > min_depth).float()
    boundary = torch.zeros_like(ocean)
    if NLAT > 2 * sponge and NLON > 2 * sponge:
        boundary[:, :, sponge:NLAT - sponge, sponge:NLON - sponge] = 1.0
    return ocean * boundary


# ──────────────────────────────────────────────────────────────────────────────
# 1. Unified spatial loss  —  max_height  &  eta
# ──────────────────────────────────────────────────────────────────────────────

def swe_spatial_loss(
    u: torch.Tensor,            # (B, C, NLAT, NLON) — any predicted spatial field
    H_raw: torch.Tensor,        # (NLAT, NLON) — raw bathymetry [m], positive=ocean
    lat_rad: torch.Tensor,      # (NLAT,) — latitudes in radians
    dlon: float,                # Δλ [rad]
    dlat: float,                # Δφ [rad]
    R: float = EARTH_RADIUS,
    g: float = G,
    min_depth: float = 10.0,    # mask cells shallower than this [m]
    sponge: int = 4,            # HySEA SPONGE_SIZE = 4 boundary cells
) -> torch.Tensor:
    """
    Bathymetry-weighted spatial Laplacian regulariser:

        R(u) = || ∇·(g·H·∇u) ||²

    Spherical-coordinate form:
        ∇·(gH·∇u) = g/(R²cos²φ)·∂(H·∂u/∂λ)/∂λ
                   + g/(R²cosφ) ·∂(H·cosφ·∂u/∂φ)/∂φ

    Derivation: linearise HySEA's SWE, combine continuity + momentum,
    eliminate qx/qy → variable-coefficient wave operator on η.
    Valid where η << H (deep ocean).  Masked elsewhere.

    Works for:
      max_height   : u = (B, 1, NLAT, NLON)
      eta timeseries: u = (B, NTIME, NLAT, NLON)  — each channel treated
                         as an independent spatial field.
    """
    B, C, NLAT, NLON = u.shape
    device = u.device

    H  = H_raw.view(1, 1, NLAT, NLON).expand(B, C, -1, -1)
    Hs = torch.clamp(H, min=min_depth)          # (B, C, NLAT, NLON)

    cos = _cos_phi_grid(lat_rad, B, C, NLAT, NLON, device)

    # Gradient of u in spherical coords
    du_dlon = _cd(u, dim=3, dx=dlon)            # ∂u/∂λ
    du_dlat = _cd(u, dim=2, dx=dlat)            # ∂u/∂φ

    # Weighted fluxes: F_λ = H·∂u/∂λ,  F_φ = H·cosφ·∂u/∂φ
    F_lon = Hs * du_dlon
    F_lat = Hs * cos * du_dlat

    # Divergence
    dFlon_dlon = _cd(F_lon, dim=3, dx=dlon)
    dFlat_dlat = _cd(F_lat, dim=2, dx=dlat)

    residual = (g * dFlon_dlon / (R ** 2 * cos ** 2)
              + g * dFlat_dlat / (R ** 2 * cos))

    mask = _masks(H, NLAT, NLON, min_depth, sponge)
    return (mask * residual ** 2).mean()


# ──────────────────────────────────────────────────────────────────────────────
# 2. Eikonal loss  —  arrival_times  (exact)
# ──────────────────────────────────────────────────────────────────────────────

def eikonal_loss(
    T_pred: torch.Tensor,       # (B, 1, NLAT, NLON) — normalised ∈ [0, 1]
    H_raw: torch.Tensor,        # (NLAT, NLON) — raw bathymetry [m]
    lat_rad: torch.Tensor,      # (NLAT,) — latitudes in radians
    dlon: float,                # Δλ [rad]
    dlat: float,                # Δφ [rad]
    T_max: float,               # denormalisation: T_physical = T_pred * T_max  [s]
    R: float = EARTH_RADIUS,
    g: float = G,
    min_depth: float = 10.0,
    sponge: int = 4,
    min_T_frac: float = 0.01,   # mask source region (T < 1 % of T_max)
) -> torch.Tensor:
    """
    Eikonal equation residual  |∇T|² = 1/(g·H):

        (1/(R·cosφ) · ∂T/∂λ)²  +  (1/R · ∂T/∂φ)²  =  1/(g·H)

    Derivation: tsunami wavefront propagates at long-wave phase speed
    c = √(gH).  First-arrival time T satisfies the eikonal exactly
    under the long-wave approximation — which holds across the entire
    deep ocean.  This is the exact constraint for arrival_times.

    Source region (T ≈ 0) is masked out because the eikonal breaks
    down at the wavefront origin.
    """
    B, _, NLAT, NLON = T_pred.shape
    device = T_pred.device

    # Denormalise T → physical seconds
    T_phys = T_pred * T_max                     # (B, 1, NLAT, NLON)

    H  = H_raw.view(1, 1, NLAT, NLON).expand(B, 1, -1, -1)
    Hs = torch.clamp(H, min=min_depth)

    cos = _cos_phi_grid(lat_rad, B, 1, NLAT, NLON, device)

    # Gradient of T [s/rad]
    dT_dlon = _cd(T_phys, dim=3, dx=dlon)
    dT_dlat = _cd(T_phys, dim=2, dx=dlat)

    # Physical gradient squared [s²/m²]
    grad_T_sq = (dT_dlon / (R * cos)) ** 2 + (dT_dlat / R) ** 2

    # Slowness squared 1/c² = 1/(gH) [s²/m²]
    slowness_sq = 1.0 / (g * Hs)

    residual = grad_T_sq - slowness_sq

    # Mask: ocean + sponge + exclude source region
    omask  = _masks(H, NLAT, NLON, min_depth, sponge)
    amask  = (T_pred > min_T_frac).float()      # arrived cells only
    mask   = omask * amask

    return (mask * residual ** 2).mean()

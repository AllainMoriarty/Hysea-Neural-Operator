"""
Multi-Fidelity PINO: FNO backbone + hybrid SWE physics residual.

Physics constraint is encoded in the LOSS (not the architecture),
following the PhysicsNeMo darcy_pino tutorial pattern.

Hybrid physics dispatch (two functions, one if-else):
  arrival_times → eikonal_loss()          [exact:  |∇T|² = 1/(gH)]
  max_height    → swe_spatial_loss()      [derived: ∇·(gH·∇u) = 0]
  eta           → swe_spatial_loss()      [derived: ∇·(gH·∇u) = 0]

Architecture is identical to MFFno; training differs via train_mf_pino().
No top-level execution — functions and classes only.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from physicsnemo.models.fno.fno import FNO
from torch.utils.checkpoint import checkpoint as _grad_ckpt
from config import GRAD_CKPT

from .swe_residuals import swe_spatial_loss, eikonal_loss
from .fno import _SpatialDecoder   # shared grid-size-independent encoder


# ──────────────────────────────────────────────────────────────────────────────
# Legacy fallback (kept for backward-compatibility / no-grid-info mode)
# ──────────────────────────────────────────────────────────────────────────────

def spectral_laplacian_residual(u: torch.Tensor) -> torch.Tensor:
    """
    Mean squared spectral (FFT-based) Laplacian of a 2-D spatial field.

    Parameters
    ----------
    u : (B, C, H, W) — model output field on any device

    Returns
    -------
    Scalar tensor — ||∇²u||² averaged over B, C, H, W.
    """
    # Always compute in float32: FFT is numerically unstable in BF16/FP16.
    u = u.float()
    B, C, H, W = u.shape

    kx = torch.fft.fftfreq(W, d=1.0 / W).to(u.device)
    ky = torch.fft.fftfreq(H, d=1.0 / H).to(u.device)
    ky_grid, kx_grid = torch.meshgrid(ky, kx, indexing="ij")

    lap_mult = -(kx_grid ** 2 + ky_grid ** 2)

    u_hat = torch.fft.fft2(u)
    lap_u = torch.fft.ifft2(
        lap_mult.unsqueeze(0).unsqueeze(0) * u_hat
    ).real

    return (lap_u ** 2).mean()


# ──────────────────────────────────────────────────────────────────────────────
# MFPino
# ──────────────────────────────────────────────────────────────────────────────

class MFPino(nn.Module):
    """
    Multi-Fidelity PINO.

    Architecture identical to MFFno.
    Physics loss is applied externally by the training loop via
    `pino_physics_loss()`, using the hybrid SWE dispatch.

    Output: f_LF + sigmoid(α_MF)*δ_MF + sigmoid(α_HF)*δ_HF

    Memory optimisations applied
    ----------------------------
    • feat_embed  : _SpatialDecoder instead of Linear(256, NLAT*NLON)
    • backbone    : gradient checkpointing when GRAD_CKPT=True (training only)
    • corrections : coord_features=False, padding=2 (halved)
    • physics     : accepts optional pre-computed pred to avoid double forward

    Parameters
    ----------
    task : str
        "max_height" | "arrival_times" | "eta"
        Controls which physics residual is used.
    H_raw : torch.Tensor or None
        Raw bathymetry in metres, shape (NLAT, NLON).
        Registered as a buffer — auto-moves with .to(device).
        If None, falls back to spectral Laplacian.
    grid_info : dict or None
        Must contain: "lat_rad" (NLAT,), "dlon" (float), "dlat" (float).
        For arrival_times also needs: "T_max" (float).
        Built by data.build_grid_info().
    """

    def __init__(
        self,
        nlat: int,
        nlon: int,
        latent_channels: int = 32,
        num_fno_layers:  int = 4,
        num_fno_modes:   int = 16,
        decoder_layers:  int = 2,
        decoder_layer_size: int = 128,
        n_fault_params:  int = 9,
        out_channels:    int = 1,
        # ── Physics parameters ────────────────────────────────────────────────
        task: str = "max_height",
        H_raw: torch.Tensor | None = None,
        grid_info: dict | None = None,
    ):
        super().__init__()
        self.nlat = nlat
        self.nlon = nlon
        self.task = task
        self.grid_info = grid_info or {}

        # Register raw bathymetry so it follows .to(device) / checkpoint saves
        if H_raw is not None:
            self.register_buffer("H_raw", H_raw)
        else:
            self.H_raw = None

        # Grid-size-independent fault-param encoder (replaces Linear(256, NLAT*NLON))
        self.feat_embed = _SpatialDecoder(
            nlat=nlat, nlon=nlon, n_fault_params=n_fault_params,
        )

        # Primary PINO backbone (trained with physics loss in Stage 1)
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
        )

        # Smaller correction networks — coord_features unnecessary for residuals
        _corr = dict(
            in_channels=2, out_channels=out_channels,
            decoder_layers=1, decoder_layer_size=64,
            dimension=2, latent_channels=16,
            num_fno_layers=2, num_fno_modes=8,
            padding=2, coord_features=False,
        )
        self.mf_correction = FNO(**_corr)
        self.hf_correction = FNO(**_corr)
        self.alpha_mf = nn.Parameter(torch.tensor(0.0))
        self.alpha_hf = nn.Parameter(torch.tensor(0.0))

    # ── Forward ───────────────────────────────────────────────────────────────

    def _build_input(self, fault_params: torch.Tensor,
                     bathy_t: torch.Tensor) -> torch.Tensor:
        feat  = self.feat_embed(fault_params)                          # (B, 1, NLAT, NLON)
        bathy = bathy_t.unsqueeze(0).unsqueeze(0).expand(feat.shape[0], 1, -1, -1)
        return torch.cat([feat, bathy], dim=1)

    def forward(self, fault_params: torch.Tensor, bathy_t: torch.Tensor,
                fidelity: str = "hf") -> torch.Tensor:
        x = self._build_input(fault_params, bathy_t)

        # Gradient checkpointing on the primary backbone saves ~60 % activation VRAM
        if GRAD_CKPT and self.training:
            lf_out = _grad_ckpt(self.fno, x, use_reentrant=False)
        else:
            lf_out = self.fno(x)

        if fidelity == "lf":
            return lf_out
        mf_out = lf_out + torch.sigmoid(self.alpha_mf) * self.mf_correction(x)
        if fidelity == "mf":
            return mf_out
        hf_out = mf_out + torch.sigmoid(self.alpha_hf) * self.hf_correction(x)
        return hf_out

    # ── Physics loss ──────────────────────────────────────────────────────────

    def pino_physics_loss(self, fault_params: torch.Tensor,
                          bathy_t: torch.Tensor,
                          fidelity: str = "lf",
                          pred: torch.Tensor | None = None) -> torch.Tensor:
        """
        Hybrid SWE physics constraint.

        Dispatch:
          arrival_times → eikonal_loss()     |∇T|² = 1/(gH)   [exact]
          max_height    → swe_spatial_loss() ∇·(gH·∇u) = 0    [linearised SWE]
          eta           → swe_spatial_loss() ∇·(gH·∇u) = 0    [linearised SWE]

        Falls back to spectral Laplacian if H_raw / grid_info not provided.

        Parameters
        ----------
        pred : optional pre-computed forward output.
            Pass the tensor already returned by forward() to avoid a second
            full forward pass (saves ~50 % compute in Stage 1 PINO training).
        """
        if pred is None:
            pred = self.forward(fault_params, bathy_t, fidelity=fidelity)

        gi = self.grid_info
        if self.H_raw is None or not gi:
            # Graceful fallback — original behaviour
            return spectral_laplacian_residual(pred)

        lat_rad = gi["lat_rad"]
        dlon    = gi["dlon"]
        dlat    = gi["dlat"]

        if self.task == "arrival_times":
            return eikonal_loss(
                pred, self.H_raw, lat_rad, dlon, dlat,
                T_max=gi.get("T_max", 1.0),
            )

        # max_height or eta — same operator, different output shape
        return swe_spatial_loss(pred, self.H_raw, lat_rad, dlon, dlat)

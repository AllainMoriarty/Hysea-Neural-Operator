import torch
import torch.nn as nn
from physicsnemo.models.fno.fno import FNO
from torch.utils.checkpoint import checkpoint as _grad_ckpt
from config import GRAD_CKPT


class _SpatialDecoder(nn.Module):
    """
    Compact fault-parameter → spatial-field encoder (grid-size independent).

    Architecture
    ------------
    fault_params (B, n_fault_params)
        → FC(hidden) + GELU → FC(hidden) + GELU → FC(base_h * base_w)
        → reshape (B, 1, base_h, base_w)
        → bilinear upsample → (B, 1, NLAT, NLON)

    Why this matters
    ----------------
    The old ``nn.Linear(256, NLAT*NLON)`` scales with grid area:
      • 1km grid, 1000×1500 pts → 384 M params → ~1.47 GB for one layer
    This decoder is O(hidden²) and grid-size independent:
      • any grid → ~20 K params → ~80 KB

    Parameters
    ----------
    nlat, nlon      : target output grid dimensions
    n_fault_params  : number of scalar fault parameters (branch input)
    hidden          : FC hidden width
    base_h, base_w  : compact spatial latent before upsample (default 8×8)
    """

    def __init__(self, nlat: int, nlon: int,
                 n_fault_params: int = 9, hidden: int = 256,
                 base_h: int = 8, base_w: int = 8):
        super().__init__()
        self.base_h = base_h
        self.base_w = base_w
        self.fc = nn.Sequential(
            nn.Linear(n_fault_params, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),         nn.GELU(),
            nn.Linear(hidden, base_h * base_w),
        )
        self.upsample = nn.Upsample(
            size=(nlat, nlon), mode="bilinear", align_corners=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        z = self.fc(x).view(B, 1, self.base_h, self.base_w)
        return self.upsample(z)   # (B, 1, NLAT, NLON)


class MFFno(nn.Module):
    """
    Multi-Fidelity Fourier Neural Operator.

    Input channels : fault-param embedding (1) + bathymetry (1) = 2
    coord_features : True adds 2 grid-coord channels internally → 4 total
    Output         : f_LF + sigmoid(α_MF)*δ_MF + sigmoid(α_HF)*δ_HF

    Memory optimisations applied
    ----------------------------
    • feat_embed  : _SpatialDecoder instead of Linear(256, NLAT*NLON)
    • backbone    : gradient checkpointing when GRAD_CKPT=True (training only)
    • corrections : coord_features=False, padding halved — unnecessary overhead
                    for small residual networks
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
    ):
        super().__init__()
        self.nlat = nlat
        self.nlon = nlon

        # Grid-size-independent fault-param encoder
        self.feat_embed = _SpatialDecoder(
            nlat=nlat, nlon=nlon, n_fault_params=n_fault_params,
        )

        # Primary LF backbone (gradient-checkpointed during training)
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

        # Smaller correction networks for MF / HF stages.
        # coord_features=False: coordinate channels are redundant for residuals.
        # padding=2 (was 4): smaller receptive-field extension is sufficient.
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

    def _build_input(self, fault_params: torch.Tensor,
                     bathy_t: torch.Tensor) -> torch.Tensor:
        """Embed fault params + bathy → (B, 2, NLAT, NLON)."""
        feat  = self.feat_embed(fault_params)                          # (B, 1, NLAT, NLON)
        bathy = bathy_t.unsqueeze(0).unsqueeze(0).expand(feat.shape[0], 1, -1, -1)
        return torch.cat([feat, bathy], dim=1)

    def forward(self, fault_params: torch.Tensor, bathy_t: torch.Tensor,
                fidelity: str = "hf") -> torch.Tensor:
        x = self._build_input(fault_params, bathy_t)

        # physicsnemo's spectral_layers.rfft2 does NOT support BF16/FP16.
        # Disable autocast for all FNO calls so they always run in float32,
        # while AMP remains active for the outer training loop.
        with torch.amp.autocast("cuda", enabled=False):
            x32 = x.float()
            if GRAD_CKPT and self.training:
                lf_out = _grad_ckpt(self.fno, x32, use_reentrant=False)
            else:
                lf_out = self.fno(x32)
            if fidelity == "lf":
                return lf_out
            mf_out = lf_out + torch.sigmoid(self.alpha_mf) * self.mf_correction(x32)
            if fidelity == "mf":
                return mf_out
            hf_out = mf_out + torch.sigmoid(self.alpha_hf) * self.hf_correction(x32)
        return hf_out

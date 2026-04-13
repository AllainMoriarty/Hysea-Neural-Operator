import torch
import torch.nn as nn
from physicsnemo.models.fno import FNO


class MFFno(nn.Module):
    """
    Multi-Fidelity Fourier Neural Operator.

    Input channels : fault-param embedding (1) + bathymetry (1) = 2
    coord_features : True adds 2 grid-coord channels internally → 4 total
    Output         : f_LF + sigmoid(α_MF)*δ_MF + sigmoid(α_HF)*δ_HF
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

        # Embed fault params (9,) → spatial field (NLAT*NLON,)
        self.feat_embed = nn.Sequential(
            nn.Linear(n_fault_params, 256), nn.GELU(),
            nn.Linear(256, nlat * nlon),
        )

        # Primary LF backbone
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

        # Smaller correction networks for MF / HF stages
        _corr = dict(
            in_channels=2, out_channels=out_channels,
            decoder_layers=1, decoder_layer_size=64,
            dimension=2, latent_channels=16,
            num_fno_layers=2, num_fno_modes=8,
            padding=4, coord_features=True,
        )
        self.mf_correction = FNO(**_corr)
        self.hf_correction = FNO(**_corr)
        self.alpha_mf = nn.Parameter(torch.tensor(0.0))
        self.alpha_hf = nn.Parameter(torch.tensor(0.0))

    def _build_input(self, fault_params, bathy_t):
        """Embed fault params + bathy → (B, 2, NLAT, NLON)."""
        B     = fault_params.shape[0]
        feat  = self.feat_embed(fault_params).view(B, 1, self.nlat, self.nlon)
        bathy = bathy_t.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
        return torch.cat([feat, bathy], dim=1)

    def forward(self, fault_params, bathy_t, fidelity: str = "hf"):
        x      = self._build_input(fault_params, bathy_t)
        lf_out = self.fno(x)
        if fidelity == "lf":
            return lf_out
        mf_out = lf_out + torch.sigmoid(self.alpha_mf) * self.mf_correction(x)
        if fidelity == "mf":
            return mf_out
        hf_out = mf_out + torch.sigmoid(self.alpha_hf) * self.hf_correction(x)
        return hf_out

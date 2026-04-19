"""
Multi-Fidelity DeepONet — pure PyTorch implementation.

Why not physicsnemo.sym.DeepONetArch?
--------------------------------------
The installed physicsnemo.sym.DeepONetArch.concat_input() expects the branch
(fault_params, batch B) and trunk (query_coords, NPTS grid points) inputs to
share the same first dimension, which is impossible for our use-case where
B << NPTS.  The correct DeepONet formulation is:

    output(b, n) = branch(fault_params[b]) · trunk(query_coords[n]) + bias

i.e. an outer product via matmul, giving shape (B, NPTS).

Architecture
------------
Branch net : MLP(n_fault_params → p)   — encodes the input function (fault)
Trunk net  : MLP(trunk_dim → p)        — encodes query coordinates (spatial)
Output     : branch @ trunk.T + bias   — shape (B, NPTS)

Multi-fidelity
--------------
Output = f_LF + sigmoid(α_MF)·δ_MF + sigmoid(α_HF)·δ_HF
Each of f_LF, δ_MF, δ_HF is a separate (_SingleDeepONet) with shared trunk.
"""
import torch
import torch.nn as nn


# ── Building blocks ───────────────────────────────────────────────────────────

class _MLP(nn.Module):
    """Fully-connected network with GELU activations."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, n_layers: int = 4):
        super().__init__()
        sizes = [in_dim] + [hidden] * (n_layers - 1) + [out_dim]
        layers: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SingleDeepONet(nn.Module):
    """
    One DeepONet (branch + trunk → outer product output).

    Parameters
    ----------
    p           : shared latent dimension
    hidden      : MLP hidden width
    trunk_dim   : dimensionality of query coordinates (2 spatial, 3 spatio-temporal)
    n_fault_params : number of branch input scalars
    """

    def __init__(self, p: int, hidden: int, trunk_dim: int,
                 n_fault_params: int = 9):
        super().__init__()
        self.branch = _MLP(n_fault_params, hidden, p)
        self.trunk  = _MLP(trunk_dim, hidden, p)
        self.bias   = nn.Parameter(torch.zeros(1))

    def forward(self, fault_params: torch.Tensor,
                query_coords: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        fault_params  : (B, n_fault_params) — one per scenario
        query_coords  : (NPTS, trunk_dim)   — shared spatial/spatio-temporal grid

        Returns
        -------
        (B, NPTS)
        """
        b = self.branch(fault_params)   # (B, p)
        t = self.trunk(query_coords)    # (NPTS, p)
        return b @ t.T + self.bias      # (B, NPTS)


# ── Multi-fidelity DeepONet ───────────────────────────────────────────────────

class MFDeepONet(nn.Module):
    """
    Multi-Fidelity DeepONet.

    Three fidelity levels: LF → MF correction → HF correction.
    Output: f_LF + sigmoid(α_MF)*δ_MF + sigmoid(α_HF)*δ_HF

    No physicsnemo.sym dependency — avoids concat_input batch-shape mismatch.
    """

    def __init__(self, p: int = 128, hidden: int = 256, trunk_dim: int = 2,
                 n_fault_params: int = 9):
        super().__init__()
        kwargs = dict(p=p, hidden=hidden, trunk_dim=trunk_dim,
                      n_fault_params=n_fault_params)
        self.lf       = _SingleDeepONet(**kwargs)
        self.mf       = _SingleDeepONet(**kwargs)
        self.hf       = _SingleDeepONet(**kwargs)
        self.alpha_mf = nn.Parameter(torch.tensor(0.0))
        self.alpha_hf = nn.Parameter(torch.tensor(0.0))
        self.trunk_dim = trunk_dim

    def forward(self, fault_params: torch.Tensor, query_coords: torch.Tensor,
                fidelity: str = "hf") -> torch.Tensor:
        """
        Parameters
        ----------
        fault_params  : (B, n_fault_params)
        query_coords  : (NPTS, trunk_dim) — e.g. db.q_sp  shape (NLAT*NLON, 2)
        fidelity      : "lf" | "mf" | "hf"

        Returns
        -------
        (B, NPTS)
        """
        lf_out = self.lf(fault_params, query_coords)
        if fidelity == "lf":
            return lf_out
        mf_out = (lf_out
                  + torch.sigmoid(self.alpha_mf)
                  * self.mf(fault_params, query_coords))
        if fidelity == "mf":
            return mf_out
        hf_out = (mf_out
                  + torch.sigmoid(self.alpha_hf)
                  * self.hf(fault_params, query_coords))
        return hf_out

import torch
import torch.nn as nn
from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.models.fourier_net import FourierNetArch
from physicsnemo.sym.models.deeponet import DeepONetArch
from physicsnemo.sym.key import Key


def build_deeponet(p: int = 128, hidden: int = 256, trunk_dim: int = 2, n_fault_params: int = 9) -> DeepONetArch:
    """
    Build a single DeepONetArch instance.

    Parameters
    ----------
    p              : latent dimension shared by branch and trunk outputs
    hidden         : hidden layer width
    trunk_dim      : query-coord dimensionality (2=spatial, 3=spatio-temporal)
    n_fault_params : number of fault-parameter inputs to the branch
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
        frequencies=("axis", list(range(10))),
    )
    return DeepONetArch(
        output_keys=[Key("field_output")],
        branch_net=branch_net,
        trunk_net=trunk_net,
        branch_dim=p,
        trunk_dim=p,
    )


class MFDeepONet(nn.Module):
    """
    Multi-Fidelity DeepONet.

    Three fidelity levels: LF → MF correction → HF correction.
    Output: f_LF + sigmoid(α_MF)*δ_MF + sigmoid(α_HF)*δ_HF
    """

    def __init__(self, p: int = 128, hidden: int = 256, trunk_dim: int = 2):
        super().__init__()
        self.lf       = build_deeponet(p, hidden, trunk_dim)
        self.mf       = build_deeponet(p, hidden, trunk_dim)
        self.hf       = build_deeponet(p, hidden, trunk_dim)
        self.alpha_mf = nn.Parameter(torch.tensor(0.0))
        self.alpha_hf = nn.Parameter(torch.tensor(0.0))
        self.trunk_dim = trunk_dim

    def _forward_one(self, net, fault_params, query_coords):
        out = net({"fault_params": fault_params, "query_coords": query_coords})
        return out["field_output"]

    def forward(self, fault_params, query_coords, fidelity: str = "hf"):
        lf_out = self._forward_one(self.lf, fault_params, query_coords)
        if fidelity == "lf":
            return lf_out
        mf_out = (lf_out
                  + torch.sigmoid(self.alpha_mf)
                  * self._forward_one(self.mf, fault_params, query_coords))
        if fidelity == "mf":
            return mf_out
        hf_out = (mf_out
                  + torch.sigmoid(self.alpha_hf)
                  * self._forward_one(self.hf, fault_params, query_coords))
        return hf_out

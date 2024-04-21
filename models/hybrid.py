import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from nqgl.sc_sae.models.configs import SAEConfig
from unpythonic import box
from jaxtyping import Float
from nqgl.sc_sae.models.base_sae import BaseSAE


class HybridSCSAEMulGrads(BaseSAE):
    MODEL_TYPE = "HybridSCSAE_MulGrads"

    def _encode(self, x_cent, spoofed_acts_box):
        mul = x_cent @ self.W_enc
        pre_acts = mul + self.b_enc
        active = mul * (pre_acts - (pre_acts - 1).detach())
        gate = (pre_acts > 0) & (active > 0)
        return torch.where(gate, active, 0)


class HybridSCSAERegGrads(BaseSAE):
    MODEL_TYPE = "HybridSCSAE_RegGrads"

    def _encode(self, x_cent, spoofed_acts_box):
        mul = x_cent @ self.W_enc
        pre_acts = mul + self.b_enc
        active_lin = pre_acts - self.b_enc.detach()
        gate = (pre_acts > 0) & (active_lin > 0)
        return torch.where(
            self.b_enc > 0,
            torch.relu(pre_acts),
            torch.where(gate, active_lin, 0),
        )

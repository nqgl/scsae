import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from nqgl.sc_sae.models.configs import SAEConfig
from unpythonic import box
from jaxtyping import Float
from nqgl.sc_sae.models.base_sae import BaseSAE


class LinearScaleSAENonMulGrads(BaseSAE):
    MODEL_TYPE = "LinearScaleSAE_NonMulGrads"

    def encode(self, x, spoofed_acts_box):
        mul = (x - self.b_dec) @ self.W_enc
        pre_acts = mul + self.b_enc
        active = pre_acts - self.b_enc.detach()
        gate = (pre_acts > 0) & (active > 0)
        return torch.where(gate, active, 0)

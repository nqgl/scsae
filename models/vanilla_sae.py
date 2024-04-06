import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from nqgl.sc_sae.models.configs import SAEConfig
from unpythonic import box
from jaxtyping import Float
from nqgl.sc_sae.models.base_sae import BaseSAE


class VanillaSAE(BaseSAE):
    MODEL_TYPE = "VanillaSAE"

    def encode(self, x, spoofed_acts_box):
        mul = (x - self.b_dec) @ self.W_enc
        pre_acts = mul + self.b_enc
        return torch.relu(pre_acts)

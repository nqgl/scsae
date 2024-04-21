import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from nqgl.sc_sae.model.configs import SAEConfig
from unpythonic import box
from jaxtyping import Float
from nqgl.sc_sae.model.base_sae import BaseSAE


class SpoofedTodo(BaseSAE):
    raise NotImplementedError
    MODEL_TYPE = ...

    def _encode(self, x_cent, spoofed_acts_box):
        raise NotImplementedError

        mul = x_cent @ self.W_enc
        pre_acts = mul + self.b_enc
        # active = pre_acts - self.b_enc.detach()
        # gate = (pre_acts > 0) & (active > 0)
        # return torch.where(gate, active, 0)

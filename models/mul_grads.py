import torch
from torch.cuda.amp import custom_bwd, custom_fwd
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from nqgl.sc_sae.models.fwd_fn_saes import ForwardFnSAE
from nqgl.sc_sae.models.configs import SAEConfig
from unpythonic import box
from jaxtyping import Float
from nqgl.sc_sae.models.base_sae import BaseSAE


class LegacySCSAE(BaseSAE):
    MODEL_TYPE = "Legacy_SCSAE_MulGrads"

    def _encode(self, x_cent, spoofed_acts_box):
        mul = x_cent @ self.W_enc
        pre_acts = mul + self.b_enc
        active = mul * (pre_acts - (pre_acts - 1).detach())
        gate = (pre_acts > 0) & (active > 0)
        return torch.where(gate, active, 0)


class SCSAE(ForwardFnSAE):
    MODEL_TYPE = "SCSAE_MulGrads"

    class ForwardFn(torch.autograd.Function):
        @staticmethod
        @custom_fwd
        def forward(ctx, mul, bias, spoofed_acts_box):
            pre_acts = mul + bias
            gate = (pre_acts > 0) & (mul > 0)
            ctx.save_for_backward(mul, bias, gate, pre_acts)
            # active = mul * (pre_acts - (pre_acts - 1).detach())
            return torch.where(gate, mul, 0)

        @staticmethod
        @custom_bwd
        def backward(ctx, grad_output):
            mul, bias, gate, pre_acts = ctx.saved_tensors
            gated_grad = torch.where(gate, grad_output, 0)
            # grad_mul, grad_bias = gated_grad.clone(), gated_grad.clone()
            grad_bias = gated_grad * mul
            grad_mul = gated_grad + grad_bias.clone()

            return grad_mul, grad_bias, None

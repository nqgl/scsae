import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from nqgl.sc_sae.models.equiv_fwd_fns import ForwardFnSAE
from nqgl.sc_sae.models.configs import SAEConfig
from unpythonic import box
from nqgl.sc_sae.models.base_sae import BaseSAE
import wandb
from torch.cuda.amp import custom_bwd, custom_fwd


class SCSAE(BaseSAE):
    MODEL_TYPE = "SCSAE_MulGrads"

    def _encode(self, x_cent, spoofed_acts_box):
        mul = x_cent @ self.W_enc
        pre_acts = mul + self.b_enc
        active = mul * (pre_acts - (pre_acts - 1).detach())
        gate = (pre_acts > 0) & (active > 0)
        return torch.where(gate, active, 0)


# class NonMulEquivFn(LinearForwardFn): ...


# class SCSAE_MulMk2(BaseSAE):
#     MODEL_TYPE = "SCSAE_MulMk2"

#     class ForwardFn(torch.autograd.Function):
#         @staticmethod
#         def forward(ctx, mul, bias, spoofed_acts_box):
#             pre_acts = mul + bias
#             gate = (pre_acts > 0) & (mul > 0)
#             ctx.save_for_backward(mul, bias, gate, pre_acts)
#             # active = mul * (pre_acts - (pre_acts - 1).detach())
#             return torch.where(gate, mul, 0)

#         @staticmethod
#         def backward(ctx, grad_output):
#             mul, bias, gate, pre_acts = ctx.saved_tensors
#             gated_grad = torch.where(gate, grad_output, 0)
#             # grad_mul, grad_bias = gated_grad.clone(), gated_grad.clone()
#             grad_bias = gated_grad
#             grad_mul = gated_grad + gated_grad * mul

#             return grad_mul, grad_bias, None

#     def _encode(self, x_cent, spoofed_acts_box):
#         mul = x_cent @ self.W_enc

#         return self.ForwardFn.apply(mul, self.b_enc, spoofed_acts_box)


class SCSAE_MulMk2(ForwardFnSAE):
    MODEL_TYPE = "SCSAE_MulMk2"

    class ForwardFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, mul, bias, spoofed_acts_box):
            pre_acts = mul + bias
            gate = (pre_acts > 0) & (mul > 0)
            ctx.save_for_backward(mul, bias, gate, pre_acts)
            # active = mul * (pre_acts - (pre_acts - 1).detach())
            return torch.where(gate, mul, 0)

        @staticmethod
        def backward(ctx, grad_output):
            mul, bias, gate, pre_acts = ctx.saved_tensors
            gated_grad = torch.where(gate, grad_output, 0)
            # grad_mul, grad_bias = gated_grad.clone(), gated_grad.clone()
            grad_bias = (
                torch.where(
                    bias < 0,
                    torch.sigmoid(pre_acts) * (1 - torch.sigmoid(pre_acts)),
                    0,
                )
                * gated_grad
                * mul
            )
            grad_mul = gated_grad * (
                1 + mul * torch.where(bias < 0, torch.sigmoid(pre_acts), 1)
            )

            return grad_mul, grad_bias, None


class SCSAE_2Bias(ForwardFnSAE):
    MODEL_TYPE = "SCSAE_2Bias"

    class ForwardFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, mul, bias, translation_bias, spoofed_acts_box):
            pre_acts = mul + bias
            active = mul + translation_bias
            gate = (pre_acts > 0) & (active > 0)
            ctx.save_for_backward(mul, bias, gate, pre_acts, active)
            # active = mul * (pre_acts - (pre_acts - 1).detach())
            return torch.where(gate, active, 0)

        @staticmethod
        def backward(ctx, grad_output):
            mul, bias, gate, pre_acts, active = ctx.saved_tensors
            gated_grad = torch.where(gate, grad_output, 0)
            # grad_mul, grad_bias = gated_grad.clone(), gated_grad.clone()
            # grad_bias = (
            #     torch.where(
            #         bias < 0,
            #         torch.sigmoid(pre_acts) * (1 - torch.sigmoid(pre_acts)),
            #         0,
            #     )
            #     * gated_grad
            #     * mul
            # )
            grad_mul = gated_grad
            # * (
            #     1 + mul * torch.where(bias < 0, torch.sigmoid(pre_acts), 1)
            # )
            grad_bias = (
                (gated_grad + grad_output)
                * torch.sigmoid(pre_acts)
                * (1 - torch.sigmoid(pre_acts))
            )

            # grad_mul, grad_bias = gated_grad.clone(), gated_grad.clone()
            grad_t_bias = torch.where(pre_acts > 0, grad_output, 0)
            # grad_bias = (
            #     torch.where(active > 0, grad_output, 0)
            #     * torch.sigmoid(pre_acts)
            #     * (1 - torch.sigmoid(pre_acts))
            # )
            grad_mul = gated_grad  # + gated_grad * mul

            return grad_mul, grad_bias, grad_t_bias, None

    def __init__(self, cfg: SAEConfig):
        super().__init__(cfg)
        self.b_enc_trans = nn.Parameter(self.b_enc.data.clone().detach())

    def _encode(self, x_cent, spoofed_acts_box):
        mul = x_cent @ self.W_enc
        return self.ForwardFn.apply(mul, self.b_enc, self.b_enc_trans, spoofed_acts_box)

    @torch.no_grad()
    def post_backward(self):
        self.b_enc.data[self.b_enc > 0] = 0
        return super().post_backward()


class InterpolatedStartSCSAE(BaseSAE):
    MODEL_TYPE = "InterpolatedStartSCSAE"

    def _encode(self, x_cent, spoofed_acts_box):
        mul = x_cent @ self.W_enc
        pre_acts = mul + self.b_enc
        normal = torch.relu(pre_acts)
        cool = mul
        out = normal.lerp(
            torch.relu(cool).float(), (mul.detach() + self.b_enc).clamp(0, 0.1) * 10
        )
        return out

    @torch.no_grad()
    def post_backward(self):
        self.b_enc.data[self.b_enc > 0] = 0
        return super().post_backward()


class SpoofTestSAE(BaseSAE):
    MODEL_TYPE = "SpoofTestSAE"

    def __init__(self, cfg: SAEConfig):
        super().__init__(cfg)
        f = torch.rand(1).item()
        i = torch.randint(0, 2, (1,))
        if i.item == 0:
            f = f * torch.rand(1).item()
        self.interpolation_ratio = f
        wandb.config.update({"interpolation_ratio": self.interpolation_ratio})

    def _encode(self, x_cent, spoofed_acts_box):
        # mul = x_cent @ self.W_enc
        # pre_acts = mul + self.b_enc
        # active = pre_acts - self.b_enc.detach()
        # gate = (active > 0) & (mul > 0)
        # active2 = active * (pre_acts - (pre_acts - 1).detach())

        # return torch.where(gate, active2, 0)
        # spoofed_acts_box << torch.where(gate, mul, 0)
        mul = x_cent @ self.W_enc
        pre_acts = mul + self.b_enc
        active = (
            mul * (pre_acts - (pre_acts - 1).detach()) * 0.5
            + (pre_acts - self.b_enc.detach()) * 0.5
        )
        gate = (pre_acts > 0) & (active > 0)
        return torch.where(gate, active, 0)

    # @torch.no_grad()
    # def post_backward(self):
    #     self.b_enc.data[self.b_enc > 0] = 0
    #     return super().post_backward()


"""
hey, so I think the bias grads should be scaled by the encoder size?
maybe like just that
that seems like it makes sense in some way
does it? yeah it make the inhibition not scale-dependent
but then how do size-specific updates behave?
bc there are 2 types of info
    - gating
    - scaling
"""


class NoEncSAE(BaseSAE):
    MODEL_TYPE = "NoEncSAE"

    def _encode(self, x_cent, spoofed_acts_box):
        mul = x_cent @ self.W_enc
        return torch.relu(mul)

    # @torch.no_grad()
    # def post_backward(self):
    #     self.b_enc.data[self.b_enc > 0] = 0
    #     return super().post_backward()


class SCSAE_LogGrad(ForwardFnSAE):
    MODEL_TYPE = "SCSAE_LogGrad"

    class ForwardFn(torch.autograd.Function):
        @staticmethod
        @custom_fwd
        def forward(ctx, mul, bias, spoofed_acts_box):
            pre_acts = mul + bias
            gate = (pre_acts > 0) & (mul > 0)
            ctx.save_for_backward(mul, bias, gate, pre_acts)
            return torch.where(gate, mul, 0)

        @staticmethod
        @custom_bwd
        def backward(ctx, grad_output):
            mul, bias, gate, pre_acts = ctx.saved_tensors
            gated_grad = torch.where(gate, grad_output, 0)

            logmult = torch.where(gate, 1 / (pre_acts + 0.01), 0)
            grad_mul, grad_bias = gated_grad.clone(), gated_grad * logmult
            return grad_mul, grad_bias, None


class SCSAE_SigGate(ForwardFnSAE):
    MODEL_TYPE = "SCSAE_SigGate"

    class ForwardFn(torch.autograd.Function):
        @staticmethod
        @custom_fwd
        def forward(ctx, mul, bias, spoofed_acts_box):
            pre_acts = mul + bias
            gate = (pre_acts > 0) & (mul > 0)
            ctx.save_for_backward(mul, bias, gate, pre_acts)
            return torch.where(gate, mul, 0)

        @staticmethod
        @custom_bwd
        def backward(ctx, grad_output):
            mul, bias, gate, pre_acts = ctx.saved_tensors
            gated_grad = torch.where(gate, grad_output, 0)

            logmult = torch.where(gate, 1 / (pre_acts + 0.01), 0)
            grad_mul = gated_grad.clone()
            grad_bias = (
                grad_output * (1 - torch.sigmoid(pre_acts)) * torch.sigmoid(pre_acts)
            )
            return grad_mul, grad_bias, None

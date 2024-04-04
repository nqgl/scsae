import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from nqgl.hsae_re.sneaky_bias.simple_reimpl.configs import SAEConfig
from unpythonic import box
from jaxtyping import Float


class BiasAdjustedSAE(nn.Module):
    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        self.W_enc: Float[Tensor, "d_data d_dict"] = nn.Parameter(
            nn.init.kaiming_uniform_(
                torch.empty(
                    cfg.d_data,
                    cfg.d_dict,
                )
            )
        )
        self.W_dec: Float[Tensor, "d_dict d_data"] = nn.Parameter(
            nn.init.kaiming_uniform_(
                torch.empty(
                    cfg.d_dict,
                    cfg.d_data,
                )
            )
        )
        if cfg.tied_init:
            self.W_enc.data[:] = self.W_dec.data.t()
        self.b_enc: Float[Tensor, "d_dict"] = nn.Parameter(torch.zeros(cfg.d_dict))
        self.b_dec: Float[Tensor, "d_data"] = (
            nn.Parameter(torch.zeros(cfg.d_data)) if self.cfg.use_b_dec else 0
        )
        self.norm_dec()

    def forward(
        self,
        x: Float[Tensor, "batch d_data"],
        acts_box: box = None,
        spoofed_acts_box: box = None,
    ):
        acts_box = acts_box or box()
        spoofed_acts_box = spoofed_acts_box or box()
        acts = self.encode(x)
        acts_box << acts
        spoofed_acts_box << acts
        return self.decode(acts)

    def encode(self, x):
        mul = (x - self.b_dec) @ self.W_enc
        pre_acts = mul + self.b_enc
        active = mul * (pre_acts - (pre_acts - 1).detach())
        gate = (pre_acts > 0) & (active > 0)
        return torch.where(gate, active, 0)

    # def encode2(self, x):
    #     mul = (x - self.b_dec + self.b_dec2) @ self.W_enc
    #     pre_acts = mul + self.b_enc
    #     # active = mul * (pre_acts - (pre_acts - 1).detach())
    #     # sanity check, this should be identical too
    #     # but just copy pasting stuff to check
    #     active = pre_acts - self.b_enc
    #     active = active * (pre_acts - (pre_acts - 1).detach())
    #     gate = (pre_acts > 0) & (active > 0)

    #     acts = torch.where(gate, active, 0)
    #     active = pre_acts - cache.bias.detach()
    #     return acts

    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec

    def post_backward(self):
        self.orthogonalize_dec_grads()

    def post_step(self):
        self.norm_dec()

    @torch.no_grad()
    def norm_dec(self):
        norm = self.W_dec.norm(dim=-1, keepdim=True)
        normed = self.W_dec / norm
        self.W_dec[:] = (
            torch.where(norm > 1, normed, self.W_dec)
            if self.cfg.selectively_norm_dec
            else normed
        )

    @torch.no_grad()
    def orthogonalize_dec_grads(self):
        grad = self.W_dec.grad
        dec_normed = self.W_dec.data / self.W_dec.data.norm(dim=-1, keepdim=True)
        grad_orth = grad - (dec_normed * grad).sum(dim=-1, keepdim=True) * dec_normed
        self.W_dec.grad[:] = grad_orth

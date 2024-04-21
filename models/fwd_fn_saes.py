from nqgl.sc_sae.models.base_sae import BaseSAE
import torch


class ForwardFnSAE(BaseSAE):
    ForwardFn: torch.autograd.Function

    def _encode(self, x_cent, spoofed_acts_box):
        mul = x_cent @ self.W_enc
        return self.ForwardFn.apply(mul, self.b_enc, spoofed_acts_box)

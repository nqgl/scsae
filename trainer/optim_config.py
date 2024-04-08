from dataclasses import dataclass
import torch


@dataclass
class OptimConfigBase:
    lr: float = 1e-3
    optim: str = "adam"
    beta1: float = 0.9
    beta2: float = 0.99

    def get_optim(self, params):
        if self.optim == "adam":
            return torch.optim.Adam(params, lr=self.lr, betas=self.betas)
        else:
            raise ValueError(f"Unknown optimizer {self.optim}")

    @property
    def betas(self):
        return (self.beta1, self.beta2)

    @betas.setter
    def betas(self, value):
        self.beta1, self.beta2 = value


class OptimConfig(OptimConfigBase):

    def __init__(self, **kwargs):
        if "betas" in kwargs:
            betas = kwargs.pop("betas")
            kwargs["beta1"], kwargs["beta2"] = betas
        super().__init__(**kwargs)

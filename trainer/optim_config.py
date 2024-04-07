from dataclasses import dataclass
import torch


@dataclass
class OptimConfig:
    lr: float = 1e-3
    betas: tuple = (0.9, 0.98)
    optim: str = "adam"

    def get_optim(self, params):
        if self.optim == "adam":
            return torch.optim.Adam(params, lr=self.lr, betas=self.betas)
        else:
            raise ValueError(f"Unknown optimizer {self.optim}")

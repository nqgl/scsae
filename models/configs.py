from dataclasses import dataclass
import torch


@dataclass
class LrSchedulerConfig:
    warmup_steps: int = 2_000
    cooldown_begin: int = 60_000
    cooldown_period: int = 20_000
    cooldown_factor: int = 10


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


@dataclass
class SAEConfig:
    d_data: int = 768
    dict_mult: int = 4
    dtype: str = None
    device: str = "cuda"
    l1_coeff: float = 1e-3
    tied_init: bool = True
    use_b_dec: bool = True
    selectively_norm_dec: bool = False
    d_dict: int = None
    sae_type: str = "LinearScaleSAE_MulGrads"

    def __post_init__(self):
        if self.d_dict is None:
            self.d_dict = self.d_data * self.dict_mult


@dataclass
class DataConfig:
    set_bos: bool = True
    dataset: str = "alancooney/sae-monology-pile-uncopyrighted-tokenizer-gpt2"
    seq_mul: int = 2
    model_name: str = "gpt2"

    def get_train_data(self): ...

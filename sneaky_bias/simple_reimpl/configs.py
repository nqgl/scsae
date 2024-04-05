from dataclasses import dataclass, field
from nqgl.hsae_re.data.buffer2 import BufferConfig
import torch


@dataclass
class LrSchedulerConfig:
    warmup_steps: int = 5_000
    cooldown_begin: int = 60_000
    cooldown_period: int = 20_000


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

    def __post_init__(self):
        if self.d_dict is None:
            self.d_dict = self.d_data * self.dict_mult

    ### Maybe add these options back in later
    # norm_dec_grads: bool = True
    # train_continue_path: str = None
    # b_enc_init: float = 0.0
    # bias_lr_coeff: float = 1


@dataclass
class DataConfig:
    set_bos: bool = True
    dataset: str = "alancooney/sae-monology-pile-uncopyrighted-tokenizer-gpt2"
    seq_mul: int = 2
    model_name: str = "gpt2"

    def get_train_data(self): ...


@dataclass
class SAETrainConfig:
    l1_coeff: float = 1 / 12
    wandb_project: str = "bias_thing_reimplemented"
    sae_cfg: SAEConfig = field(default_factory=SAEConfig)
    buffer_cfg: BufferConfig = field(default_factory=BufferConfig)
    optim_cfg: OptimConfig = field(default_factory=OptimConfig)
    data_cfg: DataConfig = None
    use_autocast: bool = True
    lr_schedule: bool = True
    lr_scheduler_cfg: LrSchedulerConfig = field(default_factory=LrSchedulerConfig)

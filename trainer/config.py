from models.configs import DataConfig, LrSchedulerConfig, OptimConfig, SAEConfig
from nqgl.hsae_re.data.buffer2 import BufferConfig


from dataclasses import dataclass, field


@dataclass
class SAETrainConfig:
    l1_coeff: float = 1 / 12
    sparsity_penalty_type: str = "l1"
    wandb_project: str = "bias_thing_reimplemented"
    sae_cfg: SAEConfig = field(default_factory=SAEConfig)
    buffer_cfg: BufferConfig = field(default_factory=BufferConfig)
    optim_cfg: OptimConfig = field(default_factory=OptimConfig)
    data_cfg: DataConfig = None
    use_autocast: bool = True
    lr_schedule: bool = True
    lr_scheduler_cfg: LrSchedulerConfig = field(default_factory=LrSchedulerConfig)

    def sparsity_penalty_from_l1(self, l1):
        if self.sparsity_penalty_type == "l1":
            return l1 * self.l1_coeff
        else:
            raise ValueError(f"Unknown penalty type {self.sparsity_penalty_type}")

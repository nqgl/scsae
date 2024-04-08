from data.dataset import DataConfig
from models.configs import SAEConfig
from nqgl.sc_sae.from_hsae_re.buffer2_no_cast import BufferConfig
from nqgl.sc_sae.trainer.optim_config import OptimConfig
from nqgl.sc_sae.data.dataset import DataConfig
from nqgl.mlutils.components.config import WandbDynamicConfig
from dataclasses import dataclass, field

from trainer.sched_config import LrSchedulerConfig


@dataclass
class SAETrainConfig(WandbDynamicConfig):
    l1_coeff: float = 1e-3
    sparsity_penalty_type: str = "l1"
    wandb_project: str = "bias_thing_reimplemented"
    sae_cfg: SAEConfig = field(default_factory=SAEConfig)
    buffer_cfg: BufferConfig = field(default_factory=BufferConfig)
    optim_cfg: OptimConfig = field(default_factory=OptimConfig)
    data_cfg: DataConfig = None
    use_autocast: bool = True
    lr_schedule: bool = True
    lr_scheduler_cfg: LrSchedulerConfig = field(default_factory=LrSchedulerConfig)

    def sparsity_penalty(self, acts):
        acts = acts.relu()
        if self.sparsity_penalty_type == "l1_sqrt":
            acts = acts.sqrt()
        elif not self.sparsity_penalty_type == "l1":
            raise ValueError(f"Unknown penalty type {self.sparsity_penalty_type}")
        return acts.mean(dim=0).sum() * self.l1_coeff

    def sparsity_penalty_from_l1(self, l1):
        if self.sparsity_penalty_type == "l1":
            return l1 * self.l1_coeff
        else:
            raise ValueError(f"Unknown penalty type {self.sparsity_penalty_type}")

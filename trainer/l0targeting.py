from nqgl.sc_sae.trainer import Trainer
from nqgl.sc_sae.trainer.config import SAETrainConfig


class L0TargetingTrainer(Trainer):
    def __init__(
        self,
        cfg: SAETrainConfig,
        model,
        val_tokens,
        legacy_cfg,
        target_l0: float,
        adjust_eps=0.0001,
    ):
        super().__init__(cfg, model, val_tokens, legacy_cfg)
        self.target_l0 = target_l0
        self.adjust_eps = adjust_eps

    def log_step(self, logdict):
        if self.t > 3000:
            if logdict["l0"] > self.target_l0:
                self.cfg.l1_coeff *= 1 + self.adjust_eps
            else:
                self.cfg.l1_coeff *= 1 - self.adjust_eps
        return super().log_step(logdict)

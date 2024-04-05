from nqgl.sc_sae.model import Trainer
import wandb

from nqgl.sc_sae.model.configs import SAETrainConfig


class MultiLoggedTrainer(Trainer):
    def __init__(self, model_name, cfg: SAETrainConfig, model, val_tokens, legacy_cfg):
        super().__init__(cfg, model, val_tokens, legacy_cfg)
        self.model_name = model_name

    def log(self, logdict, step=None):
        d = {f"{k}/{self.model_name}": v for k, v in logdict.items()}
        super().log(d, step=step)


class MultiTrainer:
    def __init__(self, cfgs, model, val_tokens, legacy_cfg):
        self.trainers = []
        self.cfg = {f"model{i}": cfg for i, cfg in enumerate(cfgs)}
        for k, cfg in self.cfg.items():
            self.trainers.append(
                MultiLoggedTrainer(k, cfg, model, val_tokens, legacy_cfg)
            )

    def train(self, datasource, skip_wandb=False):
        assert wandb.run is None or skip_wandb
        wandb.init(
            entity="sae_all",
            project="multi-test",
            config=self.cfg,
        )

        for x in datasource:
            for trainer in self.trainers:
                trainer.trainstep(x)

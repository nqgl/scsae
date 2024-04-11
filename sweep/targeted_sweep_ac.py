import torch.backends
import torch.backends.cuda
from nqgl.sc_sae.models import test
from nqgl.sc_sae.data import ac_cfg
from nqgl.sc_sae.sweep.swept_config import (
    get_configs_from_sweep,
    ConfigFromSweep,
)
from nqgl.sc_sae.models import test
from nqgl.sc_sae.trainer.l0targeting import L0TargetingTrainer
import tqdm

import wandb

import torch

# torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def run():
    wandb.init()
    cfg, lgcfg = get_configs_from_sweep(ConfigFromSweep(**wandb.config))
    trainer = L0TargetingTrainer(
        cfg,
        model=test.model,
        val_tokens=test.val_tokens,
        legacy_cfg=lgcfg,
        target_l0=25,
        adjust_eps=0,
    )
    wandb.config.update(trainer.cfg)

    ac = ac_cfg.ac
    trainer.train(ac.read_as_iter(trainer.cfg.buffer_cfg.batch_size))
    wandb.finish()


sweep_id = open("sweep/sweep_id.txt").read().strip()

SWEEP_NAME = "sweep_from_config_test"


def main():
    wandb.agent(sweep_id, function=run, entity="sae_all", project=SWEEP_NAME)


if __name__ == "__main__":
    main()

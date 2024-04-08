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


def run():
    wandb.init()
    cfg, lgcfg = get_configs_from_sweep(ConfigFromSweep(**wandb.config))
    trainer = L0TargetingTrainer(
        cfg,
        model=test.model,
        val_tokens=test.val_tokens,
        legacy_cfg=lgcfg,
        target_l0=25,
    )
    wandb.config.update(trainer.cfg)
    # wandb.init(config=trainer.cfg, reinit=True)
    # buffer = test.Buffer(test.legacy_cfg, test.train_tokens, test.model)

    # def train_buffer():
    #     for i in tqdm.tqdm(range(90000 * 20 * 1024 // 2048)):
    #         yield buffer.next()

    ac = ac_cfg.ac
    cfg.use_autocast = False
    trainer.train(ac.read_as_iter(trainer.cfg.buffer_cfg.batch_size, stop_after=32))
    # trainer.train(train_buffer())
    wandb.finish()


sweep_id = open("sweep/sweep_id.txt").read().strip()

SWEEP_NAME = "sweep_from_config_test"


def main():
    wandb.agent(sweep_id, function=run, entity="sae_all", project=SWEEP_NAME)


if __name__ == "__main__":
    main()

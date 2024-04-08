from nqgl.sc_sae.models import test
from nqgl.sc_sae.data import ac_cfg
from nqgl.sc_sae.sweep.swept_config import get_configs_from_sweep, ConfigFromSweep
import tqdm
import wandb


def run():
    wandb.init()
    trainer = test.get_trainer(*get_configs_from_sweep(ConfigFromSweep(**wandb.config)))
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

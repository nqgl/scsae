from nqgl.sc_sae.model import test, ac_cfg
import tqdm
import wandb

SWEEP_NAME = "sweep_test2"


def run():

    wandb.init(project=SWEEP_NAME)
    trainer = test.get_trainer(*test.get_configs(wandb.config))
    wandb.init(project=SWEEP_NAME, config=trainer.cfg, reinit=True)

    ac = ac_cfg.ac
    trainer.train(ac.read_as_iter(trainer.cfg.buffer_cfg.batch_size, stop_after=400))


b1_vals = [0.5, 0.8]
b2_vals = [0.95, 0.98, 0.99, 0.997]

b1_vals = [0.8, 0.9, 0.5, 0.0]
b2_vals = [0.97, 0.99, 0.999]
lr_vals = [1e-3]
cooldown_period = [10_000]
cooldown_factor = [10]
l1_coeffs = [1.5e-3, 2e-3]
sweep_configuration = {
    "method": "grid",
    "parameters": {
        "betas": {"values": [(b1, b2) for b1 in b1_vals for b2 in b2_vals]},
        "lr": {"values": lr_vals},
        "cooldown_period": {"values": cooldown_period},
        "cooldown_factor": {"values": cooldown_factor},
        "l1_coeff": {"values": l1_coeffs},
    },
}
# sweep_id = wandb.sweep(sweep=sweep_configuration, project=SWEEP_NAME, entity="sae_all")
sweep_id = "ezfy5nlu"


print(sweep_id)


def main():
    wandb.agent(sweep_id, function=run, entity="sae_all", project=SWEEP_NAME)


if __name__ == "__main__":
    main()

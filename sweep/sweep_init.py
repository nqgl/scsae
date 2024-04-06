import wandb

b1_vals = [0.7, 0.9, 0.8]
b2_vals = [0.9, 0.97, 0.99, 0.997]

sweep_configuration = {
    "method": "grid",
    "parameters": {
        "betas": {"values": [(b1, b2) for b1 in b1_vals for b2 in b2_vals]},
    },
}
sweep_id = wandb.sweep(
    sweep=sweep_configuration, project="sweep_test", entity="sae_all"
)


print(sweep_id)

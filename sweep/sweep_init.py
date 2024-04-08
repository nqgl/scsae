import wandb
from nqgl.sc_sae.sweep.swept_config import SweepConfig


sc = SweepConfig(
    l1_coeff=[1e-3],
    lr=[1e-3],
    b2=[0.99],
    # sae_type=["VanillaSAE"],
    # sparsity_penalty_type=["l1_sqrt"],
    # k=[1.5e-3, 2e-3],
    b1=[0.9],
)
sc.initialize_sweep()

import wandb
from nqgl.sc_sae.sweep.swept_config import SweepConfig


sc = SweepConfig(
    l1_coeff=[1e-3 * (1.5**i) for i in range(12)],
    lr=[1e-3],
    b2=[0.99],
    b1=[0.8],
)
sc.initialize_sweep()

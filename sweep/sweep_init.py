import wandb
from nqgl.sc_sae.sweep.swept_config import SweepConfig


sc = SweepConfig(
    l1_coeff=[1e-3, 3e-4, 7e-4],
    lr=[1e-3],
    b2=[0.99, 0.999],
    # sae_type=[
    #     "LinearScaleSAE_NonMulGrads",
    #     "LinearScaleSAE_MulGrads",
    # ],
    sparsity_penalty_type=["l1"],
    # k=[1.5e-3, 2e-3],
    b1=[0.9],
)
sc.initialize_sweep()

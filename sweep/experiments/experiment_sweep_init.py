import wandb
from nqgl.sc_sae.sweep.swept_config import SweepConfig


sc = SweepConfig(
    # l1_coeff=[4e-4, 7e-4, 1e-3, 1.4e-3, 2e-3, 3e-3],
    l1_coeff=[3e-4, 48e-4, 6e-4, 12e-4, 24e-4],
    lr=[1e-3],
    b1=[0.9],
    b2=[
        0.999,
    ],
    # sae_type=[
    #     "NoEncSAE",
    #     # "SCSAE_MulGrads",
    # ],
    sparsity_penalty_type=["l1", "l1_sqrt"],
    # k=[1.5e-3, 2e-3],
    sae_type=[
        # "HybridSCSAE_RegGrads",
        # "HybridSCSAE_MulGrads",
        "SCSAE_MulGrads",
        "SCSAE_RegGrads",
        "VanillaSAE",
        # "SqrtSAE",
    ],
    normalizer_type=["L2Normalizer"],
    cooldown_period=[20_000],
    # l0_target=[10, 40],
)
sc.initialize_sweep()

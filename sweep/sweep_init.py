import wandb
from nqgl.sc_sae.sweep.swept_config import SweepConfig


sc = SweepConfig(
    l1_coeff=[3e-4],
    lr=[1e-3],
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
        "SCSAE_MulGrads",
        "SCSAE_RegGrads",
        "VanillaSAE",
        # "SqrtSAE",
        # "HybridSCSAE_RegGrads",
        # "HybridSCSAE_MulGrads",
        # "SCSAE_LogGrad",
        # "SCSAE_SigGate",
    ],
    normalizer_type=["L2Normalizer"],
    b1=[0.9],
    l0_target=[10, 40],
)
sc.initialize_sweep()

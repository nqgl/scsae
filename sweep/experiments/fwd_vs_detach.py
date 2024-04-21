import wandb
from nqgl.sc_sae.sweep.swept_config import SweepConfig


sc = SweepConfig(
    l1_coeff=[1e-3],
    lr=[1e-3],
    b1=[0.9],
    b2=[
        0.999,
    ],
    # sae_type=[
    #     "NoEncSAE",
    #     # "SCSAE_MulGrads",
    # ],
    sparsity_penalty_type=["l1"],
    # k=[1.5e-3, 2e-3],
    sae_type=[
        # "HybridSCSAE_RegGrads",
        # "HybridSCSAE_MulGrads",
        # "SCSAE_MulGrads",
        # "SCSAE_MulEquivalent",
        # "SCSAE_RegGrads",
        # "SCSAE_NonMulEquivalent",
        # "VanillaSAE",
        # "SqrtSAE",
        "HybridSCSAE_RegGrads",
        "HybridSCSAE_MulGrads",
        "SCSAE_LogGrad",
        "SCSAE_SigGate",
    ],
    normalizer_type=["L2Normalizer"],
    # l0_target=[25],
)
sc.initialize_sweep()

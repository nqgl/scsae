from nqgl.sc_sae.models.configs import (
    SAEConfig,
)
from nqgl.sc_sae.models.base_sae import BaseSAE
from nqgl.sc_sae.models.mul_grads import SCSAE
from nqgl.sc_sae.models.vanilla_sae import VanillaSAE
from nqgl.sc_sae.models.non_mul_grads import SCSAERegGrads

# from nqgl.sc_sae.models.fwd_fn_saes import SCSAE_NonMulEquivalent, SCSAE_MulEquivalent
from nqgl.sc_sae.models.bsc import (
    SCSAE_MulMk2,
    SCSAE_2Bias,
    InterpolatedStartSCSAE,
    SpoofTestSAE,
    NoEncSAE,
    SCSAE_LogGrad,
    SCSAE_SigGate,
)


from nqgl.sc_sae.models.hybrid import HybridSCSAEMulGrads, HybridSCSAERegGrads

MODELS_LIST = [
    SCSAE,
    VanillaSAE,
    SCSAERegGrads,
    SCSAE_MulMk2,
    SCSAE_2Bias,
    InterpolatedStartSCSAE,
    SpoofTestSAE,
    NoEncSAE,
    HybridSCSAEMulGrads,
    HybridSCSAERegGrads,
    SCSAE_LogGrad,
    SCSAE_SigGate,
]

assert not any(
    [
        MODELS_LIST[i].MODEL_TYPE in [m.MODEL_TYPE for m in MODELS_LIST[i + 1 :]]
        for i in range(len(MODELS_LIST))
    ]
)

MODEL_CLASSES = {cls.MODEL_TYPE: cls for cls in MODELS_LIST}

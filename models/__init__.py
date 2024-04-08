from nqgl.sc_sae.models.configs import (
    SAEConfig,
)
from nqgl.sc_sae.models.base_sae import BaseSAE
from nqgl.sc_sae.models.mul_grads import LinearScaleSAE
from nqgl.sc_sae.models.vanilla_sae import VanillaSAE
from nqgl.sc_sae.models.non_mul_grads import LinearScaleSAENonMulGrads

MODELS_LIST = [LinearScaleSAE, VanillaSAE, LinearScaleSAENonMulGrads]

assert not any(
    [
        MODELS_LIST[i].MODEL_TYPE in [m.MODEL_TYPE for m in MODELS_LIST[i + 1 :]]
        for i in range(len(MODELS_LIST))
    ]
)

MODEL_CLASSES = {cls.MODEL_TYPE: cls for cls in MODELS_LIST}

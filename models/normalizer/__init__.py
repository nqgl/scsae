from nqgl.sc_sae.models.normalizer.base import L2NormalizerMixin, ConstL2NormalizerMixin


NORMALIZERS_LIST = [L2NormalizerMixin, ConstL2NormalizerMixin]

assert not any(
    [
        NORMALIZERS_LIST[i].NORMALIZER_TYPE
        in [m.NORMALIZER_TYPE for m in NORMALIZERS_LIST[i + 1 :]]
        for i in range(len(NORMALIZERS_LIST))
    ]
)

NORMALIZER_CLASSES = {cls.NORMALIZER_TYPE: cls for cls in NORMALIZERS_LIST}

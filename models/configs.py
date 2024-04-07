from dataclasses import dataclass
import torch


@dataclass
class SAEConfig:
    d_data: int = 768
    dict_mult: int = 4
    dtype: str = None
    device: str = "cuda"
    l1_coeff: float = 1e-3
    tied_init: bool = True
    use_b_dec: bool = True
    selectively_norm_dec: bool = False
    d_dict: int = None
    sae_type: str = "LinearScaleSAE_MulGrads"

    def __post_init__(self):
        if self.d_dict is None:
            self.d_dict = self.d_data * self.dict_mult

    def get_cls(self):
        from nqgl.sc_sae.models import MODEL_CLASSES

        return MODEL_CLASSES[self.sae_type]
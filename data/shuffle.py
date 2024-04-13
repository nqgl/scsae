import torch
from torch import Tensor
from nqgl.sc_sae.data.tabletensor import Piler
from nqgl.sc_sae.from_hsae_re.buffer2_no_cast import BufferConfig
from transformer_lens import HookedTransformer
from nqgl.sc_sae.data.locations import DATA_DIRS

from nqgl.sc_sae.data.dataset import DataConfig


def acts_generator(batch_size, d_data):
    for i in range(1024):
        yield torch.randn(batch_size, d_data)


def make_train_acts(
    token_seqs: Tensor,
    model: HookedTransformer,
    dcfg: DataConfig,
    bcfg: BufferConfig,
    max_gb_per_pile=2,
    seq_len=128,
    dtype=torch.float16,
    dtype_bytes=2,
):
    docs_per_pile = max_gb_per_pile * 2**30 // (seq_len * bc.d_data * dtype_bytes)
    num_docs = token_seqs.shape[0]
    num_piles = (num_docs + docs_per_pile - 1) // docs_per_pile
    piler = Piler(
        f"piles_{bc.site}_{bc.layer}",
        dtype=dtype,
        fixed_shape=[seq_len, bc.d_data],
        num_piles=num_piles,
    )

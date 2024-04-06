# %%

import os
import sys
import torch
import plotly.express as px
import gc
from IPython.display import HTML
from transformer_lens import HookedTransformer, utils

from nqgl.mlutils.components import Cache

from nqgl.mlutils.optimizations.norepr import fastpartial
from transformer_lens.utils import get_act_name

import tqdm

import pandas as pd

from transformer_lens import utils

import numpy as np

import gc

from matplotlib import pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from gate_through_nonlinearity import get_trainer

# %%
from training.cl_on_data import (
    get_recons_loss,
    val_tokens,
    legacy_cfg,
    model,
    sae_cfg,
    get_buffer,
)
from data.buffer2 import Buffer, BufferConfig

new_model = model.to(torch.device("cuda"), torch.float16)
del model
model = new_model

trainer = get_trainer(sae_cfg)
sae = trainer.sae

# %%
import torch

# state_dict = torch.load("../models-from-remote/floral-bee-876_420000.pt")
# state_dict = torch.load("../models-from-remote/rare-paper-818_250000.pt")
state_dict = torch.load("../models-from-remote/misunderstood-totem-808_660000.pt")
#
# %%
state_dict.keys()

# %%
sae = trainer.sae
sae.load_state_dict(state_dict)
sae = sae.eval()


# %%


rlosses = get_recons_loss(
    model,
    sae,
    buffer=None,
    all_tokens=val_tokens,
    cfg=legacy_cfg,
    bos_processed_with_hook=True,
).items()


rlosses
# %%

# %%
W_enc = sae.cachelayer.encoder.cachelayer.W
b_enc = sae.cachelayer.encoder.cachelayer.b
W_dec = sae.cachelayer.decoder.weight.transpose(-2, -1)
b_dec = sae.cachelayer.b_dec

# %%
from simple_reimpl.model import BiasAdjustedSAE, SAEConfig

SAEConfig = SAEConfig(d_data=768, d_dict=768 * 32)
bsae = BiasAdjustedSAE(SAEConfig)
bsae.W_enc.data[:] = W_enc
bsae.b_enc.data[:] = b_enc
bsae.W_dec.data[:] = W_dec
bsae.b_dec.data[:] = b_dec


# %%


class HookedBox:
    def __init__(self):
        self.x = None

    def __lshift__(self, acts):
        l0 = acts.count_nonzero(dim=-1).float().mean().item()
        print(acts.shape)
        print("l0", l0)
        self.x = acts


from unpythonic import box

bsae = bsae.to(torch.device("cuda"))
brlosses = get_recons_loss(
    model,
    lambda x: bsae(x, HookedBox(), box()),
    buffer=None,
    all_tokens=val_tokens,
    cfg=legacy_cfg,
    bos_processed_with_hook=True,
).items()


rlosses
# %%

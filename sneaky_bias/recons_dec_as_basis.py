# %%
import os
import sys
import torch
import plotly.express as px
import gc
from IPython.display import HTML
from transformer_lens import HookedTransformer, utils

from nqgl.mlutils.components import Cache

from nqgl.mlutils.optimizations.norepr import fastpartial as partial
from transformer_lens.utils import get_act_name

import tqdm

import pandas as pd

from transformer_lens import utils

import numpy as np

import gc

from matplotlib import pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
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

# %%
import torch

state_dict = torch.load("../models-from-remote/floral-bee-876_420000.pt")
# state_dict = torch.load("../models-from-remote/rare-paper-818_250000.pt")
# state_dict = torch.load("../models-from-remote/misunderstood-totem-808_660000.pt")

# %%
state_dict.keys()
# %%
sae = trainer.sae
sae.load_state_dict(state_dict)
sae = sae.eval()

# %%
# inital using normal recons to validate loaded model

rlosses = get_recons_loss(
    model,
    sae,
    buffer=None,
    all_tokens=val_tokens,
    cfg=legacy_cfg,
).items()

rlosses
# %%


@torch.no_grad()
def get_recons_loss(
    model,
    encoder,
    buffer,
    all_tokens=None,
    num_batches=5,
    local_encoder=None,
    cfg=None,
    bos_processed_with_hook=False,
):
    cfg = cfg or encoder.cfg
    if local_encoder is None:
        local_encoder = encoder
    loss_list = []
    for i in range(num_batches):
        tokens = (all_tokens if all_tokens is not None else buffer.all_tokens)[
            torch.randperm(
                len(all_tokens if all_tokens is not None else buffer.all_tokens)
            )[: max(cfg.model_batch_size // 16, 1)]
        ]
        # assert torch.all(50256 == tokens[:, 0])
        loss = model(tokens, return_type="loss")
        recons_loss = model.run_with_hooks(
            tokens,
            return_type="loss",
            fwd_hooks=[
                (
                    cfg.act_name,
                    partial(
                        replacement_hook,
                        encoder=local_encoder,
                        cfg=cfg,
                        bos_processed_with_hook=bos_processed_with_hook,
                    ),
                )
            ],
        )
        # mean_abl_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(cfg.act_name, mean_ablate_hook)])
        zero_abl_loss = model.run_with_hooks(
            tokens,
            return_type="loss",
            fwd_hooks=[(cfg.act_name, zero_ablate_hook)],
        )
        loss_list.append((loss, recons_loss, zero_abl_loss))
    losses = torch.tensor(loss_list)
    loss, recons_loss, zero_abl_loss = losses.mean(0).tolist()

    print(loss, recons_loss, zero_abl_loss)
    score = (zero_abl_loss - recons_loss) / (zero_abl_loss - loss)
    print(f"{score:.2%}")
    # print(f"{((zero_abl_loss - mean_abl_loss)/(zero_abl_loss - loss)).item():.2%}")
    return {
        "recons_score": score,
        "loss": loss,
        "recons_loss": recons_loss,
        "zero_ablation_loss": zero_abl_loss,
    }


l = []


def replacement_hook(acts, hook, encoder, cfg, bos_processed_with_hook=False):
    acts_shape = acts.shape
    acts_re = acts.reshape(-1, cfg.act_size)
    cache = Cache()
    cache.acts = ...
    mlp_post_reconstr = encoder(acts_re.reshape(-1, cfg.act_size), cache=cache)

    sae.cachelayer.decoder.weight
    l.append([acts, mlp_post_reconstr, cache["encoder"][0].acts])

    mlp_post_reconstr = mlp_post_reconstr.reshape(acts_shape)
    seq_len = acts_shape[1]
    assert seq_len == 128
    if bos_processed_with_hook:
        return mlp_post_reconstr
    return torch.cat((acts[:, :1], mlp_post_reconstr[:, 1:]), dim=1)


def mean_ablate_hook(mlp_post, hook):
    mlp_post[:, :] = mlp_post.mean([0, 1])
    return mlp_post


def zero_ablate_hook(mlp_post, hook):
    mlp_post[:, :] = 0.0
    return mlp_post


rlosses = get_recons_loss(
    model,
    sae,
    buffer=None,
    all_tokens=val_tokens,
    cfg=legacy_cfg,
).items()
in_acts, mlp_post_reconstr, feat_acts = l[0]
# %%

sae.cachelayer.decoder.weight
# %%

feat_acts.shape
# %%
acts = in_acts.view(-1, in_acts.shape[-1])
# %%
acts.shape
# %%
W_dec = sae.cachelayer.decoder.weight
with torch.inference_mode():
    v = acts @ W_dec


# %%
with torch.inference_mode():

    r = (feat_acts[0:1] > 0) * (v[0:1] * W_dec)
    s = r.sum(-1)
    err = acts[0] - s
    sae_re = sae(acts[0:1])
    sae_err = acts[0] - sae_re[0]

# %%
res_err = sae_err
reconstructed = sae_re
coeff = 0.5
with torch.inference_mode():
    for i in range(100):
        v2 = res_err.unsqueeze(0) @ W_dec
        err_feat_mags = v2[feat_acts[0:1] > 0]
        r = (feat_acts[0:1] > 0) * (v[0:1] * W_dec)
        s = r.sum(-1)
        err = acts[0] - s

# %%

selected_directions = W_dec[:, feat_acts[0] > 0]
cosims = selected_directions @ selected_directions.T
cosims[torch.arange(cosims.shape[0]), torch.arange(cosims.shape[0])] = 0
(cosims < -0.1).count_nonzero()
# %%
v2[feat_acts[0:1] > 0]

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


rlosses = get_recons_loss(
    model,
    sae,
    buffer=None,
    all_tokens=val_tokens,
    cfg=legacy_cfg,
).items()


rlosses
# %%
buf_cfg = BufferConfig(
    layer=legacy_cfg.layer,
    site=legacy_cfg.site,
    flatten_heads=legacy_cfg.flatten_heads,
    device=legacy_cfg.device,
    d_data=legacy_cfg.d_data,
    batch_size=512,
    buffer_mult=64,
    buffer_refresh_ratio=legacy_cfg.buffer_refresh_ratio,
    buffer_dtype=legacy_cfg.buffer_dtype,
    buffer_autocast_dtype="fp16",
    excl_first=False,
)

buffer = Buffer(buf_cfg, val_tokens, model=model)
# %%


# %%
def get_acts(tokens):
    acts = torch.zeros((*tokens.shape, 768))
    batch_size = 16
    with torch.inference_mode():

        site = get_act_name(legacy_cfg.site, legacy_cfg.layer)
        for i in tqdm.tqdm(range(0, len(tokens), batch_size)):
            target = acts[i : i + batch_size]
            tokens_batch = tokens[i : i + batch_size]

            def extraction_hook(acts, hook):
                target[:] = acts

            l = model.run_with_hooks(
                tokens_batch,
                fwd_hooks=[(site, extraction_hook)],
            )
            del l
    assert not (acts == 0).all(dim=-1).any()
    return acts


def get_feat_acts(acts):

    batch_size = 16
    feat_acts = torch.zeros(
        (*acts.shape[:-1], sae_cfg.d_dict), dtype=torch.float16, device="cpu"
    )

    with torch.inference_mode():
        for i in tqdm.tqdm(range(0, len(acts), batch_size)):
            cache = Cache()
            cache.acts = ...
            acts_batch = acts[i : i + batch_size]
            abshape = acts_batch.shape
            acts_batch = acts_batch.view(-1, acts_batch.shape[-1])
            sae(acts_batch.cuda(), cache=cache)
            feat_acts[i : i + batch_size] = (
                cache["encoder"][0]
                .acts.to(
                    torch.device("cpu"),
                    torch.float16,
                )
                .reshape(abshape[:-1] + (sae_cfg.d_dict,))
            )
            del cache
            gc.collect()
    assert not (feat_acts == 0).all(dim=-1).any()
    return feat_acts


# def get_acts_feats(tokens):


some_tokens = val_tokens[torch.randperm(len(val_tokens))[: 16 * 64]]
acts = get_acts(some_tokens)
feat_acts = get_feat_acts(acts)


# %%


def make_df(tokens, feat_acts, len_prefix=15, len_suffix=3):
    str_tokens = [model.to_str_tokens(t) for t in tokens]
    unique_token = [
        [f"{s}/{i}" for i, s in enumerate(str_tok)] for str_tok in str_tokens
    ]

    context = []
    batch = []
    pos = []
    label = []
    for b in range(tokens.shape[0]):
        # context.append([])
        # batch.append([])
        # pos.append([])
        # label.append([])
        for p in range(tokens.shape[1]):
            prefix = "".join(str_tokens[b][max(0, p - len_prefix) : p])
            if p == tokens.shape[1] - 1:
                suffix = ""
            else:
                suffix = "".join(
                    str_tokens[b][p + 1 : min(tokens.shape[1] - 1, p + 1 + len_suffix)]
                )
            current = str_tokens[b][p]
            context.append(f"{prefix}|{current}|{suffix}")
            batch.append(b)
            pos.append(p)
            label.append(f"{b}/{p}")

    # select all where feature > 50th percentile/usr/local/lib/python3.8/dist-packages/pandas/core/common.py

    # print(len(batch), len(pos), len(context), len(label))
    df = pd.DataFrame(
        dict(
            str_tokens=[x for y in str_tokens for x in y],
            unique_token=[x for y in unique_token for x in y],
            context=context,
            batch=batch,
            pos=pos,
            label=label,
        )
    )
    df["feat_acts"] = [x for y in utils.to_numpy(feat_acts) for x in y]
    return df


df = make_df(some_tokens, feat_acts)


# %%


def show_random_highly_activating(df, feat_idx, percentile=50, num=15):
    # select all where feature > 0.0
    # get the 50th percentile
    feat_key = f"feat_{feat_idx}"
    feat_key = f"feature_activation"

    df[feat_key] = df["feat_acts"].apply(lambda x: x[feat_idx])
    selected = df[df[feat_key] > 0]
    percentile = np.percentile(selected[feat_key], percentile)
    display(
        selected[selected[feat_key] > percentile].sample(num)
        # .style.background_gradient(cmap=colormaps["viridis"])
        .style.background_gradient("PuBu")
    )


show_random_highly_activating(df, 111)


# %%
# unbalanced closed paren feature
show_random_highly_activating(df, 90)
# %%
# partially in the eos direction but low threshold feature
show_random_highly_activating(df, 23854)


# %%

gc.collect()
torch.cuda.empty_cache()
# %%
freq_tracker = sae.cachelayer.encoder.components[0]

# %%


sae.cachelayer.encoder.components[0].cfg.decay = 0.999
# %%
sae.train()

with torch.inference_mode():
    for i in tqdm.tqdm(range(1000)):
        acts = buffer.next()
        sae(acts)
sae.eval()
# %%
sae.cachelayer.encoder.components[0].freqs
# %%


# histogram of freqs
freqs = freq_tracker.freqs
# chop off high freqs
chopped_freqs = freqs[freqs < 0.001]
plt.hist(chopped_freqs.cpu().numpy(), bins=1000)


# %%
# %%
sae.cachelayer.encoder.cachelayer.W.shape
# %%
enc_W = sae.cachelayer.encoder.cachelayer.W
enc_W.norm(dim=0).median()
dec_W = sae.cachelayer.decoder.weight
# %%

with torch.inference_mode():
    enc_W_normed = enc_W / enc_W.norm(dim=0, keepdim=True)
    dec_W_normed = dec_W / dec_W.norm(dim=0, keepdim=True)

    cosims = (enc_W_normed * dec_W_normed).sum(0)
    plt.hist(cosims.cpu().numpy(), bins=1000)
# %%
1 / cosims.median()
# %%
1 / cosims.mean()
# %%
with torch.inference_mode():
    cosims = (enc_W * dec_W).sum(0)
    plt.hist(cosims.cpu().numpy(), bins=1000)

# %%
with torch.inference_mode():
    thresh = 0.00001
    a = (freqs > thresh).count_nonzero()
    mask = freqs > thresh
    cosims = (enc_W[:, mask] * dec_W[:, mask]).sum(0)
    plt.hist(cosims.cpu().numpy(), bins=1000)

a
# %%
with torch.inference_mode():
    bos_first_act = acts[0, 0].cpu()
    bos_first_act_normed = bos_first_act / bos_first_act.norm()
    dot_w_bos_dir = (bos_first_act_normed.unsqueeze(-1) * enc_W_normed.cpu()).sum(0)
    plt.hist(dot_w_bos_dir.cpu().numpy(), bins=1000)

#
#
#
#

# %%
# Checking the "empirically pathological" thing


from typing import Callable


def subs_to_hook(subs: Callable) -> Callable:
    def hook(acts, hook):
        acts[:] = subs(acts)

    return hook


# def substitute_hook(acts, hook):
#     acts[:] = sae(acts)


def sae_subs(acts):
    return sae(acts)


sae_subs_hook = subs_to_hook(sae_subs)


def eps_scaled_dir(noise_dir_gen=torch.randn_like) -> Callable:
    def eps_scaled(acts):
        eps = (acts - sae(acts)).norm(dim=-1, keepdim=True)
        noise_dir = noise_dir_gen(acts)
        eps_noise = noise_dir / noise_dir.norm(dim=-1, keepdim=True) * eps
        return acts + eps_noise

    return eps_scaled


eps_subs_uniform = eps_scaled_dir(lambda acts: torch.rand_like(acts) - 0.5)
eps_subs_normal = eps_scaled_dir(torch.randn_like)
eps_randn_hook = subs_to_hook(eps_subs_normal)


def compare_losses(tokens, *subs_l, to_hook=subs_to_hook, model=model):
    site = get_act_name(legacy_cfg.site, legacy_cfg.layer)
    loss = model(
        tokens,
        return_type="loss",
    )

    return loss, *[
        model.run_with_hooks(
            tokens,
            return_type="loss",
            fwd_hooks=[(site, to_hook(subs))],
        )
        for subs in subs_l
    ]


@torch.inference_mode()
def gen_losses_collection(
    n=256,
    sae_subs=sae_subs,
    eps_subs=eps_subs_normal,
    to_hook=subs_to_hook,
    batch_size=4,
):
    losses, sae_subst_losses, eps_subs_losses = [], [], []
    sae_subst_diffs, eps_subs_diffs = [], []

    for i in tqdm.tqdm(range(0, n, batch_size)):
        loss, loss_sae_subst, loss_eps_subs = compare_losses(
            val_tokens[i : i + batch_size], sae_subs, eps_subs, to_hook=to_hook
        )
        losses.append(loss.item())
        sae_subst_diffs.append(loss_sae_subst.item() - loss.item())
        eps_subs_diffs.append(loss_eps_subs.item() - loss.item())
        sae_subst_losses.append(loss_sae_subst.item())
        eps_subs_losses.append(loss_eps_subs.item())

    # plt.plot(losses, label="loss")
    plt.plot(sae_subst_diffs, label="sae_subs CE loss diff")
    plt.plot(eps_subs_diffs, label="eps_subs CE loss diff")
    plt.legend()
    s, e = (
        # torch.tensor(losses).mean().item(),
        torch.tensor(sae_subst_losses).mean().item(),
        torch.tensor(eps_subs_losses).mean().item(),
    )
    print(f"SAE substitution: {s:.2f}, \neps substitution: {e:.2f}")
    return sae_subst_losses, eps_subs_losses
    # plt.show()


# %%
sae_subst_losses, eps_subs_losses = gen_losses_collection()
# %%
model.to_str_tokens(val_tokens[2])
# %%
lt, st, et = (
    torch.tensor(losses),
    torch.tensor(sae_subst_losses),
    torch.tensor(eps_subs_losses),
)
(lt < st) == (et < st)
# %%
lt.shape
lc = torch.stack([lt, st, et], dim=1)

# %%
lc.mean(dim=0)
# %%
lc.median(dim=0)
# %%
lc.std(dim=0)
# %%


def subs_to_excl_first_hook(subs):
    def hook(acts, hook):
        acts[:, 1:] = subs(acts)[:, 1:]

    return hook


t = gen_losses_collection(n=1024, to_hook=subs_to_excl_first_hook)


# %%


def sae_dec_directions(acts):
    idxs = torch.randint(0, dec_W.shape[1], acts.shape[:-1]).view(-1)
    dirs = dec_W[:, idxs].transpose(-2, -1).view(*acts.shape[:-1], -1)
    return dirs


t = gen_losses_collection(
    eps_subs=eps_scaled_dir(sae_dec_directions), to_hook=subs_to_excl_first_hook
)

# %%


def orthogonal_directions(acts):
    rand = torch.randn_like(acts)
    acts_normed = acts / acts.norm(dim=-1, keepdim=True)
    return rand - (acts_normed * rand).sum(-1, keepdim=True) * acts_normed


# %%


def active_feature_directions(acts):
    cache = Cache()
    cache.acts = ...
    sae(acts, cache=cache)
    feat_acts = cache["encoder"][0].acts.reshape(-1, sae_cfg.d_dict)
    selected = torch.multinomial(feat_acts, 1).squeeze(-1)
    dirs = dec_W[:, selected].transpose(-2, -1).view(*acts.shape[:-1], -1)
    return dirs


t = gen_losses_collection(
    eps_subs=eps_scaled_dir(active_feature_directions), to_hook=subs_to_excl_first_hook
)

# %%


def random_in_active_feat_dirs(acts):
    cache = Cache()
    cache.acts = ...
    sae(acts, cache=cache)
    feat_acts = cache["encoder"][0].acts.reshape(-1, sae_cfg.d_dict)
    rand_like_feats = -torch.rand_like(feat_acts) * (feat_acts > 0)
    # print(feat_acts.shape)
    selected = torch.multinomial(feat_acts, 1).squeeze(-1)
    v = rand_like_feats @ dec_W.transpose(-2, -1)
    # print(v.shape)
    return v.view(*acts.shape)


t = gen_losses_collection(
    eps_subs=eps_scaled_dir(random_in_active_feat_dirs), to_hook=subs_to_excl_first_hook
)
# %%

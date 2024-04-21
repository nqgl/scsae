from nqgl.sc_sae.models import SCSAE, SAEConfig
from nqgl.sc_sae.data.dataset import DataConfig
from nqgl.sc_sae.trainer import Trainer, OptimConfig, LrSchedulerConfig, SAETrainConfig
from nqgl.sc_sae.from_hsae_re.buffer2_no_cast import Buffer, BufferConfig
import tqdm
from nqgl.sae.scripts.train_hsae import load_data, HierarchicalAutoEncoderConfig

# from nqgl.hsae_re.data.buffer2 import Buffer, BufferConfig

from nqgl.sae.training.buffer import Buffer
from transformer_lens import HookedTransformer
from typing import Tuple


def get_configs(d={}) -> Tuple[SAETrainConfig, HierarchicalAutoEncoderConfig]:
    dict_mult = d.get("dict_mult", 16)

    sae_cfg = SAEConfig(
        dict_mult=dict_mult,
    )

    buf_cfg = BufferConfig(
        layer=6,
        site="resid_pre",
        flatten_heads=False,
        device="cuda",
        d_data=sae_cfg.d_data,
        batch_size=4096,
        buffer_mult=512,
        buffer_refresh_ratio=0.5,
        buffer_dtype="fp16",
        buffer_autocast_dtype=DATA_DTYPE,
        excl_first=False,
    )

    legacy_cfg = HierarchicalAutoEncoderConfig(
        site="resid_pre",
        d_data=768,
        model_name="gpt2",
        layer=buf_cfg.layer,
        gram_shmidt_trail=512,
        batch_size=buf_cfg.batch_size,
        buffer_mult=buf_cfg.buffer_mult,
        buffer_refresh_ratio=0.5,
        flatten_heads=False,
        buffer_dtype=DATA_DTYPE,
        enc_dtype=DTYPE,
        device=device,
    )

    betas = d.get("betas", (0.9, 0.999))
    cfg = SAETrainConfig(
        sae_cfg=sae_cfg,
        optim_cfg=OptimConfig(lr=d.get("lr", 1e-3), betas=betas),
        data_cfg=DataConfig(),
        use_autocast=True,
        l1_coeff=d.get("l1_coeff", 1.5e-3),
        buffer_cfg=buf_cfg,
        lr_scheduler_cfg=LrSchedulerConfig(
            warmup_steps=3_00,
            cooldown_begin=40_000,
            cooldown_period=d.get("cooldown_period", 5_000),
            cooldown_factor=d.get("cooldown_factor", 10),
        ),
    )
    return cfg, legacy_cfg


DATA_DTYPE = "fp32"
DTYPE = "fp32"
device = "cuda"


import torch

cfg, legacy_cfg = get_configs()
model = (
    HookedTransformer.from_pretrained(cfg.data_cfg.model_name)
    # .to(torch.float16)
    .to(device)
)
train_percent = 5
train_start = 5
train_tokens = load_data(
    model,
    dataset=cfg.data_cfg.dataset,
    split=f"train[{train_start}%:{train_start+train_percent}%]",
    name=cfg.data_cfg.model_name,
    front_only=False,
    seq_len=128,
    seq_mul=cfg.data_cfg.seq_mul,
    set_bos=cfg.data_cfg.set_bos,
)  # .cuda()


val_tokens = load_data(
    model,
    dataset=cfg.data_cfg.dataset,
    name=cfg.data_cfg.model_name,
    split=f"train[90%:95%]",
    front_only=False,
    seq_len=128,
    seq_mul=cfg.data_cfg.seq_mul,
    set_bos=cfg.data_cfg.set_bos,
)


def get_trainer(cfg, legacy_cfg) -> Trainer:
    return Trainer(cfg, model=model, val_tokens=val_tokens, legacy_cfg=legacy_cfg)


def main():
    # buffer = Buffer(cfg.buffer_cfg, train_tokens, model)
    buffer = Buffer(legacy_cfg, train_tokens, model)

    def train_buffer():
        for i in tqdm.tqdm(range(90000 * 20 * 1024 // 2048)):
            yield buffer.next()

    trainer = get_trainer(cfg, legacy_cfg)
    trainer.train(train_buffer())


if __name__ == "__main__":
    main()

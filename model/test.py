from nqgl.sc_sae.model import (
    Trainer,
    SAETrainConfig,
    SAEConfig,
    BiasAdjustedSAE,
    OptimConfig,
    DataConfig,
)
from nqgl.hsae_re.data.buffer2_no_cast import Buffer, BufferConfig
import tqdm
from nqgl.sae.scripts.train_hsae import (
    load_data,
    HierarchicalAutoEncoderConfig,
)

from transformer_lens import HookedTransformer

DATA_DTYPE = "fp32"
DTYPE = "fp32"
device = "cuda"


def get_configs(d={}):
    dict_mult = d.get("dict_mult", 32)

    sae_cfg = SAEConfig(
        dict_mult=dict_mult,
    )

    buf_cfg = BufferConfig(
        layer=6,
        site="resid_pre",
        flatten_heads=False,
        device="cuda",
        d_data=sae_cfg.d_data,
        batch_size=1024,
        buffer_mult=2048,
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
        batch_size=1024,
        buffer_mult=2048,
        buffer_refresh_ratio=0.5,
        flatten_heads=False,
        buffer_dtype=DATA_DTYPE,
        enc_dtype=DTYPE,
        device=device,
    )

    betas = d.get("betas", (0.8, 0.98))
    cfg = SAETrainConfig(
        sae_cfg=sae_cfg,
        optim_cfg=OptimConfig(lr=1e-3, betas=betas),
        data_cfg=DataConfig(),
        use_autocast=True,
        l1_coeff=d.get("l1_coeff", 1 / 768),
        buffer_cfg=buf_cfg,
    )
    return cfg, legacy_cfg


import torch

cfg, legacy_cfg = get_configs()
model = (
    HookedTransformer.from_pretrained(cfg.data_cfg.model_name)
    .to(torch.float16)
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
    buffer = Buffer(cfg.buffer_cfg, train_tokens, model)

    def train_buffer():
        for i in tqdm.tqdm(range(90000 * 20 * 1024 // 2048)):
            yield buffer.next()

    trainer = get_trainer(cfg, legacy_cfg)
    trainer.train(train_buffer())


if __name__ == "__main__":
    main()

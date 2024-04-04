from nqgl.hsae_re.sneaky_bias.simple_reimpl import (
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

dict_mult = 16
cfg = SAETrainConfig(
    sae_cfg=SAEConfig(
        d_dict=768 * dict_mult,
    ),
    optim_cfg=OptimConfig(lr=3e-4, betas=(0.8, 0.98)),
    data_cfg=DataConfig(),
    use_autocast=True,
    lr_schedule=True,
    l1_coeff=1 / 768,
)
DATA_DTYPE = "fp32"
DTYPE = "fp32"
device = "cuda"

legacy_cfg = HierarchicalAutoEncoderConfig(
    site="resid_pre",
    d_data=768,
    model_name="gpt2",
    layer=6,
    gram_shmidt_trail=512,
    batch_size=1024,
    buffer_mult=2048,
    buffer_refresh_ratio=0.5,
    flatten_heads=False,
    buffer_dtype=DATA_DTYPE,
    enc_dtype=DTYPE,
    device=device,
)

buf_cfg = BufferConfig(
    layer=legacy_cfg.layer,
    site=legacy_cfg.site,
    flatten_heads=legacy_cfg.flatten_heads,
    device=legacy_cfg.device,
    d_data=legacy_cfg.d_data,
    batch_size=legacy_cfg.batch_size,
    buffer_mult=legacy_cfg.buffer_mult,
    buffer_refresh_ratio=legacy_cfg.buffer_refresh_ratio,
    buffer_dtype="fp16",
    buffer_autocast_dtype=DATA_DTYPE,
    excl_first=cfg.data_cfg.excl_first,
)


# Data
from typing import Tuple
import torch

model = (
    HookedTransformer.from_pretrained(legacy_cfg.model_name).to(device)
    # .to(torch.float16)
)
train_percent = 5
train_start = 5
train_tokens = load_data(
    model,
    dataset=cfg.data_cfg.dataset,
    split=f"train[{train_start}%:{train_start+train_percent}%]",
    name=legacy_cfg.model_name,
    front_only=False,
    seq_len=128,
    seq_mul=cfg.data_cfg.seq_mul,
    set_bos=cfg.data_cfg.set_bos,
)  # .cuda()


val_tokens = load_data(
    model,
    dataset=cfg.data_cfg.dataset,
    name=legacy_cfg.model_name,
    split=f"train[90%:95%]",
    front_only=False,
    seq_len=128,
    seq_mul=cfg.data_cfg.seq_mul,
    set_bos=cfg.data_cfg.set_bos,
)  # .cuda()


trainer = Trainer(cfg, model=model, val_tokens=val_tokens, legacy_cfg=legacy_cfg)


def main():
    buffer = Buffer(buf_cfg, train_tokens, model)

    def train_buffer():
        for i in tqdm.tqdm(range(90000 * 20 * 1024 // 2048)):
            yield buffer.next()

    trainer.train(train_buffer())


if __name__ == "__main__":
    main()

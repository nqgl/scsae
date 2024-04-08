from models.test import DATA_DTYPE, DTYPE, device
from nqgl.sc_sae.from_hsae_re.buffer2_no_cast import BufferConfig
from nqgl.sae.scripts.train_hsae import HierarchicalAutoEncoderConfig
from nqgl.sc_sae.models import DataConfig, LrSchedulerConfig, SAEConfig
from nqgl.sc_sae.trainer import OptimConfig, SAETrainConfig


def get_configs(d={}):
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
        batch_size=2048,
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
        batch_size=buf_cfg.batch_size,
        buffer_refresh_ratio=0.5,
        flatten_heads=False,
        buffer_dtype=DATA_DTYPE,
        enc_dtype=DTYPE,
        device=device,
    )

    betas = d.get("betas", (0.9, 0.99))
    cfg = SAETrainConfig(
        sae_cfg=sae_cfg,
        optim_cfg=OptimConfig(lr=d.get("lr", 1e-3), betas=betas),
        data_cfg=DataConfig(),
        use_autocast=True,
        l1_coeff=d.get("l1_coeff", 1.5e-3),
        buffer_cfg=buf_cfg,
        lr_scheduler_cfg=LrSchedulerConfig(
            warmup_steps=3_00,
            cooldown_begin=50_000,
            cooldown_period=d.get("cooldown_period", 4_000),
            cooldown_factor=d.get("cooldown_factor", 10),
        ),
    )
    return cfg, legacy_cfg

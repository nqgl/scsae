from nqgl.sc_sae.models import SAEConfig
from nqgl.sc_sae.data.dataset import DataConfig, SplitConfig
from nqgl.sc_sae.trainer import SAETrainConfig, OptimConfig, LrSchedulerConfig
import tqdm
from nqgl.sc_sae.from_hsae_re.buffer2_no_cast import Buffer, BufferConfig
from nqgl.sae.scripts.train_hsae import HierarchicalAutoEncoderConfig

import wandb
from dataclasses import dataclass, asdict
from typing import List, Tuple


@dataclass
class SweepConfig:
    sweep_project: str = "scsae_comparisons"
    sae_type: Tuple[str] = (
        "SCSAE_MulGrads",
        "SCSAE_RegGrads",
        "VanillaSAE",
    )
    normalizer_type: Tuple[str] = ("ConstL2Normalizer", "L2Normalizer")
    lr: Tuple[float] = (1e-3,)
    b1: Tuple[float] = 0.9
    b2: Tuple[float] = 0.999
    cooldown_period: Tuple[int] = (5_000,)
    cooldown_factor: Tuple[int] = (10,)
    l1_coeff: Tuple[float] = (1.5e-3, 2e-3)
    sparsity_penalty_type: Tuple[str] = ("l1", "l1_sqrt")
    dict_mult: Tuple[int] = (16,)
    l0_target: Tuple[float] = (None,)

    def to_sweep_config(self, method="grid"):
        d = asdict(self)

        sd_b = {"betas": {"values": [(b1, b2) for b1 in self.b1 for b2 in self.b2]}}
        d.pop("b1")
        d.pop("sweep_project")
        d.pop("b2")
        sd = {k: {"values": v} for k, v in d.items()}
        return {
            "method": method,
            "parameters": {**sd, **sd_b},
        }

    def initialize_sweep(self, save=True, method="grid"):
        sweep_id = wandb.sweep(
            sweep=self.to_sweep_config(method=method),
            project=self.sweep_project,
            entity="sae_all",
        )
        if save:
            f = open("sweep/sweep_id.txt", "w")
            f.write(sweep_id)
            f.close()

    def combo_count(self):
        d = asdict(self)
        combos = 0
        for k, v in d.items():
            combos *= len(v)
        return combos


@dataclass
class ConfigFromSweep:
    sae_type: str
    lr: float
    betas: Tuple[float, float]
    cooldown_period: int
    cooldown_factor: int
    l1_coeff: float
    sparsity_penalty_type: str
    normalizer_type: str
    l0_target: int
    dict_mult: int = 16

    # or b1, b2 and property(betas)


def set_configs_from_sweep(
    cfg: SAETrainConfig,
    legacy_cfg: HierarchicalAutoEncoderConfig,
    scfg: ConfigFromSweep = None,
):
    scfg = scfg or ConfigFromSweep(**wandb.config)

    cfg.sae_cfg.sae_type = scfg.sae_type
    cfg.sae_cfg.normalizer_type = scfg.normalizer_type
    cfg.sae_cfg.dict_mult = scfg.dict_mult
    cfg.optim_cfg.lr = scfg.lr
    cfg.optim_cfg.betas = scfg.betas
    cfg.lr_scheduler_cfg.cooldown_period = scfg.cooldown_period
    cfg.lr_scheduler_cfg.cooldown_factor = scfg.cooldown_factor
    cfg.l1_coeff = (
        scfg.l1_coeff
        # * sparsity_coeff_adjustment(scfg)
        # if scfg.sae_type == "SCSAE_MulGrads"
        # else (
        #     scfg.l1_coeff
        #     if not "SCSAE_RegGrads"
        #     else scfg.l1_coeff * 10
        # )
    )
    cfg.sparsity_penalty_type = scfg.sparsity_penalty_type
    return cfg, legacy_cfg


def sparsity_coeff_adjustment(scfg: ConfigFromSweep):
    if scfg.sparsity_penalty_type == "l1_sqrt":
        return 1.5
    d = {
        "SCSAE_MulGrads": 0.1,
        "SCSAE_RegGrads": 0.7,
        "VanillaSAE": 3,
    }
    if scfg.sae_type not in d:
        return 1
    return d[scfg.sae_type]


def get_configs_from_sweep(scfg: ConfigFromSweep = None, adjust_sparsity_coeff=True):
    scfg = scfg or ConfigFromSweep(**wandb.config)
    dict_mult = scfg.dict_mult
    if scfg.sae_type == "SqrtSAE":
        scfg.sparsity_penalty_type = "l1_sqrt"
        scfg.sae_type = "VanillaSAE"
    if scfg.sae_type == "SqrtSCSAE":
        scfg.sparsity_penalty_type = "l1_sqrt"
        scfg.sae_type = "SCSAE_RegGrads"

    sae_cfg = SAEConfig(dict_mult=dict_mult, sae_type=scfg.sae_type)

    buf_cfg = BufferConfig(
        layer=6,
        site="resid_pre",
        flatten_heads=False,
        device="cuda",
        d_data=sae_cfg.d_data,
        batch_size=4096,
        buffer_mult=2048,
        buffer_refresh_ratio=0.5,
        buffer_dtype="fp16",
        buffer_autocast_dtype="fp32",
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
        buffer_dtype="fp32",
        enc_dtype="fp32",
        device="cuda",
    )

    cfg = SAETrainConfig(
        sae_cfg=sae_cfg,
        optim_cfg=OptimConfig(),
        data_cfg=DataConfig(
            trainsplit=SplitConfig(
                splitname="train",
                start=0,
                end=40,
                tokens_from_split=400_000_000,
            )
        ),
        use_autocast=True,
        buffer_cfg=buf_cfg,
        lr_scheduler_cfg=LrSchedulerConfig(),
    )

    set_configs_from_sweep(cfg=cfg, legacy_cfg=legacy_cfg, scfg=scfg)
    return cfg, legacy_cfg

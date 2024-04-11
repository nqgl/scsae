from .config import SAETrainConfig
from nqgl.sc_sae.models import SAEConfig, BaseSAE
from unpythonic import box
from nqgl.sc_sae.models.mul_grads import LinearScaleSAE
import torch
import wandb
from nqgl.hsae_re.training.recons_modified import get_recons_loss

from nqgl.sc_sae.models.normalizer.base import NormalizerMixinBase
from nqgl.sc_sae.trainer.freq_tracker import FreqTracker, EMAFreqTracker
from nqgl.mlutils.components.component_layer.resampler.adam_resetter import AdamResetter


class BaseNormedSAE(NormalizerMixinBase, BaseSAE): ...


class Trainer:
    def __init__(self, cfg: SAETrainConfig, model, val_tokens, legacy_cfg):
        self.cfg = cfg
        self.model: BaseNormedSAE = cfg.sae_cfg.get_cls()(cfg.sae_cfg).to("cuda")

        self.freq_tracker = FreqTracker(d_dict=cfg.sae_cfg.d_dict)
        self.ema_freq_tracker = EMAFreqTracker(d_dict=cfg.sae_cfg.d_dict)
        self.optim = cfg.optim_cfg.get_optim(self.model.parameters())
        self.t = 1
        self.logfreq = 1
        self.log_recons_freq = 2000
        self.log_hists_freq = 3000
        self.llm_model = model
        self.llm_val_tokens = val_tokens
        self.current_datasource = None

        self.scheduler_epoch_interval = 10
        self.gradscaler = torch.cuda.amp.GradScaler() if self.cfg.use_autocast else None

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optim,
            self.cfg.lr_scheduler_cfg.get_lr_sched_lambda(self),
            verbose=True,
        )
        self.legacy_cfg = legacy_cfg

    def log(self, logdict, step=None):
        if step is None:
            step = self.t
        if self.cfg.wandb_project is not None:
            wandb.log(logdict, step=step)

    def forward_train_computation(self, x, acts_box, spoofed_acts_box):
        x_pred = self.model(x, acts_box=acts_box, spoofed_acts_box=spoofed_acts_box)
        l1_for_loss = self.l1(spoofed_acts_box.x)
        mse = (x - x_pred).pow(2).mean()
        loss = mse + self.cfg.sparsity_penalty(spoofed_acts_box.x)
        return x_pred, l1_for_loss, mse, loss

    def trainstep(self, x: torch.Tensor):
        self.optim.zero_grad()
        acts_box = box()
        spoofed_acts_box = box()
        if self.cfg.use_autocast:
            # x = x.float()
            x = self.model.normalize(x)
            with torch.autocast(device_type="cuda"):
                x_pred, l1_for_loss, mse, loss = self.forward_train_computation(
                    x, acts_box, spoofed_acts_box
                )
        else:
            # x = x.float()
            x = self.model.normalize(x)
            x_pred, l1_for_loss, mse, loss = self.forward_train_computation(
                x, acts_box, spoofed_acts_box
            )
        self.process_acts(acts_box.x)
        if x.isnan().any() or x.isinf().any():
            raise ValueError(
                "x has nans or infs! Possibly a data problem, eg. some acts are all 0s, which should not happen."
            )
        if self.cfg.use_autocast:
            self.gradscaler.scale(loss).backward()
        else:
            loss.backward()
        self.model.post_backward()
        if self.cfg.use_autocast:
            self.gradscaler.step(self.optim)
            self.gradscaler.update()
        else:
            self.optim.step()
        self.model.post_step()
        with torch.inference_mode():
            l2_norm = (x - x_pred).norm(dim=-1).mean()
            l0 = self.l0(acts_box.x)
            l1 = self.l1(acts_box.x)
            spoofed_l0 = self.l0(spoofed_acts_box.x)

        logdict = {
            "l1": l1.item(),
            "l0": l0.item(),
            "spoofed_acts/l0": spoofed_l0.item(),
            "spoofed_acts/l1": l1_for_loss.item(),
            "l2_norm": l2_norm.item(),
            "mse": mse.item(),
        }
        self.log_step(logdict)
        self.do_intermittent_actions()
        self.t += 1
        return logdict

    def do_intermittent_actions(self):
        if self.t % self.scheduler_epoch_interval == 0:
            self.scheduler.step()
        if (self.t + self.cfg.reset_before_resample) % self.cfg.resample_frequency == 0:
            self.freq_tracker.reset()
        if (
            self.t % self.cfg.resample_frequency == 0
            and self.t < self.cfg.lr_scheduler_cfg.cooldown_begin * 0.75
        ):
            self.resample()

    def process_acts(self, acts):
        self.ema_freq_tracker.update(acts)
        self.freq_tracker.update(acts)

    def train(self, datasource, skip_wandb=False):
        self.model.prime_normalizer(datasource)
        self.current_datasource = datasource
        if not skip_wandb and self.cfg.wandb_project is not None and wandb.run is None:
            wandb.init(
                entity="sae_all",
                project=self.cfg.wandb_project,
                config=self.cfg,
            )
        else:
            print("Not logging to wandb")
        for x in datasource:
            logdict = self.trainstep(x)

    def log_step(self, logdict):
        if (self.t - 1) % self.logfreq == 0:
            self.log(logdict, step=self.t)
        if (self.t - 1) % (self.logfreq * 10) == 0:
            self.log(
                {
                    "num_dead/ema": self.ema_freq_tracker.freqs.lt(3e-6).sum().item(),
                    "num_dead/count": self.freq_tracker.freqs.lt(3e-6).sum().item(),
                }
            )
        if (self.t - 1) % self.log_recons_freq == 0 and self.t > 1:
            self.log_recons()
        if (self.t - 1) % self.log_hists_freq == 0:
            self.loghists()

    def loghists(self):
        freqs = self.ema_freq_tracker.freqs
        hist = wandb.Histogram(freqs.cpu().numpy())
        self.log({"frequencies": hist})

    def log_recons(self):
        self.log(
            {
                **{
                    "num_steps": self.t,
                },
                **{
                    ("recons/" + k): v
                    for k, v in get_recons_loss(
                        self.llm_model,
                        self.model,
                        buffer=None,
                        all_tokens=self.llm_val_tokens,
                        cfg=self.legacy_cfg,
                        bos_processed_with_hook=False,
                    ).items()
                },
            },
            step=self.t,
        )
        self.log(
            {
                ("recons/with_proc_bos/" + k): v
                for k, v in get_recons_loss(
                    self.llm_model,
                    self.model,
                    buffer=None,
                    all_tokens=self.llm_val_tokens,
                    cfg=self.legacy_cfg,
                    bos_processed_with_hook=True,
                    num_batches=10 if self.t % 10_000 != 0 else 100,
                ).items()
            },
            step=self.t,
        )

    @classmethod
    def l1(cls, acts):
        return acts.relu().mean(dim=0).sum()

    @classmethod
    def l0(cls, acts: torch.Tensor):
        l0 = (acts > 0).count_nonzero() / acts.shape[0]
        return l0

    @classmethod
    def l2_sq(cls, x, x_pred):
        return (x - x_pred).pow(2).mean()

    @torch.inference_mode()
    def resample(self):
        print("resampling")
        dead = self.freq_tracker.freqs < 1e-6
        num_samples = dead.count_nonzero()
        if num_samples == 0:
            return
        resample_subset_size = 819_200
        sample_points = torch.zeros(
            resample_subset_size,
            self.cfg.sae_cfg.d_data,
            device="cuda",
            dtype=torch.float32,
        )
        l2_diff_sq = torch.zeros(
            resample_subset_size, device="cuda", dtype=torch.float32
        )
        batch_size = self.cfg.buffer_cfg.batch_size
        with torch.autocast(device_type="cuda"):
            for i in range(0, resample_subset_size, batch_size):
                x = next(self.current_datasource)
                x_pred = self.model(x)
                l2_diff_sq[i : i + batch_size] = (x - x_pred).pow(2).sum(dim=-1)
                sample_points[i : i + batch_size] = x
        indices = torch.multinomial(l2_diff_sq, num_samples, replacement=False)
        assert indices.numel() == num_samples

        avg_alive_norm = self.model.W_enc[:, ~dead].norm(dim=-1).mean()
        new_directions = sample_points[indices]
        new_directions = new_directions / new_directions.norm(dim=-1, keepdim=True)
        self.model.W_enc[:, dead] = new_directions.t() * avg_alive_norm * 0.02
        self.model.W_dec[dead] = new_directions
        self.model.b_enc[dead] = 0
        self.model.norm_dec()
        print("finished resampling")
        resetter = AdamResetter(self.model)
        resetter.W_enc.transpose(-2, -1)[dead](self.optim, alive_indices=~dead)
        resetter.W_dec[dead](self.optim, alive_indices=~dead)
        resetter.b_enc[dead](self.optim, alive_indices=~dead)
        # reset the optimizer at these locations

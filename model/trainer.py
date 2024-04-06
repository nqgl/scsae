from .configs import SAEConfig, SAETrainConfig
from unpythonic import box
from .model import BiasAdjustedSAE
import torch
import wandb
from nqgl.hsae_re.training.recons_modified import get_recons_loss


class Trainer:
    def __init__(self, cfg: SAETrainConfig, model, val_tokens, legacy_cfg):
        self.cfg = cfg
        self.model = BiasAdjustedSAE(cfg.sae_cfg).to("cuda")
        self.optim = cfg.optim_cfg.get_optim(self.model.parameters())
        self.t = 1
        self.logfreq = 1
        self.log_recons_freq = 1000
        self.log_hists_freq = 1000
        self.llm_model = model
        self.llm_val_tokens = val_tokens

        self.scheduler_epoch_interval = 100
        self.gradscaler = torch.cuda.amp.GradScaler() if self.cfg.use_autocast else None

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optim, self.lr_sched_lambda, verbose=True
        )
        self.legacy_cfg = legacy_cfg

    def lr_sched_lambda(self, epoch):
        """
        param is called 'epoch' for convention, but this is *not*
        by any means an epoch, this is just a count of how many times
        scheduler.step() has been called, and is accordingly ignored
        in favor of self.t (which has clear meaning regardless of
        frequency of calls to the scheduler)
        """
        lr_mul = self.lr_sched_mul()
        if wandb.run:
            self.log(
                {"lr_mult": lr_mul, "lr": self.cfg.optim_cfg.lr * lr_mul}, step=self.t
            )
        return lr_mul

    def lr_sched_mul(self):
        if not self.cfg.lr_scheduler_cfg:
            return 1

        if self.t < self.cfg.lr_scheduler_cfg.warmup_steps:
            lr_mul = self.t / self.cfg.lr_scheduler_cfg.warmup_steps
        elif self.t > self.cfg.lr_scheduler_cfg.cooldown_begin:
            lr_mul = max(
                1
                - (self.t - self.cfg.lr_scheduler_cfg.cooldown_begin)
                / (self.cfg.lr_scheduler_cfg.cooldown_period),
                1 / self.cfg.lr_scheduler_cfg.cooldown_factor,
            )
        else:
            lr_mul = 1
        return lr_mul

    def log(self, logdict, step=None):
        if step is None:
            step = self.t
        if self.cfg.wandb_project is not None:
            wandb.log(logdict, step=step)

    def forward_train_computation(self, x, acts_box, spoofed_acts_box):
        x_pred = self.model(x, acts_box=acts_box, spoofed_acts_box=spoofed_acts_box)
        l1_for_loss = self.l1(spoofed_acts_box.x)
        mse = (x - x_pred).pow(2).mean()
        loss = mse + l1_for_loss * self.cfg.l1_coeff
        return x_pred, l1_for_loss, mse, loss

    def trainstep(self, x):
        self.optim.zero_grad()
        acts_box = box()
        spoofed_acts_box = box()
        if self.cfg.use_autocast:
            with torch.autocast(device_type="cuda"):
                x_pred, l1_for_loss, mse, loss = self.forward_train_computation(
                    x, acts_box, spoofed_acts_box
                )
        else:
            x_pred, l1_for_loss, mse, loss = self.forward_train_computation(
                x, acts_box, spoofed_acts_box
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
        if self.t % self.scheduler_epoch_interval == 0:
            self.scheduler.step()
        self.t += 1
        return logdict

    def train(self, datasource, skip_wandb=False):
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
        if (self.t - 1) % self.log_recons_freq == 0:
            self.log_recons()
        if (self.t - 1) % self.log_hists_freq == 0:
            self.loghists()

    def loghists(self):
        pass  # TODO

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

from dataclasses import dataclass
import wandb


@dataclass
class LrSchedulerConfig:
    warmup_steps: int = 5_000
    cooldown_begin: int = 75_000
    cooldown_period: int = 20_000
    cooldown_factor: int = 10
    warmup_after_resample: int = 5_000

    def get_lr_sched_lambda(self, trainer):
        def lr_sched_mul():
            if trainer.t < self.warmup_steps:
                lr_mul = trainer.t / self.warmup_steps
            elif trainer.t > self.cooldown_begin:
                lr_mul = max(
                    1 - (trainer.t - self.cooldown_begin) / (self.cooldown_period),
                    1 / self.cooldown_factor,
                )
            elif (
                trainer.t % trainer.cfg.resample_frequency < self.warmup_after_resample
            ):
                lr_mul = (
                    trainer.t % trainer.cfg.resample_frequency
                ) / self.warmup_after_resample
            else:
                lr_mul = 1
            return lr_mul

        def lr_sched_lambda(epoch):
            """
            param is called 'epoch' for convention, but this is *not*
            by any means an epoch, this is just a count of how many times
            scheduler.step() has been called, and is accordingly ignored
            in favor of trainer.t (which has clear meaning regardless of
            frequency of calls to the scheduler)
            """
            lr_mul = lr_sched_mul()
            if wandb.run:
                trainer.log(
                    {"lr_mult": lr_mul, "lr": trainer.cfg.optim_cfg.lr * lr_mul},
                    step=trainer.t,
                )
            return lr_mul

        return lr_sched_lambda

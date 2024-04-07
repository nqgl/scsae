from dataclasses import dataclass


@dataclass
class LrSchedulerConfig:
    warmup_steps: int = 2_000
    cooldown_begin: int = 60_000
    cooldown_period: int = 20_000
    cooldown_factor: int = 10

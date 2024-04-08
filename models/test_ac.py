import sys

# sys.path.append("./nqgl/hsae_re/")
from nqgl.sc_sae.models import test
from nqgl.sc_sae.data import ac_cfg
import tqdm

ac = ac_cfg.ac
cfg, lgcfg = test.get_configs(
    {
        "betas": (0.9, 0.97),
        "l1_coeff": 2e-3,
    }
)
cfg.lr_scheduler_cfg.cooldown_begin = 10_000
cfg.lr_scheduler_cfg.cooldown_period = 1_000
trainer = test.get_trainer(cfg, lgcfg)
cfg.use_autocast = False
trainer.train(ac.read_as_iter(4096))
# print("switching data sources")
# buffer = test.Buffer(
#     trainer.cfg.buffer_cfg,
#     test.train_tokens,
#     test.model,
# )


# def train_buffer():
#     for i in tqdm.tqdm(range(90000 * 20 * 1024 // 2048)):
#         yield buffer.next()


# trainer.train(train_buffer())

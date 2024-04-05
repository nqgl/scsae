import sys

# sys.path.append("./nqgl/hsae_re/")
from nqgl.sc_sae.model import test, ac_cfg
import tqdm

ac = ac_cfg.ac
trainer = test.get_trainer(*test.get_configs())

trainer.train(ac.read_as_iter(2048))
print("switching data sources")
buffer = test.Buffer(
    trainer.cfg.buffer_cfg,
    test.train_tokens,
    test.model,
)


def train_buffer():
    for i in tqdm.tqdm(range(90000 * 20 * 1024 // 2048)):
        yield buffer.next()


trainer.train(train_buffer())

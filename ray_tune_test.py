from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
import sys

# sys.path.append("./nqgl/hsae_re/")
from nqgl.sc_sae.model import test, ac_cfg
import tqdm
from ray import tune
from ray import train

import torch


# @wandb_mixin
def train_test_betas_dict(d={}):
    ac = ac_cfg.ac
    cfg, legacy_cfg = test.get_configs(d)
    cfg.wandb_project = "ray_test"
    trainer = test.get_trainer(cfg, legacy_cfg)
    # trainer.model = train.torch.prepare_model(trainer.model)
    setup_wandb(config=cfg, project="ray_test", entity="sae_all")
    trainer.train(ac.read_as_iter(1024, cast=torch.float16, stop_after=100))
    print("switching data sources")

    return {"meaningless_test_value": 37, **d}

    def train_buffer():
        for i in tqdm.tqdm(range(90000 * 20 * 1024 // 2048)):
            yield buffer.next()

    buffer = test.Buffer(
        cfg.buffer_cfg,
        test.train_tokens,
        test.model,
    )
    trainer.train(train_buffer())


b1vals = [0.7, 0.9, 0.8]
b2vals = [0.9, 0.97, 0.99, 0.997]

# b1vals = [0.5, 0.9]
# b2vals = [0.5, 0.9]
trial_space = {
    # This is an example parameter. You could replace it with filesystem paths,
    # model types, or even full nested Python dicts of model configurations, etc.,
    # that enumerate the set of trials to run.
    "betas": tune.grid_search([(b1, b2) for b1 in b1vals for b2 in b2vals])
}
train_model = tune.with_resources(train_test_betas_dict, {"cpu": 1, "gpu": 1 / 5})


tuner = tune.Tuner(
    train_model,
    param_space=trial_space,
)
# tcfg = tune.TuneConfig(tune.create_searcher("grid"), max_concurrent_trials=8)

results = tuner.fit()
print(results)

# tune.run(
#     config={
#         "wandb": {"project": "Optimization_Project", "api_key_file": "/path/to/file"},
#     }
# )
import ray


@ray.remote(num_gpus=1 / 4, num_cpus=1)
class RemoteTrainer(test.Trainer): ...


def remote_trainer(d={}):
    ac = ac_cfg.ac
    cfg, legacy_cfg = test.get_configs(d)
    cfg.wandb_project = "ray_test"
    trainer = RemoteTrainer.remote(
        cfg, model=test.model, val_tokens=test.val_tokens, legacy_cfg=legacy_cfg
    )
    # trainer.model = train.torch.prepare_model(trainer.model)
    setup_wandb(config=cfg, project="ray_test", entity="sae_all")
    trainer.train(ac.read_as_iter(1024, cast=torch.float16, stop_after=100))
    print("switching data sources")

    return {"meaningless_test_value": 37, **d}


def another():
    chunk_num = 0
    ac = ac_cfg.ac
    batch_size = 2048
    big_buffer = torch.zeros(
        ac.get_tensor_len() * 30, 768, dtype=torch.float16, device="cpu"
    )
    trainers = []
    while True:
        for i in range(30):
            big_buffer[i * ac.get_tensor_len() : (i + 1) * ac.get_tensor_len()] = (
                ac.read_chunk(chunk_num, read_device="cpu")
            )
            chunk_num += 1

        def data_iter():
            for i in range(0, big_buffer.shape[0], batch_size):
                yield big_buffer[i : i + batch_size].to("cuda")

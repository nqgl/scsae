from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
import sys

# sys.path.append("./nqgl/hsae_re/")
from nqgl.sc_sae.model import test, ac_cfg
import tqdm
from ray import tune
from ray import train


# @wandb_mixin
def train_test_betas_dict(d={}):
    ac = ac_cfg.ac
    cfg, legacy_cfg = test.get_configs(d)
    cfg.wandb_project = "ray_test"
    trainer = test.get_trainer(cfg, legacy_cfg)
    # trainer.model = train.torch.prepare_model(trainer.model)
    setup_wandb(config=cfg, project="ray_test", entity="sae_all")
    # trainer.train(ac.read_as_iter(4096, stop_after=32))
    print("switching data sources")

    def train_buffer():
        for i in tqdm.tqdm(range(90000 * 20 * 1024 // 2048)):
            yield buffer.next()

    buffer = test.Buffer(
        cfg.buffer_cfg,
        test.train_tokens,
        test.model,
    )
    trainer.train(train_buffer())
    return {"meaningless_test_value": 37, **d}


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

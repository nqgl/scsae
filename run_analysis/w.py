# %%
import wandb
import wandb.data_types
import wandb.wandb_run

api = wandb.Api()

# run = api.run("sae_all/scsae_comparisons/2x7z1z5o")
sweep1 = api.sweep("sae_all/scsae_comparisons/n796532b")
sweep2 = api.sweep("sae_all/scsae_comparisons/qrzgmfzh")
# %%
runs1 = [run for run in sweep1.runs]
runs2 = [run for run in sweep2.runs]
len(runs1), len(runs2)
# %%


# def add_nice_name(run):
#     nice_name = run.config["sae_type"]
#     if run.config["sparsity_penalty_type"] == "l1_sqrt":
#         nice_name = "Sqrt(" + nice_name + ")"
#     run.config["nice_name"] = nice_name
#     run.update()


# for run in runs1:
#     add_nice_name(run)


def add_final_nats_lost(run):
    if run.summary.get("recons_final/with_bos/loss") is None:
        raise ValueError("No recons_final/with_bos/loss")
    run.summary["final_nats_lost"] = run.summary["recons_final/with_bos/loss"]
    run.update()


# %%
loss_baseline = "recons_final/with_bos/loss"
rec_loss = "recons_final/with_bos/recons_loss"
r = runs2[2]
if r.state == "finished":
    for i, row in r.history(
        keys=[
            loss_baseline,
            rec_loss
        ]
    ).iterrows():
        print(row[]
    # r.summary
# %%

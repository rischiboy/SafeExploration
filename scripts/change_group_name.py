import wandb

api = wandb.Api()
for r in api.runs("safex/Week-18", filters={"group": "CEM-BNN"}):
    r.group = "MinMax-BNN"
    r.update()

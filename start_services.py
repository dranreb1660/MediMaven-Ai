import wandb, huggingface_hub, os, torch, numpy as np

HF_TOKEN =  "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"
WANDB_KEY = "0e7613d70774bd853ddb2dc316968c77437977be"
SEED = 42
def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
seed_everything(SEED)


# Login to the Hugging Face Hub
huggingface_hub.login(
  token=os.getenv("HF_TOKEN"), # ADD YOUR TOKEN HERE
  add_to_git_credential=False
)
wandb.login(key=os.getenv("WANDB_API_KEY"), relogin=True)
# wandb.login(key=WANDB_KEY)
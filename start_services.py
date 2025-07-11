import os, wandb, huggingface_hub, torch, numpy as np

# Seed everything for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# HF login for private weights / rate limits (optional)
if os.getenv("HF_TOKEN"):
  print("🔑 Logging into Hugging Face Hub...")
  huggingface_hub.login(
      token=os.getenv("HF_TOKEN"),
      add_to_git_credential=False,
  )
else:
  print("ℹ️  No HF_TOKEN - using public access")

if os.getenv("WANDB_API_KEY"):
  print("🔑 Logging into Weights & Biases...")
  wandb.login(key=os.getenv("WANDB_API_KEY"), relogin=True)
else:
  print("ℹ️  No WANDB_API_KEY - monitoring disabled")
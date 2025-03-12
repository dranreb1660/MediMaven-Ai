"""

#wandb sweep sweep_config.yaml => gets you a sweep_id
# wandb agent your_username/your_project/<sweep_id>

"""
import wandb
from zenml.client import Client
from pipelines.ltr_tuning_pipeline import tuning_pipeline


def run_sweep():
    print("Starting sweep run...")
    
    run = wandb.init(project="MediMaven-LTR", name="temp_run", reinit=True)

    # Read hyperparameters from wandb.config
    lr = wandb.config.learning_rate
    n_est = wandb.config.n_estimators
    m_depth = wandb.config.max_depth

    run_name=f"sweep_lr={lr}_est={n_est}_depth={m_depth}"
    # Update the run name (this works in many versions of W&B)
    run.name = run_name

    # Initialize wandb once for the entire sweep run
    print(f"Starting pipeline with lr={lr}, n_estimators={n_est}, max_depth={m_depth}")
    

    # Assume tuning_pipeline returns the final metric.
    final_metric = tuning_pipeline(
        run_name=run_name,
        parameters={
            "train_and_eval_ranker": {
                "learning_rate": lr,
                "n_estimators": n_est,
                "max_depth": m_depth
            }
        }
    )
    
    # Finish the wandb run
    wandb.finish()
    print("W&B run finished.")

if __name__ == "__main__":
    run_sweep()
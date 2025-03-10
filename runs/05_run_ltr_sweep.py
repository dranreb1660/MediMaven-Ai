"""

#wandb sweep sweep_config.yaml => gets you a sweep_id
# wandb agent your_username/your_project/<sweep_id>

"""
import wandb
from zenml.client import Client
from pipelines.ltr_tuning_pipeline import tuning_pipeline


def run_sweep():
    # 1) Initialize a new W&B run
    wandb.init()
    
    # 2) Read hyperparams from wandb.config
    lr = wandb.config.learning_rate
    n_est = wandb.config.n_estimators
    m_depth = wandb.config.max_depth

    # 3) Call the pipeline function with these parameters
    #    In older/legacy or certain ZenML versions, calling the pipeline
    #    directly triggers an immediate run (no .run() needed).
    tuning_pipeline(
        run_name=f"sweep_lr={lr}_est={n_est}_depth={m_depth}",
        parameters={
            "train_and_eval_ranker": {
                "learning_rate": lr,
                "n_estimators": n_est,
                "max_depth": m_depth
            }
        }
    )

    # 4) Optionally, log or print anything else. The pipeline run is done.

if __name__ == "__main__":
    run_sweep()
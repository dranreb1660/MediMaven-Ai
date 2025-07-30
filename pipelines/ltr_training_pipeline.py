"""
Learning-to-Rank pipeline for MediMaven using XGBoost.
Tunes ranking models to improve retrieval quality.
"""

# final_pipeline.py

import os
import json
import shutil
import numpy as np
import pandas as pd
from math import log2
import wandb

from typing import Tuple, Dict, Any, List
from xgboost import XGBRanker
from zenml.pipelines import pipeline
from zenml.steps import step

from src.backend.utils import compute_ndcg_at_k
from pipelines.ltr_tuning_pipeline import (
    fetch_ltr_data, split_train_test, build_features
)

@step
def merge_train_eval(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame
) -> pd.DataFrame:
    """Combine training and validation sets"""
    combined_df = pd.concat([train_df, eval_df], ignore_index=True)
    return combined_df

@step(enable_cache=False)
def train_final_ranker(
    combined_df: pd.DataFrame,
    test_df: pd.DataFrame,
    hyperparam_list: List[Dict[str, Any]] = [
        {"learning_rate": 0.1, "n_estimators": 200, "max_depth": 7},
        {"learning_rate": 0.1, "n_estimators": 200, "max_depth": 5},
        {"learning_rate": 0.1, "n_estimators": 200, "max_depth": 3},
        {"learning_rate": 0.01, "n_estimators": 100, "max_depth": 3},
        {"learning_rate": 0.03, "n_estimators": 50, "max_depth": 3},
    ],
    k_eval: int = 10,
    best_model_path: str = "./models/ltr_best_model.json"
) -> Dict[str, Any]:
    """Train ranker with hyperparameter sweep and save best model"""
    import wandb

    # Sort test for grouping
    test_df = test_df.sort_values("question_id")
    test_groups = test_df.groupby("question_id").size().tolist()

    X_test = test_df[["cos_sim", "l2_dist", "context_length"]].values
    y_test = test_df["label"].values

    # We'll do a loop over hyperparam_list
    best_ndcg = -1.0
    best_params = None

    for i, hps in enumerate(hyperparam_list, start=1):
        lr = hps.get("learning_rate", 0.1)
        est = hps.get("n_estimators", 100)
        depth = hps.get("max_depth", 3)
        run_name=f"finalRun_{i}_lr={lr}_est={est}_depth={depth}"


        # Train on combined_df
        combined_df = combined_df.sort_values("question_id")
        group_arr = combined_df.groupby("question_id").size().tolist()

        X_train = combined_df[["cos_sim", "l2_dist", "context_length"]].values
        y_train = combined_df["label"].values

        ranker = XGBRanker(
            objective="rank:ndcg",
            learning_rate=lr,
            n_estimators=est,
            max_depth=depth,
            eval_metric="ndcg",
            tree_method="auto"
        )
        ranker.fit(X_train, y_train, group=group_arr)

        # Evaluate on test
        y_scores = ranker.predict(X_test)
        idx = 0
        ndcg_list = []
        for gsize in test_groups:
            labels_g = y_test[idx: idx+gsize]
            scores_g = y_scores[idx: idx+gsize]
            ndcg_val = compute_ndcg_at_k(labels_g, scores_g, k_eval)
            ndcg_list.append(ndcg_val)
            idx += gsize
        mean_ndcg = float(np.mean(ndcg_list))
        print(f"[Set {i}] NDCG@{k_eval}={mean_ndcg:.4f} for lr={lr}, est={est}, depth={depth}")

        # Log to W&B
        wandb.init(project="MediMaven-LTR", job_type="final_train_evaluation",name=run_name, reinit=True)
        wandb.config.update({
            "learning_rate": lr,
            "n_estimators": est,
            "max_depth": depth
        })
        wandb.log({f"test_ndcg@{k_eval}": mean_ndcg})

        # Save if best model so far
        if mean_ndcg > best_ndcg:
            best_ndcg = mean_ndcg
            best_params = {"learning_rate": lr, "n_estimators": est, "max_depth": depth}

            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            ranker.save_model(best_model_path)
            print(f"*** New best model => saved to {best_model_path} ***")
            artifact = wandb.Artifact("best_ltr_model", type="model")
            artifact.add_file(best_model_path)
            wandb.log_artifact(artifact)

        wandb.finish()

    print(f"\nBest NDCG@{k_eval}={best_ndcg:.4f} with params={best_params}")
    return {"best_ndcg": best_ndcg, "best_params": best_params}


@pipeline
def ltr_training_pipeline():
    """Complete LTR training pipeline with hyperparameter tuning"""
    hyperparam_list= [
    {"learning_rate": 0.1, "n_estimators": 200, "max_depth": 7},
    {"learning_rate": 0.1, "n_estimators": 200, "max_depth": 5},
    {"learning_rate": 0.1, "n_estimators": 200, "max_depth": 3},
    {"learning_rate": 0.01, "n_estimators": 100, "max_depth": 3},
    {"learning_rate": 0.03, "n_estimators": 50, "max_depth": 3},
]
    ltr_collection_name =  "ltr_emb_dataset"

    df = fetch_ltr_data(ltr_collection_name)
    train_df, temp = split_train_test(df, test_size=0.4)
    eval_df, test_df = split_train_test(temp,test_size=0.5)

    train_df = build_features(train_df)
    eval_df = build_features(eval_df)
    test_df = build_features(test_df)
    combined_df = merge_train_eval(train_df, eval_df)

    train_final_ranker(
        combined_df, 
        test_df,
        hyperparam_list = hyperparam_list
    )

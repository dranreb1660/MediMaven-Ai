"""
pipelines/ltr_pipeline.py

ZenML pipeline for Learning-to-Rank (LTR) with XGBoost, assuming:
  - We already performed chunking + embedding in a separate pipeline,
    storing question_embedding/context_embedding in MongoDB.
  - Each question may have multiple contexts: 1 positive, 2 negatives, etc.
  - Our final goal is to rank contexts so that relevant ones appear first.

Data in MongoDB (collection: "ltr_dataset") might have columns:
  question, context, label, question_id, context_id,
  question_embedding, context_embedding, metadata, ...

Steps:
  1) fetch_ltr_data: Reads from "ltr_dataset" collection in MongoDB.
  2) split_train_test: Splits data into train/test sets.
  3) build_features: Computes numeric features (e.g. cos_sim, L2 distance).
  4) train_ranker: Trains an XGBRanker in ranking mode (rank:ndcg).
  5) evaluate_ranker: Computes NDCG@k on test set.
  6) save_ranker: Saves the final model artifact.

This pipeline demonstrates:
  - A production-ready approach with real embeddings and negative sampling.
  - Grouping by question_id for ranking.
  - Weights & Biases integration for experiment tracking.
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

from src.utils import compute_ndcg_at_k
from pipelines.ltr_tuning_pipeline import (
    fetch_ltr_data, split_train_test, build_features
)

@step
def merge_train_eval(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine train_df + eval_df to form one bigger training set.
    """
    combined_df = pd.concat([train_df, eval_df], ignore_index=True)
    # maybe drop duplicates or do any final cleaning
    return combined_df

# ------------- STEP C: Multi-Param Final Training ------------- #
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
    """
    Merges train+eval => final_train, then for each hyperparam in hyperparam_list:
      1) Train XGBoost ranker on final_train
      2) Evaluate on test_df (NDCG@k)
      3) If best so far, save model to best_model_path, log as W&B artifact
    Returns a dict with {best_ndcg, best_params}.
    """
    import wandb
    # We no longer need this check since the default is now properly set in the function signature

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

        # Log run to W&B
        #   (One approach is we do a single run for each set. Another is a single run for the entire loop.)
        #   We'll do "one run per set" so each appears as a separate run in W&B
        wandb.init(project="MediMaven-LTR", job_type="final_train_evaluation",name=run_name, reinit=True)
        wandb.config.update({
            "learning_rate": lr,
            "n_estimators": est,
            "max_depth": depth
        })
        wandb.log({f"test_ndcg@{k_eval}": mean_ndcg})

        # If best so far, save model & log artifact
        if mean_ndcg > best_ndcg:
            best_ndcg = mean_ndcg
            best_params = {"learning_rate": lr, "n_estimators": est, "max_depth": depth}

            # Save model
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            ranker.save_model(best_model_path)
            print(f"*** New best model => saved to {best_model_path} ***")

            # Log artifact
            artifact = wandb.Artifact("best_ltr_model", type="model")
            artifact.add_file(best_model_path)
            wandb.log_artifact(artifact)

        wandb.finish()

    print(f"\nBest NDCG@{k_eval}={best_ndcg:.4f} with params={best_params}")
    return {"best_ndcg": best_ndcg, "best_params": best_params}


# ------------- PIPELINE DEFINITION ------------- #
@pipeline
def ltr_training_pipeline():
    """
    1) fetch_and_3split_data => train_df, eval_df, test_df
    2) build_features => train_df, eval_df, test_df
    3) multi_param_training => merges train+eval => final train, tries each hyperparam set on test => saves best
    """
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

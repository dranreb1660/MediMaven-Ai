"""
Hyperparameter tuning with 

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

import os
import json
import wandb
import numpy as np
import pandas as pd
from typing import Tuple, List, Any

from zenml.pipelines import pipeline
from zenml.steps import step
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from xgboost import XGBRanker

# Utility: ensures Mongo is running, returns a db handle
from src.utils import (get_mongo_connection, ensure_mongodb_running,
          cosine_similarity,l2_distance, compute_ndcg_at_k )


# -------------- STEP A: Fetch LTR Data from Mongo -------------- #

@step
def fetch_ltr_data(
    collection_name: str = "ltr_emb_dataset"
) -> pd.DataFrame:
    """
    Reads the LTR dataset from MongoDB, which already contains:
      question, context, label, question_id, context_id,
      question_embedding, context_embedding, etc.

    Expects each row to represent (question, context) pair with label=1 or 0.
    We have (1) positive for each question, (2) negatives, etc.

    Returns:
      DataFrame with the necessary columns.
    """
    ensure_mongodb_running()
    db = get_mongo_connection()
    collection = db[collection_name]

    data = list(collection.find({}, {'_id': 0}))
    df = pd.DataFrame(data)

    # Basic cleaning
    df.dropna(subset=["question", "context", "question_embeddings", "context_embeddings", "label"], inplace=True)
    df.drop_duplicates(subset=["question_id", "context_id"], inplace=True)

    print(f"Fetched {len(df)} rows from '{collection_name}'")
    return df

# -------------- STEP B: Split Train / Test -------------- #

@step
def split_train_test(
    df: pd.DataFrame,
    test_size: float,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into train/test. In ranking scenarios, we often
    group by question. Here we do a simple row-based split. For large
    production systems, consider grouping or stratified approaches.

    Returns: (train_df, test_df)
    """
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_seed)
    print(f"Train set: {len(train_df)} rows, Test set: {len(test_df)} rows")
    return train_df, test_df

# -------------- STEP C: Build Features -------------- #

@step
def build_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Uses question_embedding/context_embedding to generate numeric features:
      - cos_sim
      - l2_dist
      - (optionally) text length, domain signals, etc.

    Output columns:
      question_id, context_id, label, cos_sim, l2_dist, ...
    """
    df = df.copy()

    # Convert stored embeddings from list to np array (if they aren't already)
    # We'll produce a final DataFrame with columns for numeric features + label
    cos_sims = []
    l2_dists = []
    context_lengths = []

    for idx, row in df.iterrows():
        q_emb = np.array(row["question_embeddings"], dtype=np.float32)
        c_emb = np.array(row["context_embeddings"], dtype=np.float32)

        cos_sims.append(cosine_similarity(q_emb, c_emb))
        l2_dists.append(l2_distance(q_emb, c_emb))
        context_lengths.append(len(str(row["context"]).split()))

    df["cos_sim"] = cos_sims
    df["l2_dist"] = l2_dists
    df["context_length"] = context_lengths

    # Return columns needed for LTR
    # question_id, context_id, label, cos_sim, l2_dist, context_length, etc.
    return df

# -------------- STEP D: Train n Evaluate XGBRanker -------------- #

@step
def train_and_eval_ranker(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    learning_rate: float,
    n_estimators: int,
    max_depth: int,
    ranking_objective: str = "rank:ndcg",
    k_eval: int = 10
) -> float:
    """
    Trains XGBRanker on train_df, evaluates on eval_df, logs NDCG, 
    returns the NDCG as the "score" for hyperparam tuning.
    """

    wandb.init(project="MediMaven-LTR", job_type="train_eval", reinit=True)
    wandb.config.update({
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "max_depth": max_depth
    })

    # Sort by question_id
    train_df = train_df.sort_values("question_id")
    eval_df = eval_df.sort_values("question_id")

    # Build group array for train
    train_group_sizes = train_df.groupby("question_id").size().values.tolist()
    # Build group array for eval
    eval_group_sizes = eval_df.groupby("question_id").size().values.tolist()

    feature_cols = ["cos_sim", "l2_dist", "context_length"]
    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values

    X_eval = eval_df[feature_cols].values
    y_eval = eval_df["label"].values

    # Train ranker
    ranker = XGBRanker(
        objective=ranking_objective,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        eval_metric="ndcg",
        tree_method="auto"
    )

    ranker.fit(
        X_train,
        y_train,
        group=train_group_sizes,
        eval_set=[(X_eval, y_eval)],
        eval_group=[eval_group_sizes],
        verbose=False
    )

    # Predict on eval
    y_scores = ranker.predict(X_eval)

    # Compute NDCG@k per query
    start = 0
    ndcg_list = []
    for gsize in eval_group_sizes:
        end = start + gsize
        labels_g = y_eval[start:end]
        scores_g = y_scores[start:end]
        ndcg_val = compute_ndcg_at_k(labels_g, scores_g, k=k_eval)
        ndcg_list.append(ndcg_val)
        start = end

    mean_ndcg = float(np.mean(ndcg_list))
    wandb.log({f"eval_ndcg@{k_eval}": mean_ndcg})
    wandb.finish()

    # Return the metric so that we can use it in W&B Sweeps
    return mean_ndcg

# ---------- TUNING PIPELINE DEFINITION ----------

@pipeline
def tuning_pipeline(run_name: str = None, parameters: dict = None):
    """
    Pipeline for hyperparameter tuning.
    Args:
        run_name: Name for the pipeline run
        parameters: Dictionary of parameters for pipeline steps
    """
    df = fetch_ltr_data()
    train_df, temp = split_train_test(df, test_size=0.4)
    eval_df, test_df = split_train_test(temp,test_size=0.5)

    train_df = build_features(train_df)
    eval_df = build_features(eval_df)
    
    # Extract parameters for train_and_eval_ranker if provided
    train_params = {}
    if parameters and "train_and_eval_ranker" in parameters:
        train_params = parameters["train_and_eval_ranker"]
    
    train_and_eval_ranker(train_df, eval_df, **train_params)

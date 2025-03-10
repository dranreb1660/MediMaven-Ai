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
from src.utils import get_mongo_connection, ensure_mongodb_running

# -------------- HELPER: Cosine Similarity & L2 Distance -------------- #

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot_val = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_val / (norm_a * norm_b + 1e-9)

def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)

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
    test_size: float = 0.2,
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

# -------------- STEP D: Train XGBRanker -------------- #

@step
def train_ranker(
    train_df: pd.DataFrame,
    learning_rate: float = 0.1,
    n_estimators: int = 100,
    ranking_objective: str = "rank:ndcg"
) -> XGBRanker:
    """
    Trains an XGBoost ranker on our numeric features, grouping by question_id.

    Feature columns might be [cos_sim, l2_dist, context_length, ...].
    Label is the binary relevance (1 or 0).
    Groups are derived from how many rows belong to each question_id.

    Returns the fitted model.
    """
    # Sort so that all rows for a question are contiguous
    train_df = train_df.sort_values("question_id")

    # Build group array for XGBoost (size of each question's doc set).
    group_series = train_df.groupby("question_id").size()
    group_sizes = group_series.values.tolist()

    # Extract features + labels
    feature_cols = ["cos_sim", "l2_dist", "context_length"]
    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values

    # Initialize ranker
    ranker = XGBRanker(
        objective=ranking_objective,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        eval_metric="ndcg",
        tree_method="auto"   # or 'gpu_hist' if you have GPU
    )

    # Train
    ranker.fit(
        X_train,
        y_train,
        group=group_sizes,
        verbose=True
    )

    # Log hyperparams to W&B
    wandb.init(project="MediMaven-LTR", job_type="train_ranker", reinit=True)
    wandb.config.update({
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "objective": ranking_objective
    })

    return ranker

# -------------- NDCG Calculation -------------- #

def compute_ndcg_at_k(labels: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    """
    Compute NDCG@k for a single query:
      1) Sort docs by predicted score descending
      2) Compute DCG of top-k
      3) Compute IDCG (ideal ranking)
      4) Return DCG/IDCG
    """
    from math import log2

    # Sort by predicted score, descending
    idx_sorted = np.argsort(-scores)
    ideal_sorted = np.argsort(-labels)

    dcg = 0.0
    idcg = 0.0

    for i in range(k):
        if i < len(idx_sorted):
            rel = labels[idx_sorted[i]]
            dcg += (2**rel - 1) / log2(i+2)
        if i < len(ideal_sorted):
            ideal_rel = labels[ideal_sorted[i]]
            idcg += (2**ideal_rel - 1) / log2(i+2)

    return dcg / (idcg + 1e-9)

# -------------- STEP E: Evaluate Ranker on Test Set -------------- #

@step
def evaluate_ranker(
    model: XGBRanker,
    test_df: pd.DataFrame,
    k_eval: int = 10
) -> float:
    """
    Computes mean NDCG@k across all queries in the test set.

    test_df must have question_id, label, and the numeric feature columns used in training.
    We predict a score for each row, group by question_id, then compute NDCG@k.

    Returns:
      mean_ndcg (float)
    """
    test_df = test_df.sort_values("question_id")

    feature_cols = ["cos_sim", "l2_dist", "context_length"]
    X_test = test_df[feature_cols].values
    y_true = test_df["label"].values

    # Build group array
    group_series = test_df.groupby("question_id").size()
    group_sizes = group_series.values.tolist()

    # Predict
    y_scores = model.predict(X_test)

    # Compute NDCG@k per query
    ndcg_values = []
    start_idx = 0
    for size in group_sizes:
        end_idx = start_idx + size
        labels_group = y_true[start_idx:end_idx]
        scores_group = y_scores[start_idx:end_idx]

        ndcg_val = compute_ndcg_at_k(labels_group, scores_group, k=k_eval)
        ndcg_values.append(ndcg_val)
        start_idx = end_idx

    mean_ndcg = float(np.mean(ndcg_values))
    print(f"NDCG@{k_eval} on test set = {mean_ndcg:.4f}")

    wandb.log({f"test_ndcg@{k_eval}": mean_ndcg})
    return mean_ndcg

# -------------- STEP F: Save Ranker Model -------------- #

@step
def save_ranker(
    model: XGBRanker,
    model_save_path: str = "./models/ltr_xgboost.json"
):
    """
    Saves the trained XGBRanker model to disk. 
    In production, you could store to S3 / MLflow / other artifact store.
    """
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save_model(model_save_path)
    print(f"✅ LTR model saved to {model_save_path}")

    wandb.log({"model_artifact_path": model_save_path})
    wandb.finish()

# -------------- DEFINE THE ZENML PIPELINE -------------- #

@pipeline
def ltr_pipeline():
    """
    Production-ready pipeline for Learning-to-Rank with XGBoost:
      1. Fetch data from 'ltr_dataset' (MongoDB).
      2. Split into train/test sets.
      3. Build numeric features from stored embeddings (cos_sim, etc.).
      4. Train the XGBRanker on the train set.
      5. Evaluate with NDCG@10 on the test set.
      6. Save the final model artifact.

    This pipeline demonstrates how negative samples
    (2 negatives, 1 positive) factor into the ranking objective,
    grouping by question_id so each question's contexts are
    ranked properly.
    """
    df = fetch_ltr_data()
    train_df, test_df = split_train_test(df)
    train_df = build_features(train_df)
    test_df = build_features(test_df)

    model = train_ranker(train_df)
    _ = evaluate_ranker(model, test_df)
    save_ranker(model)

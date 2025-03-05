"""
A ZenML pipeline that:
1. Fetches cleaned Q-C-A data from MongoDB (with metadata).
2. Performs negative sampling to create:
   - An LTR dataset: (question, context, label, answer, [metadata])
   - An LLM dataset: (input_text, target_text, [metadata])

Optionally, stores or logs these results (CSV, MongoDB, W&B).
"""

import random
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Any, Union

from zenml.pipelines import pipeline
from zenml.steps import step
from datetime import datetime
from pymongo import MongoClient

import wandb

from src.utils import ensure_mongodb_running, get_mongo_connection

# ------------------ STEP A ------------------ #
@step
def fetch_data_from_mongo(
    collection_name: str
) -> pd.DataFrame:
    """
    Step A: Fetch existing cleaned Q-C-A data from MongoDB.

    :param collection_name: Name of the collection that we want to fetch from that stores the cleaned data.
    :return: A pandas DataFrame with columns like ["question", "context", "answer", "metadata_field", ...].
    """
    db = get_mongo_connection()
    collection = db[collection_name]

    # Convert cursor to DataFrame
    data = list(collection.find({}, {"_id": 0}))
    df = pd.DataFrame(data)
    return df

# ------------------ STEP B ------------------ #
@step
def create_ltr_and_llm_datasets(
    df: pd.DataFrame,
    num_negatives: int = 2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Step B: Given the DataFrame from Mongo, create two "views":
    1) LTR dataset with negative samples
    2) LLM dataset for fine-tuning

    The LTR dataset: 
      columns ["question", "context", "label", "answer", ... plus any metadata if desired]
    The LLM dataset:
      columns ["input_text", "target_text", ... plus any metadata if desired]

    param:
        df: DataFrame with columns ["question", "context", "answer", plus metadata].
        num_negatives: How many negative contexts to sample per question.

    :return: 
        (ltr_df, llm_df)
    """

    # --- Step B1: LTR Negative Sampling --- #
    # We'll take "question", "context", "answer" are the main columns
    # and everything else is metadata we might keep.

    # For negative sampling, we create "label=1" for ground-truth Q–C pairs
    # and "label=0" for random negative contexts.
 
    #the data from the medquad is noisy so we well exclude for the LTR training

    ltr_df = df[df['Dataset'] == 'iCliniQ'].reset_index(drop=True)

    # Gather all unique contexts once
    ltr_contexts = ltr_df['context'].unique().tolist()
    ltr_records = []

    # Group by question to handle each question's positive contexts
    grouped = ltr_df.groupby("question")

    for question, group_df in grouped:
        # For each ground-truth pair
        for _, row in group_df.iterrows():
            pos_context = row["context"]
            answer = row["answer"]

            # We'll keep any extra metadata columns for reference
            row_dict = row.to_dict()

            # Positive record
            pos_record = {
                "question": question,
                "context": pos_context,
                "label": 1,
                "answer": answer
            }
            # Copy over any extra metadata fields
            for col in row_dict:
                if col not in ["question", "context", "answer"]:
                    pos_record[col] = row_dict[col]

            ltr_records.append(pos_record)

            # Negative sampling: pick random contexts that are not pos_context
            negative_candidates = [ctx for ctx in ltr_contexts if ctx != pos_context]
            neg_contexts = random.sample(
                negative_candidates, 
                min(num_negatives, len(negative_candidates))
            )

            for neg_c in neg_contexts:
                neg_record = {
                    "question": question,
                    "context": neg_c,
                    "label": 0,
                    "answer": answer
                }
                # copy over metadata
                for col in row_dict:
                    if col not in ["question", "context", "answer"]:
                        neg_record[col] = row_dict[col]

                ltr_records.append(neg_record)

    ltr_df = pd.DataFrame(ltr_records)

    # --- Step B2: Build LLM Dataset (Q + ground-truth context -> answer) --- #
    # We'll keep only positive Q–C pairs from the original df to generate (input_text, target_text).
    # You can add metadata columns if you wish.

    # We'll rename "answer" to "target_text" 
    # and create "input_text" = question + special token + context
    llm_records = []
    for _, row in df.iterrows():
        question = row["question"]
        context = row["context"]
        answer = row["answer"]
        row_dict = row.to_dict()

        # Build input_text
        input_text = f"{question} [SEP] {context}"

        llm_record = {
            "input_text": input_text,
            "target_text": answer
        }
        # copy any metadata
        for col in row_dict:
            if col not in ["question", "context", "answer"]:
                llm_record[col] = row_dict[col]

        llm_records.append(llm_record)

    llm_df = pd.DataFrame(llm_records)

    return (ltr_df, llm_df)

def convert_numpy_to_list(obj: Any) -> Any:
    """
    Recursively converts NumPy arrays to Python lists in nested structures.
    
    Args:
        obj: The object to convert, which may contain NumPy arrays
        
    Returns:
        The same object with all NumPy arrays converted to Python lists
    """
    # If it's a NumPy array, convert to list
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # If it's a NumPy scalar, convert to appropriate Python type
    elif np.isscalar(obj) and isinstance(obj, (np.generic,)):
        return obj.item()
    # If it's a pandas Timestamp, convert to datetime
    elif isinstance(obj, pd.Timestamp):
        return obj.to_pydatetime()
    # If it's a dictionary, process all its values
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    # If it's a list or tuple, process all its items
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_list(item) for item in obj]
    # Otherwise, return as is
    else:
        return obj


@step
def store_and_log_results(
    ltr_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    log_wandb: bool = True
) -> None:
    """
    Stores the LTR & LLM dataframes  writes them to MongoDB,
    and logs them to W&B if desired.

    :param ltr_df: DataFrame with LTR data.
    :param llm_df: DataFrame with LLM data.
    :param log_wandb: Whether to log the results to Weights & Biases.
    """
    import os
    import wandb
    db = get_mongo_connection()
    

    # store in MongoDB

    # Insert into new collections
    ltr_collection = db["ltr_dataset"]
    llm_collection = db["llm_dataset"]

    # Convert DF to dict and insert
    ltr_docs = ltr_df.to_dict(orient="records")
    llm_docs = llm_df.to_dict(orient="records")
    
    # Convert NumPy arrays to Python lists before inserting into MongoDB
    ltr_docs = convert_numpy_to_list(ltr_docs)
    llm_docs = convert_numpy_to_list(llm_docs)
    
    ltr_collection.insert_many(ltr_docs)
    llm_collection.insert_many(llm_docs)
    print("✅ LTR and LLM datasets inserted into MongoDB")

    # Optionally log to W&B
    if log_wandb:
        data = [ltr_df, llm_df]
        names = ["ltr_df", "llm_df"]
        run = wandb.init(project="MediMaven-DataPrep", job_type="neg_sampling")
        # Log data as W&B Tables
        ltr_table = wandb.Table(dataframe=ltr_df)
        llm_table = wandb.Table(dataframe=llm_df)
        wandb.log({"LTR Dataset": ltr_table, "LLM Dataset": llm_table})
        
        ltr_artifact = wandb.Artifact("LTR_dataset", type="dataset", 
                                    description="Negative-sampled LTR dataset")
        llm_artifact = wandb.Artifact("LLM_dataset", type="dataset",
                                    description="Fine-tuning dataset (Q+context -> answer)")

        # 4. Add CSV files to the artifacts
        ltr_artifact.add(ltr_table,name='LTR_dataset' )
        llm_artifact.add(llm_table, name='LLM_dataset')

        # 5. Log (commit) artifacts
        wandb.log_artifact(ltr_artifact)
        wandb.log_artifact(llm_artifact)

        wandb.finish()
        print("✅ Logged and registered datasets to W&B")





# ------------------ THE PIPELINE ------------------ #
@pipeline
def neg_sampling_pipeline():
    """
    ZenML pipeline that:
    1. Fetches Q-C-A data (with metadata) from MongoDB.
    2. Creates negative samples for LTR + LLM dataset.
    3. Stores or logs the output
    """
    ensure_mongodb_running()
    df = fetch_data_from_mongo(collection_name='qa_master_processed')
    ltr_df, llm_df = create_ltr_and_llm_datasets(df)
    store_and_log_results(ltr_df, llm_df)
    # The pipeline ends by returning these dataframes.
    # If we want to store them, you can add more steps, or
    # you can store them in the same step.
    return ltr_df, llm_df
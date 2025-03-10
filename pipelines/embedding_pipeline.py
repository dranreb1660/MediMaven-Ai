"""
ZenML pipeline that:
1. Fetches context passages (and optionally questions) from MongoDB or CSV.
2. Generates embeddings with a HuggingFace model.
3. Builds a FAISS index and saves it to disk.

We'll store references to each context so we can map FAISS IDs -> actual text.
"""

import os, json, time, faiss, wandb, torch
import numpy as np
import pandas as pd
from typing import Tuple, List, Any
from zenml.pipelines import pipeline
from zenml.steps import step
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient


from src.utils import get_mongo_connection, ensure_mongodb_running

model_name = "sentence-transformers/all-MiniLM-L6-v2"

ensure_mongodb_running()
db = get_mongo_connection()
# W&B run will be initialized inside the pipeline function

def get_ltr_mean_embeddings(df, embed_model, tokenizer, col ):
    print(f'-----embedding ltr {col}s  ')
    text_embeddings =[]
    unique_questions = df[[col+"_id", col]]
    for _, row in unique_questions.iterrows():
        _id = row[col+"_id"]
        question_text = row[col]
        # Possibly chunk the question if it’s also large
        chunks = chunk_text_by_tokens(question_text, tokenizer, max_tokens=300, overlap=50)
        emb_list = embed_model.encode(chunks, show_progress_bar=False)
        # Average chunk embeddings
        emb_avg = np.mean(emb_list, axis=0)
        text_embeddings.append(emb_avg)
    if df.shape[0] == len(text_embeddings):
        df[col+"_embeddings"] = text_embeddings
        print(f"added {len(text_embeddings)} to {col} in ltr_df")

    else:
        print(f"couldnt add {col} embeddings because len_df:{df.shape[0]} is diff from len_tex_embeddings: {len(text_embeddings)}")
    

    return df

def chunk_text_by_tokens(text: str, tokenizer, max_tokens=300, overlap=50):
    encoded = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(encoded):
        end = start + max_tokens
        chunk_ids = encoded[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
        start += (max_tokens - overlap)
    return chunks  

# ------------------ STEP A: Fetch Contexts ------------------ #
@step
def fetch_llm_ltr_data(llm_cntx_collection_name: str, ltr_collection_name: str) ->Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Step A: Load distinct context passages from MongoDB or your final dataset.

    :param llm_cntx_collection_name: e.g., "ltr_dataset" or "cleaned_qa_data"
    :param ltr_cntx_collection_name: e.g., "ltr_dataset" or "cleaned_qa_data"

    :return: A  tuple of DataFrame llm and ltr dataframes
    """

    llm_collection = db[llm_cntx_collection_name]
    ltr_collection = db[ltr_collection_name]

    llm_data = list(llm_collection.find({}, {'_id': 0}))
    llm_df = pd.DataFrame(llm_data)

    ltr_data = list(ltr_collection.find({}, {'_id': 0}))
    ltr_df = pd.DataFrame(ltr_data)

    # We assume 'context' is the main text field we want to embed.
    # Optionally we generate a 'context_id'.
    if "context_id" not in llm_df.columns:
        llm_df["context_id"] = llm_df.index.astype(str)  # or some unique ID from the data

    # Drop duplicates if necessary
    llm_df = llm_df.drop_duplicates(subset=["context"]).reset_index(drop=True)
    ltr_df = ltr_df.drop_duplicates(subset=["context"]).reset_index(drop=True)




    # Return only relevant columns
    return (llm_df , ltr_df)


# ------------------ STEP B: Chunk Documents ------------------ #
@step
def chunk_llm_docs(df: pd.DataFrame, model_name: str) -> pd.DataFrame:    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    start_time = time.time()

    all_rows = []
    for _, row in df.iterrows():
        original_text = row["context"]
        chunked_texts = chunk_text_by_tokens(original_text, tokenizer, overlap=50)
        for idx, chunk in enumerate(chunked_texts):
            new_row = row.copy()
            new_row["context"] = chunk
            new_row["chunk_id"] = f"{row['context_id']}_{idx}"
            all_rows.append(new_row)

    elapsed = time.time() - start_time

    # (Optional) Log to W&B
    wandb.init(project="MediMaven-Embedding", job_type="embedding_pipeline", reinit=True)
    wandb.log({
        "embedding_model": model_name,
        "num_original_docs": df.shape[0],
        "embedding_dim": len(all_rows),
        "chunking_time_sec": elapsed
    })

    return pd.DataFrame(all_rows)


# ------------------ STEP C: Generate Embeddings ------------------ #
@step
def generate_embeddings(
    llm_df: pd.DataFrame,
    ltr_df: pd.DataFrame,
    model_name = 'sentence-transformers/all-MiniLM-L6-v2',
    batch_size: int = 32,
    device_name: str = "mps"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Step C: Use a Transformer model to embed each llm context passage and ltr question and context.

    :param llm_df: DataFrame with "context_id" and "context".
    :param ltr_df: DataFrame with "context_id" and "context".
    :param batch_size: Batch size for encoding.
    :return: (llm_df, embeddings, ltr_df)
        llm_df: the same DataFrame with the embs (for reference)
        embeddings: a numpy array of shape (num_contexts, embedding_dim).
        ltr_df: the same DataFrame with the embs (for reference)

    """
    print(f"Use pytorch device_name: {device_name}")
    model = SentenceTransformer(model_name, device=device_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    llm_emb_start_time = time.time()

    llm_contexts = llm_df["context"].tolist()
    llm_embeddings = model.encode(llm_contexts, batch_size=batch_size, show_progress_bar=True)
    # Convert to list of arrays for pandas storage
    llm_df['embeddings'] = [emb.tolist() for emb in llm_embeddings]
    # Store the shape for later reconstruction
    llm_df.attrs['embedding_shape'] = llm_embeddings.shape

    llm_elapsed = time.time() - llm_emb_start_time

    # Create IDs for grouping
    if "question_id" not in ltr_df.columns:
        ltr_df["question_id"] = ltr_df["question"].factorize()[0]
    if "context_id" not in ltr_df.columns:
        ltr_df["context_id"] = ltr_df["context"].factorize()[0]

    ltr_emb_start = time.time()
    ltr_df = get_ltr_mean_embeddings(ltr_df, model,tokenizer, "question" )
    ltr_df = get_ltr_mean_embeddings(ltr_df, model,tokenizer, "context" )
    ltr_elapsed = time.time() -  ltr_emb_start    

    wandb.init(resume="auto")
    wandb.log({
        "embedding_model": model_name,
        "num_llm_texts_embedded": len(llm_contexts),
        "embedding_dim": llm_embeddings.shape[1],
        "llm_embedding_time_sec": llm_elapsed,
        "num_ltr_texts_embedded": ltr_df.shape[0],
        "ltr_embedding_time_sec": ltr_elapsed
    })

    return (llm_df, ltr_df)


# ------------------ STEP D: Build FAISS Index ------------------ #
@step
def build_faiss_index(
    df: pd.DataFrame,
    index_save_path: str = "./models/faiss_index.bin"
) -> faiss.Index:
    """
    Step D: Build a FAISS index from the embeddings and save it to disk.

    :param df: DataFrame with "context_id" for referencing each vector.
    :param embeddings: A 2D numpy array of shape (num_contexts, embedding_dim).
    :param index_save_path: Where to save the FAISS index file.
    """
    # Reconstruct the numpy array from the list of embeddings in the DataFrame
    embeddings = np.array([np.array(emb) for emb in df['embeddings']]).astype("float32")
    start_time = time.time()
    
    num_vectors, embedding_dim = embeddings.shape

    # 1. Create a FAISS index
    # For small-medium data, a simple IndexFlatL2 is fine.
    index = faiss.IndexFlatL2(embedding_dim)

    # 2. Add embeddings to the index
    index.add(embeddings)

    # 3. Save index to disk
    faiss.write_index(index, index_save_path)

    elapsed = time.time() - start_time
    # if wandb.run is not None and (wandb.run.resumed or hasattr(wandb.run, 'id') and wandb.run.id):
    # wandb.init(resume="auto")
    wandb.init(resume="auto")
    wandb.log({
        "faiss_index_type": "IndexFlatL2",
        "num_vectors": num_vectors,
        "build_index_time_sec": elapsed
    })

    print(f"✅ FAISS index (IndexFlatL2) saved to: {index_save_path}")

    # 4. Also store a mapping from index ID -> context_id in a separate file
    # so we can look up the actual text later.
    # We'll store it as a CSV with two columns: [faiss_id, context_id]
    # Here, faiss_id is just the row index in the data.
    mapping_df = pd.DataFrame({
        "faiss_id": range(num_vectors),
        "context_id": df["context_id"].tolist()
    })
    mapping_df = pd.DataFrame({
        "faiss_id": range(num_vectors),
        "context_id": df["context_id"].tolist()
        })
    mapping_dict = mapping_df.to_dict(orient='records')
    mapping_path = index_save_path.replace(".bin", "_mapping.json")

    with open(mapping_path, 'w') as f:
        json.dump(mapping_dict, f)
    print(f"✅ Mapping file saved to: {mapping_path}")

    return index


def convert_numpy_to_list(docs):
    """
    Convert NumPy arrays in documents to Python lists for MongoDB storage.
    
    Args:
        docs: List of dictionaries (documents) that may contain NumPy arrays
        
    Returns:
        List of dictionaries with NumPy arrays converted to Python lists
    """
    for doc in docs:
        for key, value in doc.items():
            if isinstance(value, np.ndarray):
                doc[key] = value.tolist()
    return docs


@step
def store_and_log_emb_results(
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
    db = get_mongo_connection()
    

    # store in MongoDB

    # Insert into new collections
    ltr_collection = db["ltr_emb_dataset"]
    llm_collection = db["llm_emb_dataset"]

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
        wandb.init(resume="auto")
        # Log data as W&B Tables
        ltr_table = wandb.Table(dataframe=ltr_df)
        llm_table = wandb.Table(dataframe=llm_df)
        wandb.log({"LTR Dataset": ltr_table, "LLM Dataset": llm_table})
        
        ltr_artifact = wandb.Artifact("LTR_emb_dataset", type="dataset", 
                                    description="LTR dataset with Q and C embeddings")
        llm_artifact = wandb.Artifact("LLM_emb_dataset", type="dataset",
                                    description="Fine-tuning dataset (Q+context -> answer) context embeddings")

        # 4. Add CSV files to the artifacts
        ltr_artifact.add(ltr_table,name='LTR_emb_dataset' )
        llm_artifact.add(llm_table, name='LLM_emb_dataset')

        # 5. Log (commit) artifacts
        wandb.log_artifact(ltr_artifact)
        wandb.log_artifact(llm_artifact)

        print("✅ Logged and registered datasets to W&B")




# ------------------ DEFINE THE PIPELINE ------------------ #
@pipeline
def embedding_pipeline():
    """
    ZenML pipeline that:
    1. Fetches context passages from MongoDB.
    2. use model's token to chunk documents into multiple chunks.
    3. Generates embeddings via a Transformer model.
    4. Builds and saves a FAISS index.
    """
    # Initialize wandb run inside the pipeline function
    # wandb.init(project="MediMaven-Embedding", job_type="embedding_pipeline", reinit=True)

    try:
        llm_df, ltr_df = fetch_llm_ltr_data('qa_master_processed','ltr_dataset' )
        llm_df = chunk_llm_docs(llm_df, model_name=model_name)    

        llm_df, ltr_df = generate_embeddings(llm_df, ltr_df)
        build_faiss_index(llm_df)
        store_and_log_emb_results(ltr_df, llm_df)
    finally:
        # Always make sure to finish the run, even if there's an exception
        if wandb.run is not None:
            wandb.finish()  # Close the W&B run


import os
import subprocess
from pymongo import MongoClient
import numpy as np

db_uri = "mongodb+srv://dranreb1660:LUV160%40me.@cluster0.lxppn.mongodb.net/?retryWrites=true&w=majority&maxPoolSize=50&connectTimeoutMS=60000&socketTimeoutMS=120000"
db_name = "medimaven_db"

def ensure_mongodb_running():
    """Checks if MongoDB is running, and starts it if not."""
    try:
        # Try connecting to MongoDB
        subprocess.run(["mongosh", "--eval", "db.runCommand({ ping: 1 })"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print("✅ MongoDB is already running.")
    except subprocess.CalledProcessError:
        print("⚠️ MongoDB is NOT running. Attempting to start it...")
        os.system("brew services start mongodb-community")
        print("✅ MongoDB is running now!!!")


def get_mongo_connection(mongo_uri:str = db_uri, db_name:str = db_name):
    client = MongoClient(mongo_uri)
    db = client[db_name]

    return db

# -------------- HELPER: Cosine Similarity & L2 Distance -------------- #

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot_val = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_val / (norm_a * norm_b + 1e-9)

def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)



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


def chunk_text_by_tokens(text: str, tokenizer, max_tokens=512, overlap=256):
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


# -------------- MongoDB Batch Operations -------------- #

def insert_documents_in_batches(collection, documents, batch_size=1000):
    """
    Insert documents into MongoDB in batches to prevent timeouts with large datasets.
    
    Args:
        collection: MongoDB collection to insert into
        documents: List of documents to insert
        batch_size: Number of documents per batch (default: 1000)
    
    Returns:
        total_inserted: Total number of documents inserted
    """
    total_docs = len(documents)
    total_inserted = 0
    
    print(f"Inserting {total_docs} documents in batches of {batch_size}...")
    
    for i in range(0, total_docs, batch_size):
        end_idx = min(i + batch_size, total_docs)
        batch = documents[i:end_idx]
        if batch:
            collection.insert_many(batch)
            total_inserted += len(batch)
            print(f"Inserted batch {i//batch_size + 1}: {len(batch)} documents ({total_inserted}/{total_docs})")
    
    print(f"✅ Successfully inserted {total_inserted} documents in {(total_docs + batch_size - 1) // batch_size} batches")
    return total_inserted


import torch

def get_device():
    """
    Get the most suitable device for PyTorch operations with proper error handling.
    Returns: torch.device
    """
    try:
        # Stage 1: Check for CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"Using CUDA (NVIDIA GPU): {torch.cuda.get_device_name(0)}")
            return device
        
        # Stage 2: Check for Apple Silicon GPU (MPS)
        if torch.backends.mps.is_available(): 
            device = "mps"
            print("Using Apple Silicon GPU (MPS)")
            return device
        
        # Stage 3: Fall back to CPU
        device = 'cpu'
        print("Using CPU")
        return device
        
    except Exception as e:
        print(f"Error detecting device, falling back to CPU: {str(e)}")
        return torch.device("cpu")

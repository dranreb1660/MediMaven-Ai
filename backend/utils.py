import os
import subprocess
# from pymongo import MongoClient
import numpy as np
from backend.app import config

# db_uri = config.MONGO_URL

# def ensure_mongodb_running():
#     """Checks if MongoDB is running, and starts it if not."""
#     try:
#         # Try connecting to MongoDB
#         subprocess.run(["mongosh", "--eval", "db.runCommand({ ping: 1 })"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
#         print("✅ MongoDB is already running.")
#     except subprocess.CalledProcessError:
#         print("⚠️ MongoDB is NOT running. Attempting to start it...")
#         os.system("brew services start mongodb-community")
#         print("✅ MongoDB is running now!!!")


# def get_mongo_connection(mongo_uri:str = db_uri, db_name:str = db_name):
#     client = MongoClient(mongo_uri)
#     db = client[db_name]

#     return db

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




import torch

def get_device():
    """Get best available device for PyTorch operations"""
    try:
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
            return device
        
        if torch.backends.mps.is_available(): 
            device = "mps"
            print("Using Apple Silicon GPU")
            return device
        
        device = 'cpu'
        print("Using CPU")
        return device
        
    except Exception as e:
        print(f"Device detection failed, using CPU: {str(e)}")
        return torch.device("cpu")

def clean_response(text: str) -> str:
    if "." in text:
        return text[:text.rfind(".")+1]  # cut off at last full stop
    return text  # fallback if no period found


from pathlib import Path
import hashlib, time, json, torch
import logging

def file_sha(path: Path) -> str:
    return hashlib.md5(open(path, "rb").read()).hexdigest()

class Timer:
    """Simple timer for measuring elapsed time"""
    __slots__ = ('_start',)
    
    def __init__(self):
        self._start = time.perf_counter()
    
    def elapsed(self):
        """Get elapsed seconds"""
        return time.perf_counter() - self._start

def best_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def get_logger(name: str):
    """Get a configured logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

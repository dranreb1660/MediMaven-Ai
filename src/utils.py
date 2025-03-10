import os
import subprocess
from pymongo import MongoClient
import numpy as np

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


def get_mongo_connection(mongo_uri:str = "mongodb://localhost:27017/", db_name:str = "medimaven_db"):
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
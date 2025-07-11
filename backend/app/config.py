# from pydantic_settings import BaseSettings
from pathlib import Path
import os


from pathlib import Path
import os

ENABLE_MONITORING = os.getenv("ENABLE_MONITORING", True)
ENABLE_CACHING = os.getenv("ENABLE_CACHING", False)
REDIS_URL = os.getenv("REDIS_URL", None)  # e.g. redis://localhost:6379/0

# Medical retrieval parameters
BM25_MEDICAL_K1 = 1.6              # Tuned for clinical term distribution
BM25_MEDICAL_B = 0.85              # Medical document length normalization
MEDICAL_HNSW_EF = 128              # 128-256 for clinical accuracy
QDRANT_QUANTIZATION = True         # Enable for 2-4x speedup
QDRANT_UPLOAD_THREADS = 8          # Match CPU cores


# ── Project root (resolve once) ─────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]          # …/backend/app → project

# ── Data and model artefacts ───────────────────────────────────────────────
DATA_DIR   = Path(os.getenv("DATA_DIR",   ROOT / "data" / "final"))
MODEL_DIR  = Path(os.getenv("MODEL_DIR",  ROOT / "models" / "v1.1"))

MONGO_URL = os.getenv("MONGO_URL", " ")  # Default MongoDB URL

# Retrieval
QDRANT_CLOUD_URL = os.getenv("QDRANT_CLOUD_URL", "https://cloud.qdrant.io")

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", " ")

QDRANT_DIR = MODEL_DIR / "qdrant"
EMB_NPY    = DATA_DIR / "chunk_embeddings.npy"
META_PQT   = DATA_DIR / "kb_chunks.parquet"
QCOLL      = os.getenv("QDRANT_COLLECTION", "medimaven_chunks")
BM25_PKL   = MODEL_DIR / "bm25" / "bm25.pkl"
EMBED_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"


# Ranking
LAMBDA_PATH  = MODEL_DIR / "ltr_lambdamart" / "ltr_lambdamart.txt"
CE_PATH    = MODEL_DIR / "ltr_cross-encoder"

# LLM weights
FP16_DIR   = MODEL_DIR / "llama3_8b_fp16"
AWQ_DIR    = MODEL_DIR / "llama3_8b_awq"

# Threshold (GiB) above which we prefer fp16
MIN_FP16_GIB = float(os.getenv("MIN_FP16_GIB", 22))

# Front-end CORS
ALLOWED_ORIGINS = [
    "https://www.medimaven-ai.com",
    "https://medimaven-ai.com",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


# ── Ensure directories exist ───────────────────────────────────────────────
if __name__ == "__main__":
    print("Project Root:", ROOT)
    print("Data Directory:", DATA_DIR)
    print("Model Directory:", MODEL_DIR)
    # for d in [DATA_DIR, MODEL_DIR, QDRANT_DIR]:
    #     if not d.exists():
    #         print(f"Creating directory: {d}")
    #         d.mkdir(parents=True, exist_ok=True)
    #     else:
    #         print(f"Directory already exists: {d}")

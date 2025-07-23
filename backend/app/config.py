from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True, verbose=True)

# Environment detection
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
IS_PRODUCTION = ENVIRONMENT == "production"

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO" if IS_PRODUCTION else "DEBUG")

# Feature flags
ENABLE_MONITORING = os.getenv("ENABLE_MONITORING", "true").lower() == "true"
ENABLE_CACHING = os.getenv("ENABLE_CACHING", "true").lower() == "true"
REDIS_URL = os.getenv("REDIS_URL", None)

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:Luv160%40me.@db.lrutkocupxqczzdtshkw.supabase.co:5432/postgres",
)

# Database pool settings (production optimized)
DATABASE_POOL_SIZE = int(os.getenv("DATABASE_POOL_SIZE", "10" if IS_PRODUCTION else "5"))
DATABASE_MAX_OVERFLOW = int(os.getenv("DATABASE_MAX_OVERFLOW", "20" if IS_PRODUCTION else "10"))
DATABASE_POOL_TIMEOUT = int(os.getenv("DATABASE_POOL_TIMEOUT", "30"))
DATABASE_POOL_RECYCLE = int(os.getenv("DATABASE_POOL_RECYCLE", "3600"))

# Authentication
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN", "medimaven-dev.us.auth0.com")
AUTH0_AUDIENCE = os.getenv("AUTH0_AUDIENCE", "https://api.medimaven-ai.com")

# Medical retrieval parameters (environment-specific)
if IS_PRODUCTION:
    # Conservative production settings
    BM25_MEDICAL_K1 = float(os.getenv("BM25_MEDICAL_K1", "1.2"))
    BM25_MEDICAL_B = float(os.getenv("BM25_MEDICAL_B", "0.75"))
    MEDICAL_HNSW_EF = int(os.getenv("MEDICAL_HNSW_EF", "64"))
    QDRANT_QUANTIZATION = os.getenv("QDRANT_QUANTIZATION", "true").lower() == "true"
    MIN_FP16_GIB = float(os.getenv("MIN_FP16_GIB", "24"))
else:
    # Development settings
    BM25_MEDICAL_K1 = float(os.getenv("BM25_MEDICAL_K1", "1.6"))
    BM25_MEDICAL_B = float(os.getenv("BM25_MEDICAL_B", "0.85"))
    MEDICAL_HNSW_EF = int(os.getenv("MEDICAL_HNSW_EF", "128"))
    QDRANT_QUANTIZATION = os.getenv("QDRANT_QUANTIZATION", "false").lower() == "true"
    MIN_FP16_GIB = float(os.getenv("MIN_FP16_GIB", "22"))

# Qdrant configuration
QDRANT_UPLOAD_THREADS = min(int(os.getenv("QDRANT_UPLOAD_THREADS", "4")), 8)
QDRANT_CLOUD_URL = os.getenv("QDRANT_CLOUD_URL", "https://cloud.qdrant.io")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

# Project paths (validated)
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data" / "final"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models" / "v1.1"))

# Validate critical paths exist
def validate_paths():
    """Validate that critical paths exist"""
    critical_paths = [DATA_DIR, MODEL_DIR]
    missing_paths = [p for p in critical_paths if not p.exists()]
    
    if missing_paths:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Missing paths: {missing_paths}")
        
        # Create directories if possible
        for path in missing_paths:
            try:
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {path}")
            except Exception as e:
                logger.error(f"Failed to create {path}: {e}")

# Call validation
validate_paths()

# Data and model file paths
QDRANT_DIR = MODEL_DIR / "qdrant"
EMB_NPY = DATA_DIR / "chunk_embeddings.npy"
META_PQT = DATA_DIR / "kb_chunks.parquet"
QCOLL = os.getenv("QDRANT_COLLECTION", "medimaven_chunks")
BM25_PKL = MODEL_DIR / "bm25" / "bm25.pkl"
EMBED_NAME = os.getenv("EMBED_MODEL", "pritamdeka/S-PubMedBert-MS-MARCO")

# Ranking model paths
LAMBDA_PATH = MODEL_DIR / "ltr_lambdamart" / "ltr_lambdamart.txt"
CE_PATH = MODEL_DIR / "ltr_cross-encoder"

# LLM weight paths
FP16_DIR = MODEL_DIR / "llama3_8b_fp16"
AWQ_DIR = MODEL_DIR / "llama3_8b_awq"

# CORS configuration
if IS_PRODUCTION:
    # Restrict CORS in production
    ALLOWED_ORIGINS = [
        origin.strip() 
        for origin in os.getenv("ALLOWED_ORIGINS", "").split(",") 
        if origin.strip()
    ]
    if not ALLOWED_ORIGINS:
        ALLOWED_ORIGINS = [
            "https://www.medimaven-ai.com",
            "https://medimaven-ai.com"
        ]
else:
    # Permissive CORS for development
    ALLOWED_ORIGINS = ["*"]

# Resource limits
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "50" if IS_PRODUCTION else "20"))
MAX_REQUEST_SIZE = int(os.getenv("MAX_REQUEST_SIZE", "1048576"))  # 1MB
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60" if IS_PRODUCTION else "120"))

# Cache configuration
CACHE_DEFAULT_TTL = int(os.getenv("CACHE_DEFAULT_TTL", "300"))
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))

# Model performance settings
GPU_MEMORY_FRACTION = float(os.getenv("GPU_MEMORY_FRACTION", "0.75" if IS_PRODUCTION else "0.8"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "16" if IS_PRODUCTION else "32"))

# Monitoring and health check settings
HEALTH_CHECK_TIMEOUT = int(os.getenv("HEALTH_CHECK_TIMEOUT", "10"))
METRICS_ENABLED = os.getenv("METRICS_ENABLED", "true" if IS_PRODUCTION else "false").lower() == "true"

# Ensure required directories exist
def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        QDRANT_DIR,
        BM25_PKL.parent,
        LAMBDA_PATH.parent,
    ]
    
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Could not create directory {directory}: {e}")

# Call directory creation
ensure_directories()

# Configuration summary for debugging
def get_config_summary():
    """Get configuration summary for debugging"""
    return {
        "environment": ENVIRONMENT,
        "is_production": IS_PRODUCTION,
        "log_level": LOG_LEVEL,
        "enable_monitoring": ENABLE_MONITORING,
        "enable_caching": ENABLE_CACHING,
        "redis_configured": bool(REDIS_URL),
        "data_dir": str(DATA_DIR),
        "model_dir": str(MODEL_DIR),
        "qdrant_quantization": QDRANT_QUANTIZATION,
        "allowed_origins": len(ALLOWED_ORIGINS),
        "gpu_memory_fraction": GPU_MEMORY_FRACTION,
    }

# Validation function for startup
def validate_config():
    """Validate configuration for startup"""
    issues = []
    
    # Check critical files
    critical_files = [EMB_NPY, META_PQT]
    for file_path in critical_files:
        if not file_path.exists():
            issues.append(f"Missing critical file: {file_path}")
    
    # Check database URL
    if not DATABASE_URL:
        issues.append("DATABASE_URL not configured")
    
    # Check model directories
    if not FP16_DIR.exists() and not AWQ_DIR.exists():
        issues.append("No model weight directories found")
    
    if issues:
        import logging
        logger = logging.getLogger(__name__)
        logger.error("Configuration validation failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False
    
    return True

if __name__ == "__main__":
    print("MediMaven Configuration")
    print("=" * 40)
    
    summary = get_config_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\nValidation:", "✅ PASSED" if validate_config() else "❌ FAILED")
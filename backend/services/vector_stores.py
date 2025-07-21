from __future__ import annotations
from pathlib import Path
import pickle, json, numpy as np, pandas as pd, time, torch, logging
from typing import List, Optional, Dict
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient, models as qm
from sentence_transformers import SentenceTransformer
from backend.utils import file_sha, get_logger
from backend.app import config

logger = get_logger(__name__)

class BM25Store:
    """Production-ready BM25 store with robust error handling"""
    
    def __init__(self, pkl_path: Path, chunks_df: pd.DataFrame, force_rebuild=False):
        self.pkl_path = pkl_path
        self.medical_terms = self._load_medical_terms()
        
        try:
            if not force_rebuild and pkl_path.exists():
                self._load_from_pickle()
            else:
                self._build_index(chunks_df)
                
            logger.info(f"✅ BM25Store ready with {len(self.ids)} documents")
            
        except Exception as e:
            logger.error(f"BM25Store initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize BM25 store: {e}")
    
    def _load_medical_terms(self) -> set:
        """Load medical terminology for enhanced processing"""
        return {
            "diabetes", "myocardial", "infarction", "hypertension", 
            "carcinoma", "pneumonia", "bronchitis", "arthritis"
        }
    
    def _fast_medical_tokenize(self, text: str) -> List[str]:
        """Fast tokenization optimized for medical text"""
        if not text:
            return []
        
        try:
            # Simple but effective tokenization
            text = text.lower()
            # Remove punctuation but keep hyphens in medical terms
            text = ''.join(c if c.isalnum() or c.isspace() or c == '-' else ' ' for c in text)
            tokens = [word for word in text.split() if len(word) > 1]
            return tokens
            
        except Exception as e:
            logger.warning(f"Tokenization failed: {e}")
            return text.split() if text else []
    
    def _build_index(self, chunks_df: pd.DataFrame):
        """Build BM25 index with progress tracking"""
        if chunks_df.empty:
            raise ValueError("Cannot build index from empty dataframe")
        
        logger.info(f"⏳ Building BM25 index for {len(chunks_df)} documents...")
        start_time = time.time()
        
        try:
            # Tokenize all documents
            tokenized = []
            for text in chunks_df["text"]:
                tokens = self._fast_medical_tokenize(str(text))
                tokenized.append(tokens)
            
            # Build BM25 index
            self.bm = BM25Okapi(
                tokenized, 
                k1=config.BM25_MEDICAL_K1, 
                b=config.BM25_MEDICAL_B
            )
            
            # Store document data
            self.ids = chunks_df["id"].tolist()
            self.docs = chunks_df["text"].tolist()
            self.urls = chunks_df.get("url", [""] * len(chunks_df)).tolist()
            self.sources = chunks_df.get("source", [""] * len(chunks_df)).tolist()
            
            # Save to pickle
            self._save_to_pickle()
            
            elapsed = time.time() - start_time
            logger.info(f"✅ BM25 index built in {elapsed:.1f}s")
            
        except Exception as e:
            logger.error(f"Index building failed: {e}")
            raise
    
    def _save_to_pickle(self):
        """Save index to pickle file with error handling"""
        try:
            self.pkl_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "bm": self.bm,
                "ids": self.ids,
                "docs": self.docs,
                "urls": self.urls,
                "sources": self.sources
            }
            
            with open(self.pkl_path, "wb") as f:
                pickle.dump(data, f)
                
        except Exception as e:
            logger.error(f"Failed to save BM25 index: {e}")
            raise
    
    def _load_from_pickle(self):
        """Load index from pickle with validation"""
        try:
            with open(self.pkl_path, "rb") as f:
                data = pickle.load(f)
            
            # Validate required keys
            required_keys = ["bm", "ids", "docs", "urls", "sources"]
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                raise ValueError(f"Corrupted pickle file, missing keys: {missing_keys}")
            
            # Load data
            for key in required_keys:
                setattr(self, key, data[key])
            
            logger.info(f"📁 Loaded BM25 index with {len(self.ids)} documents")
            
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            raise
    
    def search(self, query: str, top_k: int = 200) -> List[Dict]:
        """Search with error handling and result validation"""
        if not query or not query.strip():
            return []
        
        try:
            query_tokens = self._fast_medical_tokenize(query.lower())
            if not query_tokens:
                return []
            
            scores = self.bm.get_scores(query_tokens)
            
            # Handle edge cases
            if len(scores) == 0:
                return []
            
            # Get top results
            top_k = min(top_k, len(scores))
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            results = []
            for i in top_indices:
                if i < len(self.ids) and scores[i] > 0:  # Only include positive scores
                    results.append({
                        "id": self.ids[i],
                        "score": float(scores[i]),
                        "text": self.docs[i],
                        "url": self.urls[i] if i < len(self.urls) else "",
                        "source": self.sources[i] if i < len(self.sources) else ""
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

class QdrantStore:
    """Production-ready Qdrant store with connection pooling and retry logic"""
    
    def __init__(self, collection_name: str, embed_model_name: str, embeddings_path: Path,
                 metadata_path: Path, storage_path: Optional[Path] = None,
                 cloud_url: Optional[str] = None, api_key: Optional[str] = None,
                 use_cloud: bool = False, force_rebuild: bool = False):
        
        self.collection_name = collection_name
        self.embeddings_path = embeddings_path
        self.metadata_path = metadata_path
        self.max_retries = 3
        self.retry_delay = 1.0
        
        try:
            # Initialize client with retry logic
            self._init_client(use_cloud, cloud_url, api_key, storage_path)
            
            # Initialize embedding model
            self._init_embedding_model(embed_model_name)
            
            # Ensure collection exists
            self._ensure_collection(force_rebuild)
            
            logger.info("✅ QdrantStore initialized successfully")
            
        except Exception as e:
            logger.error(f"QdrantStore initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize Qdrant store: {e}")
    
    def _init_client(self, use_cloud: bool, cloud_url: str, api_key: str, storage_path: Path):
        """Initialize Qdrant client with retry logic"""
        for attempt in range(self.max_retries):
            try:
                if use_cloud:
                    if not cloud_url or not api_key:
                        raise ValueError("Cloud URL and API key required for cloud mode")
                    
                    self.client = QdrantClient(
                        url=cloud_url, 
                        api_key=api_key, 
                        prefer_grpc=True,
                        timeout=30
                    )
                    self.storage_type = "cloud"
                    logger.info("☁️ Connected to Qdrant Cloud")
                else:
                    self.storage_path = storage_path or config.QDRANT_DIR
                    self.storage_path.mkdir(parents=True, exist_ok=True)
                    
                    self.client = QdrantClient(
                        path=str(self.storage_path),
                        prefer_grpc=True
                    )
                    self.storage_type = "local"
                    logger.info(f"💻 Using local Qdrant storage: {self.storage_path}")
                
                # Test connection
                self.client.get_collections()
                break
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Failed to connect to Qdrant after {self.max_retries} attempts: {e}")
                
                time.sleep(self.retry_delay * (2 ** attempt))
                logger.warning(f"Qdrant connection attempt {attempt + 1} failed: {e}")
    
    def _init_embedding_model(self, embed_model_name: str):
        """Initialize embedding model with device detection"""
        try:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            
            self.embed_model = SentenceTransformer(embed_model_name, device=device)
            logger.info(f"⚙️ Embedding model loaded on: {device}")
            
        except Exception as e:
            logger.error(f"Embedding model initialization failed: {e}")
            raise
    
    def _ensure_collection(self, force_rebuild: bool):
        """Ensure collection exists with proper validation"""
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            collection_exists = self.collection_name in collections
            
            # Check if rebuild is needed
            need_rebuild = force_rebuild or not collection_exists
            
            if collection_exists and not force_rebuild:
                # Validate existing collection
                info = self.get_collection_info()
                if not info or info.get("points_count", 0) == 0:
                    logger.warning("Existing collection is empty, rebuilding...")
                    need_rebuild = True
            
            if need_rebuild:
                self.rebuild_collection()
            else:
                logger.info(f"✅ Using existing collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Collection validation failed: {e}")
            raise
    
    def rebuild_collection(self):
        """Rebuild collection with optimized settings"""
        logger.info(f"🏗️ Rebuilding collection: {self.collection_name}")
        
        try:
            # Load data
            vecs = np.load(self.embeddings_path)
            metas = pd.read_parquet(self.metadata_path)
            
            if len(vecs) != len(metas):
                raise ValueError(f"Vector count ({len(vecs)}) != metadata count ({len(metas)})")
            
            # Clean metadata
            metas = self._clean_metadata(metas)
            
            # Delete existing collection
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass  # Collection might not exist
            
            # Create collection with optimized settings
            self._create_collection(vecs.shape[1])
            
            # Upload in batches
            batch_size = 1000
            total = len(vecs)
            
            for i in range(0, total, batch_size):
                end_idx = min(i + batch_size, total)
                self._upload_batch(metas, vecs, i, end_idx)
                
                if (i + batch_size) % 5000 == 0 or end_idx == total:
                    logger.info(f"  ↑ Uploaded {end_idx}/{total} vectors")
            
            # Save verification data for local storage
            if self.storage_type == "local":
                self._save_verification_data()
            
            logger.info(f"✅ Collection rebuilt with {total} vectors")
            
        except Exception as e:
            logger.error(f"Collection rebuild failed: {e}")
            raise
    
    def _clean_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean metadata for Qdrant compatibility"""
        try:
            df = df.copy()
            
            # Ensure required columns exist
            for col in ["id", "text"]:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in metadata")
            
            # Clean string columns
            str_columns = ["url", "section", "source", "text"]
            for col in str_columns:
                if col in df.columns:
                    df[col] = df[col].astype(str).fillna("")
            
            return df
            
        except Exception as e:
            logger.error(f"Metadata cleaning failed: {e}")
            raise
    
    def _create_collection(self, vector_size: int):
        """Create collection with production-optimized settings"""
        quantization = None
        if config.QDRANT_QUANTIZATION:
            quantization = qm.ScalarQuantization(
                scalar=qm.ScalarQuantizationConfig(
                    type=qm.ScalarType.INT8, 
                    quantile=0.95
                )
            )
        
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qm.VectorParams(
                size=vector_size, 
                distance=qm.Distance.COSINE,
                on_disk=True
            ),
            hnsw_config=qm.HnswConfigDiff(
                m=16,
                ef_construct=150,
                full_scan_threshold=10000,
                max_indexing_threads=2,
                on_disk=False
            ),
            optimizers_config=qm.OptimizersConfigDiff(
                indexing_threshold=20000,
                max_optimization_threads=2
            ),
            quantization_config=quantization
        )
    
    def _upload_batch(self, metas: pd.DataFrame, vecs: np.ndarray, start: int, end: int):
        """Upload batch of vectors with error handling"""
        try:
            batch_slice = metas.iloc[start:end]
            batch_ids = batch_slice["id"].tolist()
            batch_vectors = vecs[start:end].astype(np.float32).tolist()
            
            # Create payloads with explicit typing
            batch_payloads = []
            for _, row in batch_slice.iterrows():
                payload = {
                    "url": str(row.get("url", "")),
                    "section": str(row.get("section", "")),
                    "source": str(row.get("source", "")),
                    "text": str(row.get("text", ""))[:2000],  # Limit text length
                }
                batch_payloads.append(payload)
            
            # Upload batch
            self.client.upsert(
                collection_name=self.collection_name,
                points=qm.Batch(
                    ids=batch_ids, 
                    vectors=batch_vectors, 
                    payloads=batch_payloads
                ),
                wait=True
            )
            
        except Exception as e:
            logger.error(f"Batch upload failed for range {start}-{end}: {e}")
            raise
    
    def _save_verification_data(self):
        """Save verification checksums for local storage"""
        try:
            chk_file = self.storage_path / f"{self.collection_name}_sha.json"
            verification_data = {
                "vec_sha": file_sha(self.embeddings_path),
                "meta_sha": file_sha(self.metadata_path),
                "timestamp": time.time()
            }
            
            with open(chk_file, "w") as f:
                json.dump(verification_data, f)
                
        except Exception as e:
            logger.warning(f"Failed to save verification data: {e}")
    
    def get_collection_info(self) -> Optional[Dict]:
        """Get collection information with error handling"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "points_count": info.points_count,
                "vectors_count": info.vectors_count or info.points_count,
                "indexed_vectors_count": getattr(info, 'indexed_vectors_count', None),
                "status": info.status.value if hasattr(info.status, 'value') else str(info.status),
                "optimizer_status": info.optimizer_status.value if hasattr(info.optimizer_status, 'value') else str(info.optimizer_status)
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return None
    
    def search(self, query: str, top_k: int = 300) -> List[Dict]:
        """Search with retry logic and error handling"""
        if not query or not query.strip():
            return []
        
        for attempt in range(self.max_retries):
            try:
                # Generate embedding
                emb = self.embed_model.encode([query], show_progress_bar=False)[0]
                
                # Perform search
                hits = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=emb,
                    limit=min(top_k, 1000),  # Limit for stability
                    search_params=qm.SearchParams(hnsw_ef=config.MEDICAL_HNSW_EF)
                )
                
                # Format results
                results = []
                for hit in hits:
                    result = {"id": hit.id, "score": float(hit.score)}
                    if hit.payload:
                        result.update(hit.payload)
                    results.append(result)
                
                return results
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Qdrant search failed after {self.max_retries} attempts: {e}")
                    return []
                
                time.sleep(self.retry_delay)
                logger.warning(f"Search attempt {attempt + 1} failed: {e}")
        
        return []
    
    def close(self):
        """Graceful shutdown with error handling"""
        try:
            if hasattr(self, 'client'):
                # Qdrant client doesn't always have a close method
                if hasattr(self.client, 'close'):
                    self.client.close()
                else:
                    # Just clear the reference
                    self.client = None
                    
        except Exception as e:
            logger.warning(f"Error during Qdrant close: {e}")

# Helper class for medical tokenization (simplified)
class MedicalTokenizer:
    """Lightweight medical tokenizer for production use"""
    
    def __init__(self):
        self.synonyms = {
            "mi": "myocardial infarction",
            "cad": "coronary artery disease", 
            "chf": "congestive heart failure",
            "t2d": "type 2 diabetes",
            "dm": "diabetes mellitus"
        }
    
    def tokenize(self, text: str) -> List[str]:
        """Simple but effective tokenization"""
        if not text:
            return []
        
        # Basic preprocessing
        text = text.lower()
        text = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text)
        tokens = [word for word in text.split() if len(word) > 1]
        return tokens
    
    def expand_query(self, query: str) -> str:
        """Expand medical abbreviations"""
        expanded = query
        for abbr, full in self.synonyms.items():
            if abbr in query.lower():
                expanded += f" {full}"
        return expanded

# src/backend/services/rag_modules/stores.py
from __future__ import annotations
from pathlib import Path
import pickle, json, numpy as np, pandas as pd, time, torch
from typing import List, Optional, Dict
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient, models as qm
from sentence_transformers import SentenceTransformer
from backend.utils import file_sha, get_logger
from backend.app import config
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

logger = get_logger(__name__)

class MedicalTokenizer:
    """Clinical text tokenization with UMLS-aware processing"""
    def __init__(self):
        self.nlp = self._load_spacy_model()
        self.synonyms = self._load_medical_synonyms()
    
    def _load_spacy_model(self):
        try:
            import spacy
            return spacy.load("en_core_web_sm")
        except:
            logger.warning("Using simple tokenizer")
            return None
    
    def _load_medical_synonyms(self):
        return {
            "mi": "myocardial infarction", "cad": "coronary artery disease",
            "chf": "congestive heart failure", "t2d": "type 2 diabetes", "dm": "diabetes mellitus"
        }
    
    def tokenize(self, text: str) -> List[str]:
        if not self.nlp:
            return text.split()
        doc = self.nlp(text.lower())
        return [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    
    def expand_query(self, query: str) -> str:
        for abbr, full in self.synonyms.items():
            if abbr in query.lower():
                query += f" {full}"
        return query

class BM25Store:
    """Optimized sparse retrieval for medical contexts"""
    def __init__(self, pkl_path: Path, chunks_df: pd.DataFrame, force_rebuild=False):
        self.pkl_path = pkl_path
        self.medical_terms = self._load_medical_terms()
        
        if not force_rebuild and pkl_path.exists():
            self._load_from_pickle()
        else:
            self._build_index(chunks_df)
    
    def _load_medical_terms(self):
        return {"diabetes", "myocardial", "infarction", "hypertension", "carcinoma"}
    
    def _fast_medical_tokenize(self, text: str) -> List[str]:
        text = text.lower().translate(str.maketrans('', '', '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))
        return [word for word in text.split() if len(word) > 1]
    
    def _build_index(self, chunks_df: pd.DataFrame):
        logger.info("⏳ Building BM25 index...")
        start = time.time()
        tokenised = [self._fast_medical_tokenize(t) for t in chunks_df["text"]]
        
        self.bm = BM25Okapi(tokenised, k1=config.BM25_MEDICAL_K1, b=config.BM25_MEDICAL_B)
        self.ids = chunks_df["id"].tolist()
        self.docs = chunks_df["text"].tolist()
        self.urls = chunks_df["url"].tolist()
        self.sources = chunks_df["source"].tolist()
        
        self.pkl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.pkl_path, "wb") as f:
            pickle.dump({
                "bm": self.bm, "ids": self.ids, "docs": self.docs,
                "urls": self.urls, "sources": self.sources
            }, f)
        
        logger.info(f"✅ BM25 built in {time.time()-start:.1f}s with {len(self.ids)} docs")
    
    def _load_from_pickle(self):
        with open(self.pkl_path, "rb") as f:
            data = pickle.load(f)
        for key in ["bm", "ids", "docs", "urls", "sources"]:
            setattr(self, key, data[key])
        self.medical_terms = self._load_medical_terms()
    
    def search(self, query: str, top_k: int = 200) -> List[Dict]:
        query_tokens = self._fast_medical_tokenize(query.lower())
        scores = self.bm.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [{
            "id": self.ids[i], "score": float(scores[i]),
            "text": self.docs[i], "url": self.urls[i], "source": self.sources[i]
        } for i in top_indices]

class QdrantStore:
    """Optimized vector store for medical retrieval"""
    def __init__(self, collection_name: str, embed_model_name: str, embeddings_path: Path,
                 metadata_path: Path, storage_path: Optional[Path] = None,
                 cloud_url: Optional[str] = None, api_key: Optional[str] = None,
                 use_cloud: bool = False, force_rebuild: bool = False):
        self.collection_name = collection_name
        self.embeddings_path = embeddings_path
        self.metadata_path = metadata_path
        
        # Initialize client
        if use_cloud:
            self.client = QdrantClient(url=cloud_url, api_key=api_key, prefer_grpc=True)
            self.storage_type = "cloud"
            logger.info("☁️ Connected to Qdrant Cloud")
        else:
            self.storage_path = storage_path or config.QDRANT_DIR
            self.client = QdrantClient(
                path=str(self.storage_path),
                prefer_grpc=True
            )
            self.storage_type = "local"
            logger.info("💻 Using local Qdrant storage")
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        # Initialize embedding model
        self.embed_model = SentenceTransformer(embed_model_name, device=device)
        logger.info(f"⚙️ Using device: {device}")

        # Build/verify collection
        self._ensure_collection(force_rebuild)

    def _ensure_collection(self, force_rebuild: bool):
        try:
            collections = self.client.get_collections().collections
            need_rebuild = force_rebuild or (self.collection_name not in [c.name for c in collections])
        except:
            need_rebuild = True
        
        if not need_rebuild and self.storage_type == "local":
            chk_file = self.storage_path / f"{self.collection_name}_sha.json"
            if chk_file.exists():
                with open(chk_file) as f:
                    if json.load(f) != {"vec_sha": file_sha(self.embeddings_path), 
                                        "meta_sha": file_sha(self.metadata_path)}:
                        need_rebuild = True
        
        if need_rebuild:
            self.rebuild_collection()
    
    def _clean_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure payload data types are Qdrant-compatible"""
        for col in df.columns:
            if col != "id" and df[col].dtype == object:
                df[col] = df[col].astype(str).fillna("")
        return df
    
    def rebuild_collection(self):
        logger.info(f"🏗️ Rebuilding collection: {self.collection_name}")
        vecs = np.load(self.embeddings_path)
        metas = self._clean_metadata(pd.read_parquet(self.metadata_path))
        total = len(vecs)
        
        # Delete existing collection
        try:
            self.client.delete_collection(self.collection_name)
        except:
            pass
        
        # Create new collection
        quantization = qm.ScalarQuantization(
            scalar=qm.ScalarQuantizationConfig(type=qm.ScalarType.INT8, quantile=0.95)
        ) if config.QDRANT_QUANTIZATION else None
        
        # Create collection with proper HNSW indexing configuration
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qm.VectorParams(
                size=vecs.shape[1], 
                distance=qm.Distance.COSINE, 
                on_disk=True
            ),
            hnsw_config=qm.HnswConfigDiff(
                m=16,
                ef_construct=150,
                full_scan_threshold=10000,
                max_indexing_threads=2,  # Enable indexing threads
                on_disk=False  # Keep HNSW index in memory for better performance
            ),
            optimizers_config=qm.OptimizersConfigDiff(
                indexing_threshold=20000,
                max_optimization_threads=2
            ),
            quantization_config=quantization
        )
        
        # Upload in batches
        batch_size = 1000
        for i in range(0, total, batch_size):
            self._upload_batch(metas, vecs, i, min(i+batch_size, total))
            logger.info(f"  ↑ Uploaded {min(i+batch_size, total)}/{total}")
        
        # Save verification data
        if self.storage_type == "local":
            chk_file = self.storage_path / f"{self.collection_name}_sha.json"
            with open(chk_file, "w") as f:
                json.dump({
                    "vec_sha": file_sha(self.embeddings_path),
                    "meta_sha": file_sha(self.metadata_path)
                }, f)
        
        logger.info(f"✅ Collection rebuilt with {total} vectors")
    
    def _upload_batch(self, metas: pd.DataFrame, vecs: np.ndarray, start: int, end: int):
        batch_ids = metas.iloc[start:end]["id"].tolist()
        batch_vectors = vecs[start:end].astype(np.float32).tolist()
        
        # Create payload with explicit typing
        batch_payloads = []
        for _, row in metas.iloc[start:end].iterrows():
            batch_payloads.append({
                "url": str(row.get("url", "")),
                "section": str(row.get("section", "")),
                "source": str(row.get("source", "")),
                "text": str(row.get("text", "")),
            })
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=qm.Batch(ids=batch_ids, vectors=batch_vectors, payloads=batch_payloads),
            wait=True
        )
            
    def get_collection_info(self) -> Dict:
        """Get information about the collection including point count"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "points_count": info.points_count,
                "vectors_count": info.vectors_count or info.points_count,  # Use points_count if vectors_count is None
                "indexed_vectors_count": getattr(info, 'indexed_vectors_count', None),
                "segments_count": getattr(info, 'segments_count', None),
                "status": info.status.value if hasattr(info.status, 'value') else str(info.status),
                "optimizer_status": info.optimizer_status.value if hasattr(info.optimizer_status, 'value') else str(info.optimizer_status)
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return None
            
    def search(self, query: str, top_k: int = 300) -> List[Dict]:
        emb = self.embed_model.encode([query], show_progress_bar=False)[0]
        try:
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=emb,
                limit=top_k,
                search_params=qm.SearchParams(hnsw_ef=config.MEDICAL_HNSW_EF)
            )
            return [{"id": h.id, "score": h.score, **h.payload} for h in hits]
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return []
    
    def close(self):
        """Graceful shutdown (call from FastAPI /startup)."""
        if hasattr(self.client, "close"):
            self.client.close()
        else:
            logger.warning("Qdrant client does not support close method")

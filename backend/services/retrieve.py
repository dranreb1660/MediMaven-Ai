from __future__ import annotations
import numpy as np, pandas as pd, weave, logging
import os
from typing import List, Dict
from backend.app import config
from backend.services.vector_stores import QdrantStore, BM25Store
import concurrent.futures
import asyncio
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Skip weave in CI unless we have the API key
if not os.getenv('CI') or os.getenv('WANDB_API_KEY'):
    try:
        weave.init("Medimaven-rag-production")
        logger.info("Weave initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize weave: {e}")
else:
    logger.info("Skipping weave initialization in CI environment")

class Retriever():
    """Production-ready hybrid retriever with robust error handling"""
    
    def __init__(self, qdrant_force_rebuild: bool = False, bm_force_rebuild: bool = False, 
                 use_cloud: bool = False, **data):
        self._initialized = False

        try:
            # Load metadata and embeddings
            self._load_data()
            
            # Set up BM25 and Qdrant stores with error handling
            self._init_stores(qdrant_force_rebuild, bm_force_rebuild, use_cloud)
            
            self._initialized = True
            logger.info("✅ Retriever initialized successfully")
            
        except Exception as e:
            logger.error(f"Retriever initialization failed: {e}")
            self._cleanup()
            raise RuntimeError(f"Failed to initialize retriever: {e}")
    
    def _load_data(self):
        """Load chunk metadata and embeddings from disk"""
        try:
            # Load chunk metadata
            if not config.META_PQT.exists():
                raise FileNotFoundError(f"Metadata file not found: {config.META_PQT}")
            
            self.chunks = pd.read_parquet(config.META_PQT)
            if self.chunks.empty:
                raise ValueError("Metadata file is empty")
            
            # Load embeddings
            if not config.EMB_NPY.exists():
                raise FileNotFoundError(f"Embeddings file not found: {config.EMB_NPY}")
            
            self.vecs = np.load(config.EMB_NPY)
            if self.vecs.size == 0:
                raise ValueError("Embeddings file is empty")
            
            # Make sure metadata and embeddings match up
            if len(self.chunks) != len(self.vecs):
                raise ValueError(f"Metadata ({len(self.chunks)}) and embeddings ({len(self.vecs)}) count mismatch")
            
            # Set up embedding model
            self.embed_model = self._build_embed(config.EMBED_NAME)
            
            logger.info(f"📚 Loaded {len(self.chunks)} chunks with {self.vecs.shape[1]}D embeddings")
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            raise
    
    def _init_stores(self, qdrant_force_rebuild: bool, bm_force_rebuild: bool, use_cloud: bool):
        """Set up BM25 and Qdrant search stores"""
        try:
            # Set up BM25 sparse search
            self.bm25_store = BM25Store(
                pkl_path=config.BM25_PKL,
                chunks_df=self.chunks,
                force_rebuild=bm_force_rebuild
            )
            
            # Set up Qdrant dense search
            self.qdrant_store = QdrantStore(
                collection_name=config.QCOLL,
                embed_model_name=config.EMBED_NAME,
                embeddings_path=config.EMB_NPY,
                metadata_path=config.META_PQT,
                storage_path=config.QDRANT_DIR,
                cloud_url=config.QDRANT_CLOUD_URL,
                api_key=config.QDRANT_API_KEY,
                use_cloud=use_cloud,
                force_rebuild=qdrant_force_rebuild
            )
            
            # Check that Qdrant is working
            qdrant_info = self.qdrant_store.get_collection_info()
            if not qdrant_info:
                raise RuntimeError("Qdrant store validation failed")
            
            logger.info(f"✅ Stores initialized - BM25: {len(self.bm25_store.ids)}, Qdrant: {qdrant_info.get('points_count', 0)}")
            
        except Exception as e:
            logger.error(f"Store initialization failed: {e}")
            raise
    
    @staticmethod
    def _build_embed(model_name: str):
        """Load embedding model on best available device"""
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            return SentenceTransformer(model_name, device=device)
        except Exception as e:
            logger.error(f"Embedding model initialization failed: {e}")
            raise
    
    def sparse_search(self, query: str, top_k: int = 200) -> List[Dict]:
        """BM25 search with error handling"""
        if not self._initialized:
            return []
        
        try:
            return self.bm25_store.search(query, top_k)
        except Exception as e:
            logger.error(f"Sparse search failed: {e}")
            return []
    
    def dense_search(self, query: str, top_k: int = 300) -> List[Dict]:
        """Vector search with error handling"""
        if not self._initialized:
            return []
        
        try:
            return self.qdrant_store.search(query, top_k)
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []
    
    def fuse(self, dense: List[Dict], sparse: List[Dict], k: int = 200, 
             k_rrf: int = 60, w_dense: float = 2.0, w_sparse: float = 1.0) -> List[Dict]:
        """Reciprocal Rank Fusion with error handling"""
        try:
            if not dense and not sparse:
                return []
            
            fused, scores = {}, {}
            
            def _add_result(item: Dict, channel: str, rank: int, weight: float):
                """Add item to fusion results"""
                item_id = item.get("id")
                if not item_id:
                    return
                
                if item_id not in fused:
                    fused[item_id] = {
                        "id": item_id,
                        "text": item.get("text", ""),
                        "url": item.get("url", ""),
                        "source": item.get("source", ""),
                        "dense_score": None, "dense_rank": None,
                        "bm25_score": None, "bm25_rank": None,
                    }
                
                fused[item_id][f"{channel}_score"] = item.get("score", 0.0)
                fused[item_id][f"{channel}_rank"] = rank
                scores[item_id] = scores.get(item_id, 0) + weight / (k_rrf + rank)
            
            # Add dense and sparse results to fusion
            for rank, item in enumerate(dense):
                _add_result(item, "dense", rank, w_dense)
            
            for rank, item in enumerate(sparse):
                _add_result(item, "bm25", rank, w_sparse)
            
            # Sort by RRF score
            result = [
                {**fused[item_id], "rrf_score": score}
                for item_id, score in scores.items()
            ]
            
            result.sort(key=lambda x: x["rrf_score"], reverse=True)
            return result[:k]
            
        except Exception as e:
            logger.error(f"Fusion failed: {e}")
            # Fallback to whatever we have
            return (dense or sparse)[:k]
    
    def retrieve(self, query: str, mode: str = "fuse") -> List[Dict]:
        """Main retrieval method - supports dense, sparse, or fused search"""
        if not self._initialized:
            return []
        
        if not query or not query.strip():
            return []
        
        try:
            if mode == "dense":
                return self.dense_search(query)
            elif mode == "sparse":
                return self.sparse_search(query)
            
            # Fused retrieval with parallel execution
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                dense_future = executor.submit(self.dense_search, query)
                sparse_future = executor.submit(self.sparse_search, query)
                
                try:
                    dense = dense_future.result(timeout=10.0)
                    sparse = sparse_future.result(timeout=10.0)
                except concurrent.futures.TimeoutError:
                    logger.warning("Retrieval timeout, using partial results")
                    dense = dense_future.result() if dense_future.done() else []
                    sparse = sparse_future.result() if sparse_future.done() else []
            
            return self.fuse(dense, sparse)
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
    
    @weave.op()
    async def aretrieve(self, query: str, mode: str = "fuse") -> List[Dict]:
        """Async retrieval wrapper"""
        if not query or not query.strip():
            return []
        
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self.retrieve, query, mode)
        except Exception as e:
            logger.error(f"Async retrieval failed: {e}")
            return []
    
    def _cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'qdrant_store'):
                self.qdrant_store.close()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    def close(self):
        """Public cleanup method"""
        self._cleanup()
        self._initialized = False

# Test functionality
if __name__ == "__main__":
    async def main():
        try:
            r = Retriever(
                qdrant_force_rebuild=False,
                bm_force_rebuild=False,
                use_cloud=False
            )
            
            # Test async retrieval
            results = await r.aretrieve("What causes type 2 diabetes?")
            print(f"✅ Async test: {len(results)} results")
            
            # Test sync retrieval
            results_sync = r.retrieve("What causes type 2 diabetes?")
            print(f"✅ Sync test: {len(results_sync)} results")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")

    asyncio.run(main())
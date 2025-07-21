from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import torch, os, numpy as np, weave, logging
import lightgbm as lgb
from sentence_transformers import CrossEncoder
import asyncio

from backend.services.retrieve import Retriever
from backend.app import config
from backend.utils import Timer

logger = logging.getLogger(__name__)

LAMBDA_PATH = config.LAMBDA_PATH
CE_PATH = config.CE_PATH

class Ranker():
    """Learning-to-Rank with simplified error handling and resource management"""
    
    def __init__(self, retriever: Retriever, lgbm_path=LAMBDA_PATH, ce_path=CE_PATH, **data):
        self.retriever = retriever
        self._initialized = False
        
        try:
            self._init_models(lgbm_path, ce_path)
            self._build_lookup_tables()
            self._initialized = True
            logger.info("✅ Ranker initialized successfully")
            
        except Exception as e:
            logger.error(f"Ranker initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize ranker: {e}")
    
    def _init_models(self, lgbm_path: Path, ce_path: Path):
        """Initialize ranking models with validation"""
        if not lgbm_path.exists():
            raise FileNotFoundError(f"LambdaMART model not found: {lgbm_path}")
        if not ce_path.exists():
            raise FileNotFoundError(f"Cross-encoder model not found: {ce_path}")
        
        # Load LambdaMART
        self.lgbm = lgb.Booster(model_file=str(lgbm_path))
        self.lgbm.params["n_jobs"] = max(1, min(os.cpu_count() - 1, 4))  # Limit threads
        
        # Load Cross-encoder with device detection
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ce = CrossEncoder(
            ce_path, device=device,
            tokenizer_kwargs={"padding": True, "truncation": True},
            model_kwargs={"torch_dtype": torch.float16} if device == "cuda" else {}
        )
    
    def _build_lookup_tables(self):
        """Build lookup tables with validation"""
        chunks = self.retriever.chunks
        if chunks.empty:
            raise ValueError("No chunks available for ranking")
        
        # ID to row mapping
        self.id2row = {cid: i for i, cid in enumerate(chunks["id"])}
        
        # Normalized vectors
        vecs = self.retriever.vecs
        if vecs.size == 0:
            raise ValueError("No vectors available for ranking")
            
        self.vec_norm = vecs.copy()
        norms = np.linalg.norm(self.vec_norm, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        self.vec_norm /= norms
        
        # Document lengths
        self.lengths = chunks["text"].str.split().str.len().fillna(0).to_numpy()
        
        # BM25 mapping
        self.bm_store = self.retriever.bm25_store
        self.bm25_ids = {cid: i for i, cid in enumerate(self.bm_store.ids)}
    
    def _feat_mat(self, query: str, q_emb: np.ndarray, docs: List[Dict]) -> np.ndarray:
        """Extract features with error handling"""
        try:
            ids = [d["id"] for d in docs]
            
            # Validate IDs exist in mappings
            valid_ids = [id for id in ids if id in self.id2row and id in self.bm25_ids]
            if not valid_ids:
                logger.warning("No valid IDs found for feature extraction")
                return np.zeros((len(docs), 3), dtype=np.float32)
            
            ridx = np.array([self.id2row[id] for id in valid_ids], dtype=np.int32)
            bidx = np.array([self.bm25_ids[id] for id in valid_ids], dtype=np.int32)
            
            # BM25 scores
            query_tokens = query.split()
            if not query_tokens:
                bm25_vec = np.zeros(len(valid_ids), dtype=np.float32)
            else:
                bm25_scores = self.bm_store.bm.get_scores(query_tokens)
                bm25_vec = bm25_scores[bidx].astype(np.float32)
            
            # Cosine similarity
            cos_vec = (self.vec_norm[ridx] @ q_emb).astype(np.float32)
            
            # Document lengths
            lens = self.lengths[ridx].astype(np.float32)
            
            # Handle mismatched lengths
            features = np.zeros((len(docs), 3), dtype=np.float32)
            features[:len(valid_ids)] = np.stack([bm25_vec, cos_vec, lens], axis=1)
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            # Return zero features as fallback
            return np.zeros((len(docs), 3), dtype=np.float32)
    
    def ltr_lamda(self, query: str, docs: List[Dict], top_k: int = 100) -> List[Dict]:
        """LambdaMART ranking with error handling"""
        if not self._initialized or not docs:
            return docs[:top_k]
        
        try:
            q_emb = self.retriever.embed_model.encode([query], show_progress_bar=False)[0]
            X = self._feat_mat(query, q_emb, docs)
            
            # Handle empty feature matrix
            if X.size == 0:
                logger.warning("Empty feature matrix, returning original order")
                return docs[:top_k]
            
            lgbm_scores = self.lgbm.predict(X)
            
            # Add scores to documents
            for i, (doc, score) in enumerate(zip(docs, lgbm_scores)):
                doc["ltr_score"] = float(score) if not np.isnan(score) else 0.0
            
            # Sort and return top k
            ranked = sorted(docs, key=lambda x: x.get("ltr_score", 0.0), reverse=True)
            return ranked[:top_k]
            
        except Exception as e:
            logger.error(f"LambdaMART ranking failed: {e}")
            return docs[:top_k]  # Fallback to original order
    
    def ltr_ce(self, query: str, docs: List[Dict], top_k: int = 20) -> List[Dict]:
        """Cross-encoder ranking with batching and error handling"""
        if not self._initialized or not docs:
            return docs[:top_k]
        
        try:
            # Limit input size for stability
            input_docs = docs[:min(len(docs), 100)]
            pairs = [[query, d.get('text', '')[:1000]] for d in input_docs]  # Truncate text
            
            # Batch size based on available memory
            batch_size = 16 if torch.cuda.is_available() else 8
            
            scores = self.ce.predict(
                pairs, 
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            # Add scores and handle NaN values
            for doc, score in zip(input_docs, scores):
                doc['ce_score'] = float(score) if not np.isnan(score) else 0.0
            
            # Sort and return
            ranked = sorted(input_docs, key=lambda x: x.get('ce_score', 0.0), reverse=True)
            return ranked[:top_k]
            
        except Exception as e:
            logger.error(f"Cross-encoder ranking failed: {e}")
            return docs[:top_k]
        finally:
            # Clear GPU cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def cascade_rank(self, query: str, docs: List[Dict], ltop_k=100, ctop_k=5) -> List[Dict]:
        """Cascade ranking with fallbacks"""
        if not docs:
            return []
        
        try:
            # First stage: LambdaMART
            first_stage = self.ltr_lamda(query, docs, ltop_k)
            if not first_stage:
                return docs[:ctop_k]
            
            # Second stage: Cross-encoder
            final = self.ltr_ce(query, first_stage, ctop_k)
            return final or first_stage[:ctop_k]
            
        except Exception as e:
            logger.error(f"Cascade ranking failed: {e}")
            return docs[:ctop_k]
    
    @weave.op()
    def rerank(self, query: str, docs: List[Dict], mode: str = "cascade") -> List[Dict]:
        """Main reranking interface with mode selection"""
        if not docs:
            return []
        
        if mode == "lambda":
            return self.ltr_lamda(query, docs)
        elif mode == "cross-encoder":
            return self.ltr_ce(query, docs)
        else:
            return self.cascade_rank(query, docs)
    
    @weave.op()
    async def rerank_from_retriever(self, query: str, mode: str = "cascade") -> List[Dict]:
        """Async reranking with comprehensive error handling"""
        try:
            t0 = Timer()
            
            # Retrieve documents
            docs = await self.retriever.aretrieve(query)
            if not docs:
                logger.warning(f"No documents retrieved for query: {query}")
                return []
            
            # Rerank documents
            result = self.rerank(query, docs, mode)
            
            elapsed = t0.elapsed()
            logger.info(f"Reranking completed in {elapsed:.2f}s ({len(docs)} -> {len(result)} docs)")
            
            return result
            
        except Exception as e:
            logger.error(f"Async reranking failed: {e}")
            # Return empty list rather than raising
            return []

# Test functionality
if __name__ == "__main__":
    async def main():
        try:
            r = Ranker(Retriever())
            res = await r.rerank_from_retriever("What causes type 2 diabetes?")
            print(f"✅ Ranker test successful: {len(res)} results")
            if res:
                print(f"Top result: {res[0].get('text', '')[:100]}...")
        except Exception as e:
            print(f"❌ Ranker test failed: {e}")

    asyncio.run(main())
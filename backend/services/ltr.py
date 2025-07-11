# ──────────────────────────────────────────────────────────────────────────────
# ltr.py – Ranker class (LambdaMART → Cross-Encoder)
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import torch, os, joblib, numpy as np, weave
import lightgbm as lgb
from sentence_transformers import CrossEncoder
import asyncio

from backend.services.retrieve import Retriever
from backend.app import config
from backend.utils import Timer
from pydantic import BaseModel, ConfigDict
BaseModel.model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

LAMBDA_PATH  = config.LAMBDA_PATH
CE_PATH    = config.CE_PATH
class Ranker():  # Inherits from weave.Model for monitoring
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"
    
    def __init__(self, retriever: Retriever, lgbm_path=LAMBDA_PATH, ce_path=CE_PATH, **data):
        super().__init__(**data)  # Weave model init
        self.retriever = retriever
        self.lgbm = lgb.Booster(model_file=str(lgbm_path))
        self.lgbm.params["n_jobs"] = max(1, os.cpu_count() - 1)  # use all but one CPU core
        # self.lgbm.set_params(n_jobs=max(1, os.cpu_count() - 1))
        # Cross-encoder on GPU half-precision
        self.ce = CrossEncoder(
            ce_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            tokenizer_kwargs={"padding": True, "truncation": True},
            model_kwargs={"torch_dtype": torch.float16},
        )

        # lookup tables (once)
        chunks = self.retriever.chunks
        self.id2row   = {cid: i for i, cid in enumerate(chunks["id"])}
        self.vec_norm = self.retriever.vecs.copy()
        self.vec_norm /= np.linalg.norm(self.vec_norm, axis=1, keepdims=True)
        self.lengths  = chunks["text"].str.split().str.len().to_numpy()
        self.bm_store      = self.retriever.bm25_store
        self.bm25_ids = {cid: i for i, cid in enumerate(self.bm_store.ids)}

    # ── feature matrix (vectorised) ----------------------------------------
    def _feat_mat(self, query: str, q_emb, docs: List[Dict]) -> np.ndarray:
        ids   = [d["id"] for d in docs]
        ridx  = np.fromiter((self.id2row[i]   for i in ids), dtype=np.int32)
        bidx  = np.fromiter((self.bm25_ids[i] for i in ids), dtype=np.int32)

        bm25_vec = self.bm_store.bm.get_scores(query.split())[bidx]
        cos_vec  = (self.vec_norm[ridx] @ q_emb).astype(np.float32)
        lens     = self.lengths[ridx]
        return np.stack([bm25_vec, cos_vec, lens], 1).astype(np.float32)

    # ── public API ----------------------------------------------------------
    def ltr_lamda(self, query:str, docs: List[Dict], top_k:int=100) -> List[Dict]:
        q_emb = self.retriever.embed_model.encode([query])[0]
        X     = self._feat_mat(query, q_emb, docs)
        lgbm_scores = self.lgbm.predict(X)
        for d, sc in zip(docs, lgbm_scores):
            d["ltr_score"] = float(sc)
        top_lgbm = sorted(docs, key=lambda x: x["ltr_score"], reverse=True)[:top_k]
        return top_lgbm

    # Optimize cross-encoder with batching
    def ltr_ce(self, query:str, docs: List[Dict], top_k:int=20) -> List[Dict]:
        pairs = [[query, d['text']] for d in docs]
        # Batch processing with attention to GPU memory
        batch_size = 32 if torch.cuda.is_available() else 8
        scores = self.ce.predict(
            pairs, 
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        for d,s in zip(docs, scores):
            d['ce_score']=float(s)
        reranked = sorted(docs, key=lambda x: x['ce_score'], reverse=True)
        return reranked[:top_k]

    def cascade_rank(self, query: str, docs: List[Dict], ltop_k=100, ctop_k=5) -> List[Dict]:
        """LambdaMART → Cross‑Encoder cascade."""
        first = self.ltr_lamda(query, docs, ltop_k)
        final = self.ltr_ce(query, first, ctop_k)
        return final
    # @weave.op()
    def rerank(self, query, docs, mode = "cascade"):
        if mode == "lambda":
            return self.ltr_lamda(query, docs)
        elif mode == "cross-encoder":
            return self.ltr_ce(query, docs)
        else:
            return self.cascade_rank(query, docs)
        
    @weave.op()
    async def rerank_from_retriever(self, query: str, mode="cascade") -> List[Dict]:
        """Async version with proper error handling"""
        try:
            t0 = Timer()
            docs = await self.retriever.aretrieve(query)
            result = self.rerank(query, docs, mode)
            print(f"Rerank completed in {t0.elapsed():.2f}s")
            return result
        except Exception as e:
            print(f"Error in rerank_from_retriever: {str(e)}")
            raise

if __name__ == "__main__":

    # print("Ranker test")
    # print("LambdaMART model:", LAMBDA_PATH)
    # print("Cross-Encoder model:", CE_PATH)
    # ret = Retriever()
    # d = ret.retrieve("What causes type 2 diabetes?")
    # r = Ranker(ret)
    # re_docs = r.rerank("What causes type 2 diabetes?", d, mode="cascade")
    # print(f"Retrieved {len(d)} docs, reranked to {len(re_docs)}")
    # print("Top 5 reranked docs:")
    # print(re_docs[:5])
    import asyncio
    async def main():
        
        r = Ranker(Retriever())
        res = await r.rerank_from_retriever("What causes type 2 diabetes?")
        print(res[0])

    asyncio.run(main())



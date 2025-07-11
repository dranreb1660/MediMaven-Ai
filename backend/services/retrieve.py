# src/backend/services/rag_modules/retrieve.py
from __future__ import annotations
import numpy as np, pandas as pd, weave
from typing import List, Dict
from backend.app import config
from backend.services.vector_stores import QdrantStore, BM25Store
from pydantic import BaseModel, ConfigDict
import concurrent.futures
import asyncio
from sentence_transformers import SentenceTransformer


# BaseModel.model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
weave.init("Medimaven-rag-production")

class Retriever():
    """Hybrid retriever combining sparse (BM25) and dense (Qdrant) search"""
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"
    def __init__(self, qdrant_force_rebuild: bool = False, bm_force_rebuild: bool = False, use_cloud: bool = False, **data):
        super().__init__(**data)
        # Load metadata
        self.chunks = pd.read_parquet(config.META_PQT)
        self.vecs = np.load(config.EMB_NPY)
        self.embed_model= self._build_embed(config.EMBED_NAME)

        # Initialize stores
        self.bm25_store = BM25Store(
            pkl_path=config.BM25_PKL,
            chunks_df=self.chunks,
            force_rebuild=bm_force_rebuild
        )
        
        # Initialize Qdrant with verification

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
        qdrant_info = self.qdrant_store.get_collection_info()
                # if qdrant_info and qdrant_info.get("points_count") == len(self.chunks):


        print("✅ Retriever initialized with:")
        print(f"  - BM25 docs: {len(self.bm25_store.ids)}")
        print(f"  - Qdrant info: {qdrant_info}")
        
        # Final verification
        if qdrant_info is None:
            raise RuntimeError(
                "Qdrant initialization failed after 3 attempts. "
                "Please check your Qdrant configuration and try again."
            )
        elif qdrant_info.get("points_count") != len(self.chunks):
            print(
                f"Qdrant count mismatch: Expected {len(self.chunks)}, "
                f"got {qdrant_info.get('points_count')}"
            )
    @staticmethod
    def _build_embed(model_name: str):
        return SentenceTransformer(model_name)
    
    # @weave.op()
    def sparse_search(self, query: str, top_k: int = 200) -> List[Dict]:
        return self.bm25_store.search(query, top_k)

    def dense_search(self, query: str, top_k: int = 300) -> List[Dict]:
        return self.qdrant_store.search(query, top_k)
    
    def fuse(
        self,
        dense: List[Dict],
        sparse: List[Dict],
        k: int = 200,
        k_rrf: int = 60,
        w_dense: float = 2.0,
        w_sparse: float = 1.0,
    ) -> List[Dict]:
        fused, scr = {}, {}
        def _add(itm, ch, rk, w):
            fid = itm["id"]
            if fid not in fused:
                fused[fid] = {
                    "id": fid,
                    "text": itm["text"],
                    "url": itm.get("url"),
                    "source": itm.get("source"),
                    "dense_score": None, "dense_rank": None,
                    "bm25_score":  None, "bm25_rank":  None,
                }
            fused[fid][f"{ch}_score"] = itm["score"]
            fused[fid][f"{ch}_rank"]  = rk
            scr[fid] = scr.get(fid, 0) + w / (k_rrf + rk)

        for r, d in enumerate(dense):  _add(d, "dense",  r, w_dense)
        for r, s in enumerate(sparse): _add(s, "bm25",   r, w_sparse)

        return sorted(
            [{**fused[i], "rrf_score": sc} for i, sc in scr.items()],
            key=lambda x: x["rrf_score"],
            reverse=True,
        )[:k]

    # @weave.op()
    def retrieve(self, query: str, mode="fuse") -> List[Dict]:
        if mode == "dense":
            return self.dense_search(query)
        if mode == "sparse":
            return self.sparse_search(query)
        
        # Run both searches in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            dense_future = executor.submit(self.dense_search, query)
            sparse_future = executor.submit(self.sparse_search, query)
            dense, sparse = dense_future.result(), sparse_future.result()
        
        return self.fuse(dense, sparse)
    
    @weave.op()
    async def aretrieve(self, query: str, mode="fuse") -> List[Dict]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.retrieve, query, mode)

# CLI test remains the same
if __name__ == "__main__":
    async def main():
        # Initialize the retriever
        r = Retriever(
            qdrant_force_rebuild=False,
            bm_force_rebuild=False,
            use_cloud=False  # Set to True if you want to use Qdrant Cloud
        )
        
        # Perform a retrieval
        results = await r.aretrieve("What causes type 2 diabetes?")
        print(f'Async results[:5]--------:\n {results[:5]}')

        res = r.retrieve("What causes type 2 diabetes?")
        print(f'Sync results[:5]--------:\n {res[:5]}')
        
    import asyncio
    asyncio.run(main())

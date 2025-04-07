import os
import sys
import gc
import json
import time
import faiss
import numpy as np
import xgboost as xgb
from typing import List, Dict, Any, Optional

# Torch imports
import torch
from sentence_transformers import SentenceTransformer

# Other imports
import weave
from src.utils import get_mongo_connection, cosine_similarity, l2_distance, get_device, clean_response
from zenml import step, pipeline
import requests

# Initialize monitoring tools
weave.init("medical-rag-production")

class MedicalRAGSystem:
    """End-to-end medical QA RAG system using TGI and monitoring."""

    def __init__(
        self,
        faiss_index_path: str = "./models/faiss_index.bin",
        metadata_path: str = "./models/faiss_index_data.json",
        ltr_model_path: str = "./models/ltr_best_model.json",
        embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        tgi_url: str = os.getenv("TGI_API_URL", "http://localhost:8080"),
        device=get_device(),
    ):
        self.device = device
        self.tgi_url = tgi_url

        try:
            self.embed_model = SentenceTransformer(embed_model_name, device=self.device)
        except Exception as e:
            print(f"Error loading embedding model on {self.device}, falling back to CPU: {str(e)}")
            self.device = torch.device("cpu")
            self.embed_model = SentenceTransformer(embed_model_name, device="cpu")

        self.retriever = faiss.read_index(faiss_index_path)
        with open(metadata_path, "rb") as f:
            self.metadata = json.load(f)
        self.ranker = xgb.XGBRanker()
        self.ranker.load_model(ltr_model_path)
        self.db = get_mongo_connection()

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_model.encode(query, convert_to_numpy=True)

    @weave.op()
    def retrieve(self, query_emb: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        distances, indices = self.retriever.search(np.expand_dims(query_emb, 0), top_k)
        return [
            {
                'doc_id': idx,
                "context_id": self.metadata[idx]["context_id"],
                "score": score,
                "context": self.metadata[idx]["context"],
            }
            for score, idx in zip(distances[0], indices[0])
        ]

    @weave.op()
    def rerank(self, query_embedding, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        docs_embeddings = [self.embed_query(query=doc['context']) for doc in docs]
        cosims = [cosine_similarity(query_embedding, doc_emb) for doc_emb in docs_embeddings]
        l2s = [l2_distance(query_embedding, doc_emb) for doc_emb in docs_embeddings]
        lens = [len(doc['context'].split()) for doc in docs]
        features = np.array([cosims, l2s, lens]).T

        scores = self.ranker.predict(features)
        for doc, score in zip(docs, scores):
            doc["ltr_score"] = float(score)
        return sorted(docs, key=lambda x: x["ltr_score"], reverse=True)

    @weave.op()
    def generate(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        try:
            context = "\n\n".join(d["context"] for d in context_docs[:3])

            sys_msg = (
                "You are a helpful medical AI assistant. Provide clear and concise answers based on the given question with the help of given context. "
                "The answer is after the [/INST] tag so generate and summarize everything after the [/INST] tag."
            )

            prompt = f"[INST] <<SYS>>\n{sys_msg}\n<<SYS>>\nQuestion: {query}\nContext: {context}\n[/INST]"

            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 256,
                    "do_sample": True,
                    "temperature": 0.7,
                    "stop": ["<|eot_id|>"]
                }
            }
            start_time = time.time()
            response = requests.post(f"{self.tgi_url}/generate", json=payload, timeout=60)
            latency = time.time() - start_time

            if response.status_code == 200:
                text = clean_response(response.json().get("generated_text", ""))
                print(f"TGI response latency: {latency:.2f}s")
            else:
                raise ValueError(f"TGI error: {response.status_code}, {response.text}")

            self.cleanup_memory()
            return text.strip()

        except Exception as e:
            print(f"Generation error: {e}")
            self.cleanup_memory()
            return "[ERROR] LLM generation failed."

    def cleanup_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def log_to_db(self, query: str, answer: str, docs: List[Dict[str, Any]]):
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj

        converted_docs = [convert_numpy_types(doc) for doc in docs]
        try:
            self.db["rag_inference_logs"].insert_one({
                "query": query,
                "answer": answer,
                "top_docs": converted_docs,
                "timestamp": time.time(),
            })
        except Exception as e:
            print(f"Warning: Failed to log to database: {e}")

    def run(self, query: str) -> str:
        try:
            query_emb = self.embed_query(query)
            retrieved_docs = self.retrieve(query_emb)
            if not retrieved_docs:
                raise ValueError("No documents retrieved")

            ranked_docs = self.rerank(query_emb, retrieved_docs)
            answer = self.generate(query, ranked_docs)
            self.log_to_db(query, answer, ranked_docs[:3])
            return answer

        except Exception as e:
            print(f"Error in pipeline: {e}")
            self.cleanup_memory()
            return "I encountered an error processing your medical query. Please try again later."

# --- ZenML Integration ---
@step
def load_rag_system() -> MedicalRAGSystem:
    """Load the RAG system (ZenML will cache this)."""
    # Initialize system with memory optimization flags
    rag = MedicalRAGSystem()
    return rag

@step
def execute_query(rag: MedicalRAGSystem, query: str) -> str:
    """Run a query through the RAG pipeline."""
    return rag.run(query)

@pipeline(enable_cache=True)
def rag_inference_pipeline(query: str):
    """Production RAG pipeline with monitoring."""
    rag = load_rag_system()
    return execute_query(rag, query)
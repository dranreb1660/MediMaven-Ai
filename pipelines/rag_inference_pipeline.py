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
from gptqmodel import GPTQModel

# Other imports
import weave
from src.utils import get_mongo_connection, cosine_similarity, l2_distance, get_device
from zenml import step, pipeline

# Initialize monitoring tools
weave.init("medical-rag-production")
# mlflow.set_tracking_uri("http://localhost:5000")

class MedicalRAGSystem:
    """End-to-end medical QA RAG system with monitoring and fallback handling."""
    
    def __init__(self, faiss_index_path: str = "./models/faiss_index.bin",
            metadata_path: str = "./models/faiss_index_data.json",
            ltr_model_path: str = "./models/ltr_best_model.json",
            llm_path: str = "./models/llama_finetuned/gptqmodel_4bit",
            embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
            device = get_device()):
        
            self.device = device
       
            # Initialize embedding model with fallback
            try:
                self.embed_model = SentenceTransformer(embed_model_name, device=self.device)
            except Exception as e:
                print(f"Error loading embedding model on {self.device}, falling back to CPU: {str(e)}")
                self.device = torch.device("cpu")
                self.embed_model = SentenceTransformer(embed_model_name, device="cpu")
            
            # Load retrieval components
            self.retriever = faiss.read_index(faiss_index_path)
            with open(metadata_path, "rb") as f:
                self.metadata = json.load(f)
            self.ranker = xgb.XGBRanker()
            self.ranker.load_model(ltr_model_path)
            
            # Load LLM with memory management
            try:
                gc.collect()
                if self.device== "mps":
                    torch.mps.empty_cache()
                elif self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                
                self.llm = GPTQModel.load(llm_path, device=self.device)
                self.tokenizer = self.llm.tokenizer
            except (RuntimeError, MemoryError) as e:
                print(f"Memory error loading LLM on {self.device}, falling back to CPU: {str(e)}")
                gc.collect()
                self.llm = GPTQModel.load(llm_path, device="cpu")
                self.tokenizer = self.llm.tokenizer
            
            # Initialize monitoring
            self.db = get_mongo_connection()
            
  
    # ---- Pipeline Steps ----
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query using the configured model."""
        return self.embed_model.encode(query, convert_to_numpy=True)

    @weave.op()
    def retrieve(self, query_emb: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve documents from FAISS index."""
        distances, indices = self.retriever.search(np.expand_dims(query_emb, 0), top_k)
        return [   { 'doc_id': idx,
                    "context_id": self.metadata[idx]["context_id"],
                    "score": score,
                    "context": self.metadata[idx]["context"]} 
                    for score, idx in zip(distances[0], indices[0])]

    @weave.op()
    def rerank(self, query_embedding, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Re-rank documents using XGBoost LTR model."""
        # Feature engineering (customize based on your LTR training)        
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
        """Generate answer using fine-tuned LLM."""
        try:
            # Clean memory before generation
            if self.device == "mps" or self.device.startswith("cuda"):
                if self.device == "mps":
                    torch.mps.empty_cache()
                elif self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                
            context = "\n\n".join(d["context"] for d in context_docs[:3])  
            def build_prompt(query: str, context: str) -> str:
                sys_msg = (
                    "You are a helpful medical AI assistant. "
                    "Provide clear and concise answers based on the given question with the help of given context."
                    "the answer is after the [/INST] tag so generate and summerize evrything after the [/INST] tag"
                )

                # Example format for Llama instructions:
                prompt = (
                    f"[INST] <<SYS>>\n{sys_msg}\n<<SYS>>\n"
                    f"Question: {query}\n"
                    f"Context: {context}\n"
                    "[/INST]"
                )
                return prompt
            prompt = build_prompt(query, context)

            # Manage memory during tokenization and generation
            with torch.no_grad():
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                output = self.llm.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True
                )
                response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Clear memory after generation
            del inputs, output
            gc.collect()
            if self.device == "mps" or self.device.startswith("cuda"):
                if self.device == "mps":
                    torch.mps.empty_cache()
                elif self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                
            return response.split("[/INST]")[-1].strip()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "not enough memory" in str(e).lower():
                # Memory-specific error handling
                print(f"Memory error during generation: {str(e)}")
                gc.collect()
                if self.device == "mps" or self.device.startswith("cuda"):
                    if self.device == "mps":
                        torch.mps.empty_cache()
                    elif self.device.startswith("cuda"):
                        torch.cuda.empty_cache()
                
                # Try again with CPU if not already on CPU
                if self.device != "cpu":
                    print("Falling back to CPU for this generation")
                    # Move model to CPU temporarily for this generation
                    with torch.no_grad():
                        cpu_inputs = self.tokenizer(prompt, return_tensors="pt")
                        # Process on CPU
                        tmp_model = self.llm.to("cpu")
                        cpu_output = tmp_model.generate(
                            **cpu_inputs,
                            max_new_tokens=128,  # Reduced tokens for safety
                            temperature=0.7,
                            do_sample=True
                        )
                        response = self.tokenizer.decode(cpu_output[0], skip_special_tokens=True)
                        # Move model back to original device
                        self.llm.to(self.device)
                        del tmp_model, cpu_inputs, cpu_output
                        gc.collect()
                        return response.split("[/INST]")[-1].strip()
            
            # Re-raise other errors
            raise

    def log_to_db(self, query: str, answer: str, docs: List[Dict[str, Any]]):
        """Store results in MongoDB with timestamp."""
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
        
        # Convert numpy types to Python native types
        converted_docs = [convert_numpy_types(doc) for doc in docs]
        
        try:
            self.db["rag_inference_logs"].insert_one({
                "query": query,
                "answer": answer,
                "top_docs": converted_docs,
                "timestamp": time.time()
            })
        except Exception as e:
            print(f"Warning: Failed to log to database: {e}")

    # ---- End-to-End Pipeline ----
    def run(self, query: str) -> str:
        """Execute full RAG pipeline with monitoring and fallbacks."""
        try:
            # Embed
            query_emb = self.embed_query(query)
            
            # Retrieve
            retrieved_docs = self.retrieve(query_emb)
            if not retrieved_docs:
                raise ValueError("No documents retrieved")
            
            # Rerank
            ranked_docs = self.rerank(query_emb, retrieved_docs)
            
            # Generate
            answer = self.generate(query, ranked_docs)
            
            # Log
            self.log_to_db(query, answer, ranked_docs[:3])
            
            return answer
        
        except Exception as e:
            # Fallback to a safe response
            # weave.log({"error": str(e)})
            print(f'error occured and caught ==>:  {e}')
            
            # Clean up memory in case of errors
            gc.collect()
            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device.startswith("cuda"):
                torch.cuda.empty_cache()
                
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
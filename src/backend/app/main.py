from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time
import os
import wandb
from src.backend.app.schemas import ChatRequest, ChatResponse
from src.backend.services.rag_service import RAGService
from pipelines.rag_inference_pipeline import MedicalRAGSystem
import torch

from src.backend.app.config import config
print("📟 Torch device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
print("🔥 CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("🚀 CUDA device:", torch.cuda.get_device_name(0))

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

@app.on_event("startup")
def init_wandb():
    wandb.init(project="medical-rag-production", config={
        "model": config.MODEL_PATH,
        "faiss_index": config.FAISS_INDEX_PATH
    })

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    start_time = time.time()
    try:
        # rag = RAGService.get_instance()
        rag = MedicalRAGSystem(
                # llm_path=config.MODEL_PATH,
                # faiss_index_path=config.FAISS_INDEX_PATH
            )
        answer = rag.run(request.query)
        
        wandb.log({
            "query": request.query,
            "latency_ms": (time.time() - start_time) * 1000,
            "answer_length": len(answer)
        })
        
        return {
            "answer": answer,
            "latency_ms": round((time.time() - start_time) * 1000, 2),
            "model_version": os.path.basename(config.MODEL_PATH)
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
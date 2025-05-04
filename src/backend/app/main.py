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
origins = [
    "https://www.medimaven-ai.com",
    "https://medimaven-ai.com",          # root → 301 → www, still safe
    "http://localhost:5173",             # vite dev
    "http://127.0.0.1:5173",
]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # or ["*"] for total open
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],   # '*'' is fine too
    allow_headers=["*"],
)
@app.on_event("startup")
def init_wandb():
    wandb.init(project="medical-rag-production", config={
        "model": config.MODEL_PATH,
        "faiss_index": config.FAISS_INDEX_PATH
    })

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    start_time = time.time()
    try:
        # rag = RAGService.get_instance()
        rag = MedicalRAGSystem(
            faiss_index_path = config.faiss_index_path,
            metadata_path= config.metadata_path,
            ltr_model_path = config.ltr_model_path,
            embed_model_name = config.EMBED_MODEL_NAME,
            ############## TGI URL ##############
            tgi_url = config.TGI_API_URL
            )
        answer = rag.run(request.query)
        
        wandb.log({
            "query": request.query,
            "latency": f'{(time.time() - start_time)}s',
            "answer_length": len(answer)
        })
        
        return {
            "answer": answer,
            "latency": round((time.time() - start_time), 2),
            "model_version": os.path.basename(config.MODEL_PATH)
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
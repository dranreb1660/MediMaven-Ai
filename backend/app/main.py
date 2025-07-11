# ─────────────────────────────────────────────────────────────────────────────
# src/backend/app/main.py ─ production FastAPI entry-point
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import os, time, hashlib
from contextlib import asynccontextmanager
from uuid import uuid4
from collections import OrderedDict
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch, wandb, weave

# pydantic request/response models
from backend.app.schemas import ChatRequest, ChatResponse, ConversationMessage

# New modular RAG stack
from backend.services.medimaven import MediMaven
from backend.services.generate import Backend
from backend.services.caching import InHouseCache
from backend.app import config
from backend.utils import Timer
# ---------------------------------------------------------------------------
# 0) Lifespan Event Handler
# ---------------------------------------------------------------------------
bot: MediMaven | None = None  # Singleton RAG engine


# Global cache instance
response_cache = InHouseCache(max_size=500)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events"""
    global bot
    # ─── Cache Setup ─────────────────────────────────────────────────────
    # Use Redis cache if configured, otherwise use in-memory cache
    if config.REDIS_URL:
        import redis.asyncio as redis
        redis_client = redis.from_url(config.REDIS_URL)
        
        class RedisCache:
            async def get(self, key: str):
                return await redis_client.get(key)
            
            async def set(self, key: str, value: any, expire: int = 300):
                await redis_client.set(key, value, ex=expire)
        
        response_cache = RedisCache()
    
    # ─── Startup Logic ─────────────────────────────────────────────────
    # Choose LLM backend automatically: vLLM if GPU, else transformers
    llm_backend = Backend.VLLM if torch.cuda.is_available() else Backend.TRANSFORMERS
    bot = MediMaven(llm_backend)

    # Init Weights & Biases run
    if config.ENABLE_MONITORING:
        wandb.init(
            project="Medimaven-rag-production",
            config={
                "backend": llm_backend.name,
                "model_path": os.getenv("MODEL_PATH", config.MODEL_DIR),
            },
        )
        weave.init("Medimaven-rag-production")
    print("✅ MediMaven engine initialised with", llm_backend.name)
    
    yield  # App runs here
    
    # ─── Shutdown Logic ────────────────────────────────────────────────
    if bot is None:
        return
        
    gen = bot.generator
    if hasattr(gen, "engine"):  # VLLM backend
        shutdown_fn = getattr(gen.engine, "shutdown", None) or getattr(
            gen.engine, "close", None
        )
        if shutdown_fn:
            await shutdown_fn()
            print("🛑 vLLM engine shut down")

# ---------------------------------------------------------------------------
# 1) Environment + CORS
# ---------------------------------------------------------------------------
print("📟 Torch device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
if torch.cuda.is_available():
    print("🚀 CUDA device:", torch.cuda.get_device_name(0))

origins = config.ALLOWED_ORIGINS

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Intent recognition helpers
INTENT_SUFFIX = {
    "elaborate": " detailed explanation causes pathophysiology",
    "prevent":   " preventive measures lifestyle emergency steps",
    "scared":    " reassurance prognosis risk statistics"
}

def enhance_query(history: list, new_query: str) -> str:
    """Augment query based on detected intent"""
    suffix = next((v for k,v in INTENT_SUFFIX.items() if k in new_query.lower()), "")
    return " ".join(h["user"] + " " + h["assistant"] for h in history[-3:]) + " " + new_query + suffix

# ---------------------------------------------------------------------------
# 2) Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}

# In-memory conversation store (replace with Redis in production)
conversation_store: Dict[str, dict] = {}

def update_conversation(memory: dict, user_query: str, assistant_response: str, max_history: int = 3):
    """Update conversation history with new turn"""
    if "turns" not in memory:
        memory["turns"] = []
        
    memory["turns"].append({
        "user": user_query, 
        "assistant": assistant_response
    })
    
    # Maintain conversation window
    if len(memory["turns"]) > max_history:
        memory["turns"] = memory["turns"][-max_history:]

def should_bypass_cache(query: str) -> bool:
    """Determine if we should bypass cache for this query"""
    bypass_phrases = ["update", "latest", "current", "new", "emergency", "urgent"]
    return any(phrase in query.lower() for phrase in bypass_phrases)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    global bot
    if bot is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialised")

    t0 = Timer()
    
    try:
        # Get or create conversation ID
        cid = req.conversation_id or str(uuid4())
        memory = conversation_store.get(cid, {"turns": []})
        first_turn = len(memory["turns"]) == 0

        # ─── Query Processing ──────────────────────────────────────────
        if not first_turn:  # Follow-up question
            standalone_q = await bot.rewrite_followup(req.query, memory["turns"])
            stitched_query = standalone_q
        else:
            stitched_query = req.query
            
        # Generate cache key (query + last 2 conversation turns)
        context_str = stitched_query + "|".join(t["user"] for t in memory["turns"][-2:])
        context_hash = hashlib.sha256(context_str.encode()).hexdigest()
        cache_key = f"response:{context_hash}"
        
        # Check cache
        cached_response = None
        cache_hit = False
        
        if config.ENABLE_CACHING and not should_bypass_cache(stitched_query):
            cached_response = response_cache.get(cache_key)
            
        if cached_response:
            result = cached_response
            cache_hit = True
        else:
            # Process query through RAG pipeline (original response without welcome)
            result = await bot.answer(stitched_query)
            if config.ENABLE_CACHING:
                # Cache the ORIGINAL response (without welcome)
                response_cache.set(cache_key, result)

        # ─── Add welcome message ONLY for first turn ───────────────────
        # Use a copy of the result to avoid modifying cached data
        final_answer = result["answer"]
        if first_turn:
            final_answer = "👋 Welcome to MediMaven.\n" + final_answer

        # Update conversation with the FINAL answer (may include welcome)
        update_conversation(memory, req.query, final_answer)
        conversation_store[cid] = memory
        
        total_latency = round(t0.elapsed(), 3)

        return ChatResponse(
            answer=final_answer,  # Use the final answer (may include welcome)
            citations=result["citations"],
            latency=total_latency,
            cache_hit=cache_hit,
            conversation_id=cid,
            messages=[
                ConversationMessage(user=turn["user"], assistant=turn["assistant"])
                for turn in memory["turns"]
            ],
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
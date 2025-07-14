# # ─────────────────────────────────────────────────────────────────────────────
# # src/backend/app/main.py  —  production FastAPI entry-point (refactored)
# # ─────────────────────────────────────────────────────────────────────────────
# from __future__ import annotations

# import os, time, json, hashlib, uuid
# from contextlib import asynccontextmanager
# from typing import Dict

# import torch, wandb, weave
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import StreamingResponse
# from sse_starlette.sse import EventSourceResponse  # keep for future upgrades

# from backend.app import config
# from backend.app.schemas import ChatRequest, ChatResponse, ConversationMessage
# from backend.services.medimaven import MediMaven
# from backend.services.generate import Backend
# from backend.services.caching import InHouseCache
# from backend.utils import Timer

# # ────────────────────────────────────────────────────────────────────────
# # 0) Globals & helpers
# # ────────────────────────────────────────────────────────────────────────
# bot: MediMaven | None = None                    # singleton RAG engine
# conversation_store: Dict[str, dict] = {}        # in-memory; replace w/ Redis
# response_cache = InHouseCache(max_size=500)     # LRU if Redis not configured

# INTENT_SUFFIX = {
#     "elaborate": " detailed explanation causes pathophysiology",
#     "prevent":   " preventive measures lifestyle emergency steps",
#     "scared":    " reassurance prognosis risk statistics",
# }

# def should_bypass_cache(q: str) -> bool:
#     return any(t in q.lower() for t in ("update", "latest", "current",
#                                         "new", "emergency", "urgent"))

# def enhance_query(history: list, q: str) -> str:
#     suf = next((v for k, v in INTENT_SUFFIX.items() if k in q.lower()), "")
#     stitched_history = " ".join(f"{t['user']} {t['assistant']}" for t in history[-3:])
#     return f"{stitched_history} {q} {suf}".strip()

# def update_conv(memory: dict, user: str, assistant: str, k: int = 3):
#     memory.setdefault("turns", []).append({"user": user, "assistant": assistant})
#     if len(memory["turns"]) > k:
#         memory["turns"] = memory["turns"][-k:]

# # ────────────────────────────────────────────────────────────────────────
# # 1) FastAPI app + lifecycle
# # ────────────────────────────────────────────────────────────────────────
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global bot, response_cache

#     # 1. cache backend (Redis if URL provided)
#     if config.REDIS_URL:
#         import redis.asyncio as redis
#         redis_client = redis.from_url(config.REDIS_URL)

#         class RedisCache:
#             async def get(self, k): return await redis_client.get(k)
#             async def set(self, k, v, expire=300): await redis_client.set(k, v, ex=expire)
#         response_cache = RedisCache()  # type: ignore

#     # 2. initialise MediMaven engine
#     backend_choice = Backend.VLLM if torch.cuda.is_available() else Backend.TRANSFORMERS
#     bot = MediMaven(backend_choice)

#     if config.ENABLE_MONITORING:
#         wandb.init(project="Medimaven-rag-production",
#                    config={"backend": backend_choice.name,
#                            "model_path": os.getenv("MODEL_PATH", config.MODEL_DIR)})
#         weave.init("Medimaven-rag-production")

#     print("✅ MediMaven initialised with", backend_choice.name)
#     yield

#     # 3. graceful shutdown (vLLM engine)
#     if bot and hasattr(bot.generator, "engine"):
#         close_fn = getattr(bot.generator.engine, "shutdown",
#                            getattr(bot.generator.engine, "close", None))
#         if close_fn:
#             await close_fn()
#             print("🛑 vLLM engine shut down")

# app = FastAPI(lifespan=lifespan)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=config.ALLOWED_ORIGINS,
#     allow_credentials=True,
#     allow_methods=["GET", "POST", "OPTIONS"],
#     allow_headers=["*"],
# )

# # ────────────────────────────────────────────────────────────────────────
# # 2) Health
# # ────────────────────────────────────────────────────────────────────────
# @app.get("/health")
# def health_check(): return {"status": "ok"}

# # ────────────────────────────────────────────────────────────────────────
# # 3) Helper that BOTH routes share (full RAG pipeline without I/O)
# # ────────────────────────────────────────────────────────────────────────
# async def process_query(req: ChatRequest) -> tuple[dict, str, dict]:
#     """
#     Returns:
#         result      – dict from bot.answer(...)   (answer  + citations)
#         cid         – conversation id
#         memory      – mutable conversation history entry
#     """
#     if bot is None:
#         raise HTTPException(503, "RAG engine not initialised")

#     cid = req.conversation_id or str(uuid.uuid4())
#     memory = conversation_store.get(cid, {"turns": []})
#     first_turn = len(memory["turns"]) == 0

#     stitched_q = req.query if first_turn else await bot.rewrite_followup(req.query, memory["turns"])
#     stitched_q = enhance_query(memory["turns"], stitched_q)

#     # cache key: query + last 2 user messages
#     key_src = stitched_q + "|".join(t["user"] for t in memory["turns"][-2:])
#     cache_key = f"resp:{hashlib.sha256(key_src.encode()).hexdigest()}"

#     result, cache_hit = None, False
#     if config.ENABLE_CACHING and not should_bypass_cache(stitched_q):
#         result = await response_cache.get(cache_key)  # type: ignore

#     if not result:
#         result = await bot.answer(stitched_q)         # { answer, citations }
#         if config.ENABLE_CACHING:
#             await response_cache.set(cache_key, result)  # type: ignore
#     else:
#         cache_hit = True

#     # prepend welcome for first turn
#     if first_turn:
#         result = result.copy()
#         result["answer"] = "👋 Welcome to MediMaven.\n" + result["answer"]

#     # update conversation store
#     update_conv(memory, req.query, result["answer"])
#     conversation_store[cid] = memory

#     return result, cid, memory, cache_hit

# # ────────────────────────────────────────────────────────────────────────
# # 4) Legacy JSON endpoint  (kept for backward-compat)
# # ────────────────────────────────────────────────────────────────────────
# @app.post("/chat", response_model=ChatResponse)
# async def chat_endpoint(req: ChatRequest):
#     timer = Timer()
#     try:
#         result, cid, memory, cache_hit = await process_query(req)
#         return ChatResponse(
#             answer=result["answer"],
#             citations=result["citations"],
#             latency=round(timer.elapsed(), 3),
#             cache_hit=cache_hit,
#             conversation_id=cid,
#             messages=[ConversationMessage(**t) for t in memory["turns"]],
#         )
#     except Exception as e:
#         raise HTTPException(500, f"Processing error: {e}")

# # ────────────────────────────────────────────────────────────────────────
# # 5) NEW /chat/stream  (single-call SSE with meta frame)
# # ────────────────────────────────────────────────────────────────────────
# @app.post("/chat/stream")
# async def chat_stream(req: ChatRequest):
#     """
#     Streams tokens as they are generated.
#     Final frame includes the same metadata as /chat so frontend
#     never needs a follow-up call.
#     Latency is measured **until the first token is yielded**.
#     """
#     if bot is None:
#         raise HTTPException(503, "RAG engine not initialised")

#     # reuse all query-handling, cache, rewrite logic
#     result, cid, memory, cache_hit = await process_query(req)

#     async def event_gen():
#         t0 = time.perf_counter()
#         first_token_sent, first_latency = False, 0.0

#         # If the response came from cache, stream it token-by-token quickly
#         if cache_hit:
#             for tok in result["answer"].split():
#                 if not first_token_sent:
#                     first_latency = time.perf_counter() - t0
#                     first_token_sent = True
#                 yield f"data: {json.dumps({'token': tok + ' '})}\n\n"
#                 await asyncio.sleep(0.005)
#         else:
#             async for tok in bot.stream_answer(req.query):
#                 if not first_token_sent:
#                     first_latency = time.perf_counter() - t0
#                     first_token_sent = True
#                 yield f"data: {json.dumps({'token': tok})}\n\n"

#         meta = {
#             "done": True,
#             "answer": result["answer"],
#             "citations": result["citations"],
#             "latency": round(first_latency, 3),
#             "conversation_id": cid,
#             "messages": memory["turns"],
#             "cache_hit": cache_hit,
#         }
#         yield f"data: {json.dumps(meta)}\n\n"

#     return StreamingResponse(event_gen(),
#                              media_type="text/event-stream")
    
    
    
    # ─────────────────────────────────────────────────────────────────────────────
# src/backend/app/main.py   – FastAPI entry-point with SSE
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import os, json, time, asyncio, hashlib, uuid
from contextlib import asynccontextmanager
from typing import Dict

import torch, wandb, weave, re, string
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.app import config
from backend.app.schemas import ChatRequest, ChatResponse, ConversationMessage
from backend.services.medimaven import MediMaven
from backend.services.generate import Backend
from backend.services.caching import InHouseCache
from backend.utils import Timer

# ─── globals ────────────────────────────────────────────────────────────────
bot: MediMaven | None = None
conversation_store: Dict[str, dict] = {}
response_cache: InHouseCache | any = InHouseCache(max_size=500)

def update_conv(mem: dict, user: str, assistant: str, k: int = 3):
    mem.setdefault("turns", []).append({"user": user, "assistant": assistant})
    if len(mem["turns"]) > k:
        mem["turns"] = mem["turns"][-k:]

def should_bypass(q: str):  # simple heuristic
    return any(t in q.lower() for t in ("update", "latest", "current",
                                        "new", "emergency", "urgent"))


RE_CONTRACTIONS = [
    (re.compile(r"\b([A-Za-z]+)\s+n’t\b", re.I), r"\1n’t"),   # don’t, can’t
    (re.compile(r"\b([A-Za-z]+)\s+s\b",    re.I), r"\1’s"),   # it’s, what’s
    (re.compile(r"\b([A-Za-z]+)\s+d\b",    re.I), r"\1’d"),   # I’d, you’d
    (re.compile(r"\b([A-Za-z]+)\s+ll\b",   re.I), r"\1’ll"),  # we’ll
    (re.compile(r"\b([A-Za-z]+)\s+re\b",   re.I), r"\1’re"),  # they’re
    (re.compile(r"\b([A-Za-z]+)\s+ve\b",   re.I), r"\1’ve"),  # we’ve
]

def postprocess(text: str) -> str:
    # collapse " . [3]." → " [3]."
    text = re.sub(r"\.\s+\[(\d+)]\.", r" [\1].", text)

    # contractions
    for pat, repl in RE_CONTRACTIONS:
        text = pat.sub(repl, text)

    # capitalize first letter + after sentence ends
    def _caps(m):
        return m.group(0).upper()
    text = re.sub(r"(?:^|[.!?]\s+)(\w)", _caps, text)

    return text


# ─── lifespan ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global bot, response_cache

    if config.REDIS_URL:
        import redis.asyncio as redis
        rc = redis.from_url(config.REDIS_URL)

        class RedisCache:
            async def get(self, k): return await rc.get(k)
            async def set(self, k, v, ex=300): await rc.set(k, v, ex=ex)
        response_cache = RedisCache()  # type: ignore

    backend_choice = Backend.VLLM if torch.cuda.is_available() else Backend.TRANSFORMERS
    bot = MediMaven(backend_choice)
    print("✅ MediMaven started with", backend_choice.name)

    if config.ENABLE_MONITORING:
        wandb.init(project="Medimaven-rag-production",
                   config={"backend": backend_choice.name,
                           "model_path": os.getenv("MODEL_PATH", config.MODEL_DIR)})
        weave.init("Medimaven-rag-production")

    yield  # ── application runs ──

    if bot and hasattr(bot.generator, "engine"):
        close_fn = getattr(bot.generator.engine,
                           "shutdown",
                           getattr(bot.generator.engine, "close", None))
        if close_fn: await close_fn()

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ─── shared RAG pipeline (non-stream) ───────────────────────────────────────
async def run_rag(req: ChatRequest):
    if bot is None:
        raise HTTPException(503, "RAG engine not initialised")

    cid  = req.conversation_id or str(uuid.uuid4())
    mem  = conversation_store.get(cid, {"turns": []})
    first_turn = not mem["turns"]

    stitched_q = req.query if first_turn else \
        await bot.rewrite_followup(req.query, mem["turns"])

    # simple two-turn cache key
    key_src = stitched_q + "|".join(t["user"] for t in mem["turns"][-2:])
    cache_key = f"resp:{hashlib.sha256(key_src.encode()).hexdigest()}"

    result, cache_hit = None, False
    if config.ENABLE_CACHING and not should_bypass(stitched_q):
        result = await response_cache.get(cache_key)  # type: ignore

    if not result:
        result = await bot.answer_rag(stitched_q)
        if config.ENABLE_CACHING:
            await response_cache.set(cache_key, result)  # type: ignore
    else:
        cache_hit = True

    if first_turn:
        result = result.copy()
        result["answer"] = "👋 Welcome to MediMaven.\n" + result["answer"]

    update_conv(mem, req.query, result["answer"])
    conversation_store[cid] = mem

    return result, cid, mem, cache_hit

# ─── health ────────────────────────────────────────────────────────────────
@app.get("/health")
def health(): return {"status": "ok"}

# ─── legacy JSON endpoint ──────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    timer = Timer()
    result, cid, mem, cache_hit = await run_rag(req)
    return ChatResponse(
        answer          = result["answer"],
        citations       = result["citations"],
        latency         = round(timer.elapsed(), 3),
        cache_hit       = cache_hit,
        conversation_id = cid,
        messages        = [ConversationMessage(**t) for t in mem["turns"]],
    )

# ─── streaming endpoint (single-call, first-turn welcome, cache support) ────
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    if bot is None:
        raise HTTPException(503, "RAG engine not initialised")

    # ── replicate run_rag() logic WITHOUT full generation ───────────────
    cid  = req.conversation_id or str(uuid.uuid4())
    mem  = conversation_store.get(cid, {"turns": []})
    first = not mem["turns"]

    stitched_q = req.query if first else \
        (print("Rewriting follow-up query..."), await bot.rewrite_followup(req.query, mem["turns"]))[1]

    key_src  = stitched_q + "|".join(t["user"] for t in mem["turns"][-2:])
    cache_key = f"resp:{hashlib.sha256(key_src.encode()).hexdigest()}"

    cached, cache_hit = None, False
    if config.ENABLE_CACHING and not should_bypass(stitched_q):
        cached = await response_cache.get(cache_key)    # type: ignore

    if cached:
        # fast path – stream cached answer word-by-word
        ranked_docs = []     # already embedded in cached['citations']
        prompt      = None
        cache_hit   = True
    else:
        ranked_docs, prompt = await bot.prepare_stream(stitched_q)
        cache_hit           = False

    # provisional assistant message in history
    update_conv(mem, req.query, "")
    conversation_store[cid] = mem

    # ── streaming coroutine ─────────────────────────────────────────────
    async def events():
        t0 = time.perf_counter()
        latency_first, answer_buf = None, ""

        # 1) optionally prepend welcome
        if first:
            welcome = "👋 Welcome to MediMaven.\n"
            answer_buf += welcome
            yield f"data: {json.dumps({'token': welcome})}\n\n"

        # 2) send tokens
        if cache_hit:
            # cached: stream quickly word-by-word
            for w in cached["answer"].split(" "):
                if latency_first is None:
                    latency_first = time.perf_counter() - t0
                answer_buf += w + " "
                yield f"data: {json.dumps({'token': w + ' '})}\n\n"
        else:
            async for tok in bot.stream_generator(prompt):
                if latency_first is None:
                    latency_first = time.perf_counter() - t0
                answer_buf += tok
                yield f"data: {json.dumps({'token': tok}, ensure_ascii=False)}\n\n"

        # 3) finalise history + maybe cache
        answer_clean = postprocess(bot.clean_text(answer_buf))
        mem["turns"][-1]["assistant"] = answer_clean

        if not cache_hit and config.ENABLE_CACHING:
            to_cache = {
                "answer": answer_clean,
                "citations": [
                    {"id": r["id"], "source": r.get("source"),
                     "url": r.get("url"), "rank": i + 1}
                    for i, r in enumerate(ranked_docs[:5])
                ],
            }
            await response_cache.set(cache_key, to_cache)   # type: ignore

        # 4) meta frame
        meta = {
            "done": True,
            "answer": answer_clean,
            "citations": cached["citations"] if cache_hit else [
                {"id": r["id"], "source": r.get("source"),
                 "url": r.get("url"), "rank": i + 1}
                for i, r in enumerate(ranked_docs[:5])
            ],
            "latency": round(latency_first or 0, 3),
            "conversation_id": cid,
            "messages": mem["turns"],
        }
        yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(),
                             media_type="text/event-stream")


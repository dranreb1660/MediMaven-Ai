# src/backend/app/main.py – FastAPI entry-point with SSE (stateless, history-driven)
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import os, json, time, hashlib, uuid, re
import asyncio
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, AsyncGenerator
from datetime import datetime, timezone

import torch, wandb, weave
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy import select, func, text, cast
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy import bindparam
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from pydantic import ValidationError

from backend.app import config
from backend.app.schemas import ChatRequest, ChatResponse, ConversationMessage
from backend.services.medimaven import MediMaven
from backend.services.generate import Backend
from backend.services.caching import InHouseCache
from backend.utils import Timer
from backend.models import Conversation
from backend.services.db import get_db
from backend.auth import get_current_user_optional, get_current_user
from backend.utils import get_logger
# ─── Setup Logging ──────────────────────────────────────────────────────────

logger = get_logger(__name__)
# ─── Constants & Configuration ──────────────────────────────────────────────
MAX_QUERY_LENGTH = 10000
MAX_HISTORY_TURNS = 50  # Frontend history for follow-ups
MAX_CACHE_KEY_LENGTH = 250
CACHE_TTL_SECONDS = 300
DB_RETRY_ATTEMPTS = 3
DB_RETRY_DELAY = 0.5

# Database history management
MAX_THREADS_PER_USER = 20
MAX_TURNS_PER_THREAD = 50

# ─── Globals ────────────────────────────────────────────────────────────────
bot: Optional[MediMaven] = None
response_cache: InHouseCache | Any = InHouseCache(max_size=500)

RE_CONTRACTIONS = [
    (re.compile(r"\b([A-Za-z]+)\s+n’t\b", re.I), r"\1n’t"),   # don’t, can’t
    (re.compile(r"\b([A-Za-z]+)\s+s\b",    re.I), r"\1’s"),   # it’s, what’s
    (re.compile(r"\b([A-Za-z]+)\s+d\b",    re.I), r"\1’d"),   # I’d, you’d
    (re.compile(r"\b([A-Za-z]+)\s+ll\b",   re.I), r"\1’ll"),  # we’ll
    (re.compile(r"\b([A-Za-z]+)\s+re\b",   re.I), r"\1’re"),  # they’re
    (re.compile(r"\b([A-Za-z]+)\s+ve\b",   re.I), r"\1’ve"),  # we’ve
]

def should_bypass(q: str) -> bool:
    """Check if query should bypass cache due to time-sensitive keywords."""
    if not q or not isinstance(q, str):
        return False
    bypass_terms = ("update", "latest", "current", "new", "emergency", "urgent", "today", "now")
    return any(term in q.lower() for term in bypass_terms)

def postprocess(text: str) -> str:
    """Post-process generated text with proper error handling."""
    if not text or not isinstance(text, str):
        return ""
    
    try:
        # Clean up citation formatting
        text = re.sub(r"\.\s+\[(\d+)]\.", r" [\1].", text)
        
        # Apply contractions
        for pat, repl in RE_CONTRACTIONS:
            text = pat.sub(repl, text)
        
        # Capitalize sentences
        text = re.sub(r"(?:^|[.!?]\s+)(\w)", lambda m: m.group(0).upper(), text)
        
        return text.strip()
    except Exception as e:
        logger.error(f"Error in postprocess: {e}")
        return text.strip()

def validate_conversation_id(cid: str) -> bool:
    """Validate conversation ID format."""
    if not cid or not isinstance(cid, str):
        return False
    
    # Check if it's a valid UUID format
    try:
        uuid.UUID(cid)
        return True
    except ValueError:
        return False

def sanitize_query(query: str) -> str:
    """Sanitize user query with length limits and content filtering."""
    if not query or not isinstance(query, str):
        raise HTTPException(400, "Query cannot be empty")
    
    query = query.strip()
    if len(query) > MAX_QUERY_LENGTH:
        raise HTTPException(400, f"Query too long (max {MAX_QUERY_LENGTH} characters)")
    
    if not query:
        raise HTTPException(400, "Query cannot be empty after sanitization")
    
    return query

def validate_history(history: list) -> list:
    """Validate and sanitize conversation history."""
    if not history:
        return []
    
    if len(history) > MAX_HISTORY_TURNS:
        logger.info(f"History truncated from {len(history)} to {MAX_HISTORY_TURNS} turns")
        history = history[-MAX_HISTORY_TURNS:]
    
    validated_history = []
    for i, msg in enumerate(history):
        try:
            if isinstance(msg, dict):
                # Ensure required fields exist
                if "user" not in msg or "assistant" not in msg:
                    logger.info(f"Skipping invalid history item at index {i}")
                    continue
                validated_history.append(msg)
            else:
                # Handle Pydantic model
                validated_history.append(msg.model_dump())
        except Exception as e:
            logger.info(f"Error validating history item at index {i}: {e}")
            continue
    
    return validated_history

async def db_retry_wrapper(operation, *args, max_retries: int = DB_RETRY_ATTEMPTS):
    """Wrapper for database operations with retry logic."""
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return await operation(*args)
        except (SQLAlchemyError, IntegrityError) as e:
            last_exception = e
            logger.info(f"DB operation failed (attempt {attempt + 1}/{max_retries}): {e}")
            
            if attempt < max_retries - 1:
                await asyncio.sleep(DB_RETRY_DELAY * (2 ** attempt))  # Exponential backoff
            continue
        except Exception as e:
            # Non-retryable error
            logger.error(f"Non-retryable DB error: {e}")
            raise
    
    logger.error(f"DB operation failed after {max_retries} attempts")
    raise last_exception

class ConversationManager:
    """Manages conversation history with automatic cleanup and optimization."""
    
    def __init__(self, db_session):
        self.db = db_session
    
    async def append_turn_optimized(
        self, 
        user: Dict[str, Any], 
        cid: str, 
        turn: Dict[str, Any], 
        first: bool = False
    ):
        """Append turn with automatic history management using efficient PostgreSQL operations."""
        user_id = user["sub"]
        
        # Add metadata to turn
        turn_with_meta = {
            **turn,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        try:
            if first:
                await self._handle_new_conversation(user_id, cid, turn_with_meta)
            else:
                await self._append_to_existing_conversation(cid, turn_with_meta)
            
            logger.debug(f"✅ Turn appended with history management for cid={cid}")
            
        except Exception as e:
            logger.error(f"History management failed, using fallback: {e}")
            await self._simple_append_fallback(user_id, cid, turn_with_meta, first)
    
    async def _handle_new_conversation(self, user_id: str, cid: str, turn: Dict[str, Any]):
        """Handle new conversation with thread limit enforcement in single transaction."""
        async with self.db.begin():
            # Clean up old threads if needed (single efficient query)
            cleanup_query = text("""
                WITH user_conversations AS (
                    SELECT cid, updated_at,
                           ROW_NUMBER() OVER (ORDER BY updated_at DESC) as rn
                    FROM conversations 
                    WHERE user_id = :user_id
                ),
                old_conversations AS (
                    SELECT cid 
                    FROM user_conversations 
                    WHERE rn >= :max_threads
                )
                DELETE FROM conversations 
                WHERE cid IN (SELECT cid FROM old_conversations)
            """)
            
            await self.db.execute(
                cleanup_query, 
                {"user_id": user_id, "max_threads": MAX_THREADS_PER_USER}
            )
            
            # Create preview and insert new conversation
            user_msg = turn.get("user", "")
            preview = " ".join(user_msg.split()[:6]) if user_msg else "New conversation"
            preview = preview[:100]
            
            stmt = pg_insert(Conversation.__table__).values(
                cid=cid,
                user_id=user_id,
                preview=preview,
                messages=[turn],
                created_at=func.now(),
                updated_at=func.now()
            )
            
            await self.db.execute(stmt)
    
    async def _append_to_existing_conversation(self, cid: str, turn: Dict[str, Any]):
        """Append to existing conversation with turn limit enforcement."""
        try:
            # First, get the current conversation
            stmt = select(Conversation).where(Conversation.cid == cid)
            result = await self.db.execute(stmt)
            conversation = result.scalar_one_or_none()
            
            if not conversation:
                raise ValueError(f"Conversation {cid} not found")
            
            # Get current messages and append new turn
            current_messages = conversation.messages or []
            
            # Apply turn limit: remove oldest if we're at the limit
            if len(current_messages) >= MAX_TURNS_PER_THREAD:
                current_messages = current_messages[1:]  # Remove the first (oldest) message
            
            # Add the new turn
            updated_messages = current_messages + [turn]
            
            # Update the conversation
            conversation.messages = updated_messages
            conversation.updated_at = func.now()
            
            # Commit the changes
            await self.db.commit()
            logger.info(f"Successfully appended turn to conversation {cid}")
            
        except Exception as e:
            # Rollback on any error to clean transaction state
            try:
                await self.db.rollback()
            except Exception as rollback_error:
                logger.error(f"Rollback failed: {rollback_error}")
            
            logger.error(f"Error appending to conversation {cid}: {e}")
            raise
    
    async def _simple_append_fallback(self, user_id: str, cid: str, turn: Dict[str, Any], first: bool):
        """Fallback method without history management."""
        try:
            one_param = {"one": [turn]}
            
            if first:
                preview = " ".join(turn.get("user", "").split()[:6])[:100]
                stmt = pg_insert(Conversation.__table__).values(
                    cid=cid,
                    user_id=user_id,
                    preview=preview,
                    messages=[turn],
                    created_at=func.now(),
                    updated_at=func.now()
                )
            else:
                stmt = (
                    Conversation.__table__.update()
                    .where(Conversation.cid == cid)
                    .values(
                        messages=cast(Conversation.messages, JSONB).op("||")(bindparam("one")),
                        updated_at=func.now(),
                    )
                )
            
            await self.db.execute(stmt, one_param)
            await self.db.commit()
            logger.info(f"[# simple append fallback]Successfully appended turn to conversation {cid}")

        except Exception as e:
            logger.error(f"Fallback append failed: {e}")

async def append_turn_to_db(user: Dict[str, Any], cid: str, turn: Dict[str, Any], db, first: bool = False):
    """Enhanced DB append with automatic history management."""
    if not user or not cid or not turn:
        logger.info("Skipping DB append - missing data")
        return
    
    if not validate_conversation_id(cid):
        logger.error(f"Invalid conversation ID format: {cid}")
        return
    
    try:
        conv_manager = ConversationManager(db)
        await conv_manager.append_turn_optimized(user, cid, turn, first)
        
    except Exception as e:
        logger.error(f"Enhanced DB append failed for cid={cid}: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with proper error handling."""
    global bot, response_cache

    try:
        # ── Optional Redis fallback for response cache ──
        if config.REDIS_URL:
            try:
                import redis.asyncio as redis
                rc = redis.from_url(config.REDIS_URL)
                
                # Test Redis connection
                await rc.ping()
                
                class RedisCache:
                    def __init__(self, redis_client):
                        self.redis = redis_client
                    
                    async def get(self, k: str):
                        try:
                            result = await self.redis.get(k)
                            return json.loads(result) if result else None
                        except Exception as e:
                            logger.info(f"Redis get failed: {e}")
                            return None
                    
                    async def set(self, k: str, v: Any, ex: int = CACHE_TTL_SECONDS):
                        try:
                            await self.redis.set(k, json.dumps(v), ex=ex)
                        except Exception as e:
                            logger.warning(f"Redis set failed: {e}")
                
                response_cache = RedisCache(rc)
                logger.info("✅ Redis cache initialized")
            except Exception as e:
                logger.warning(f"Redis initialization failed, using in-memory cache: {e}")

        # ── Initialize MediMaven ──
        backend_choice = Backend.VLLM if torch.cuda.is_available() else Backend.TRANSFORMERS
        bot = MediMaven(backend_choice)
        
        # ── Optional monitoring ──
        if config.ENABLE_MONITORING:
            try:
                wandb.init(project="Medimaven-rag-production", config={"backend": backend_choice.name})
                weave.init("Medimaven-rag-production")
                logger.info("✅ Monitoring initialized")
            except Exception as e:
                logger.warning(f"Monitoring initialization failed: {e}")

        # Start background cleanup task for history management
        cleanup_task = asyncio.create_task(periodic_cleanup_task())

        logger.info(f"✅ MediMaven started with {backend_choice.name}")
        yield

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    finally:
        # ── Cleanup ──
        try:
            # Cancel background cleanup task
            if 'cleanup_task' in locals():
                cleanup_task.cancel()
                try:
                    await cleanup_task
                except asyncio.CancelledError:
                    pass

            if bot and hasattr(bot.generator, "engine"):
                shutdown = getattr(bot.generator.engine, "shutdown", None) or getattr(bot.generator.engine, "close", None)
                if shutdown:
                    await shutdown()
            
            if hasattr(response_cache, 'redis'):
                await response_cache.redis.close()
                
            logger.info("✅ Cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

async def periodic_cleanup_task():
    """Background task for periodic history cleanup."""
    while True:
        try:
            db_gen = get_db()
            db = await db_gen.__anext__()
            
            try:
                # Simple cleanup: delete old conversations beyond limit per user
                cleanup_query = text("""
                    DELETE FROM conversations 
                    WHERE cid NOT IN (
                        SELECT cid FROM conversations 
                        WHERE user_id = conversations.user_id 
                        ORDER BY updated_at DESC 
                        LIMIT :max_threads
                    )
                """)
                
                result = await db.execute(cleanup_query, {"max_threads": MAX_THREADS_PER_USER})
                await db.commit()
                
                if result.rowcount > 0:
                    logger.info(f"Periodic cleanup: {result.rowcount} old threads removed")
                    
            finally:
                await db.close()
            
            # Run cleanup every hour
            await asyncio.sleep(3600)
            
        except Exception as e:
            logger.error(f"Periodic cleanup error: {e}")
            await asyncio.sleep(3600)

# ─── FastAPI Application ────────────────────────────────────────────────────
app = FastAPI(
    lifespan=lifespan,
    title="MediMaven RAG API",
    description="Medical AI Assistant with RAG capabilities",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def run_rag(req: ChatRequest) -> tuple:
    """Core RAG logic with comprehensive error handling."""
    if bot is None:
        raise HTTPException(503, "RAG engine not initialized")

    try:
        # 1) Validate and sanitize input
        query = sanitize_query(req.query)
        cid = req.conversation_id or str(uuid.uuid4())
        
        if req.conversation_id and not validate_conversation_id(req.conversation_id):
            raise HTTPException(400, "Invalid conversation ID format")

        # 2) Validate and seed memory from client history
        validated_history = validate_history(req.history or [])
        mem = {"turns": validated_history}
        first = not mem["turns"]

        # 3) Rewrite follow-up if needed
        try:
            q = query if first else await bot.rewrite_followup(query, mem["turns"])
        except Exception as e:
            logger.warning(f"Follow-up rewrite failed, using original query: {e}")
            q = query

        # 4) Generate cache key with length validation
        key_parts = [q] + [t.get("user", "") for t in mem["turns"][-2:]]
        key_src = "|".join(filter(None, key_parts))
        if len(key_src) > MAX_CACHE_KEY_LENGTH:
            key_src = key_src[:MAX_CACHE_KEY_LENGTH]
        
        cache_key = f"resp:{hashlib.sha256(key_src.encode('utf-8')).hexdigest()}"

        # 5) Attempt cache retrieval
        result = None
        cache_hit = False
        
        if config.ENABLE_CACHING and not should_bypass(q):
            try:
                result = await response_cache.get(cache_key)
                cache_hit = bool(result)
            except Exception as e:
                logger.warning(f"Cache retrieval failed: {e}")

        # 6) Generate answer if not cached
        if not result:
            try:
                result = await bot.answer_rag(q)
                
                # Cache the result
                if config.ENABLE_CACHING:
                    try:
                        await response_cache.set(cache_key, result)
                    except Exception as e:
                        logger.warning(f"Cache storage failed: {e}")
            except Exception as e:
                logger.error(f"RAG answer generation failed: {e}")
                raise HTTPException(500, "Failed to generate response")

        # 7) Add welcome message for first turn
        if first:
            result = result.copy()
            result["answer"] = "👋 Welcome to MediMaven.\n" + result.get("answer", "")

        # 8) Append turn to memory
        new_turn = {
            "user": query,
            "assistant": result.get("answer", ""),
            "citations": result.get("citations", []),
            "latency": result.get("latency_s"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        mem["turns"].append(new_turn)

        return result, cid, mem, cache_hit, first

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in run_rag: {e}")
        raise HTTPException(500, "Internal server error")

@app.get("/health")
def health():
    """Health check endpoint."""
    try:
        status = {
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bot_initialized": bot is not None,
            "cuda_available": torch.cuda.is_available()
        }
        return status
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    req: ChatRequest,
    request: Request,
    user=Depends(get_current_user_optional),
    db=Depends(get_db)
):
    """Main chat endpoint with comprehensive error handling."""
    timer = Timer()
    
    try:
        # Log request details (without sensitive data)
        client_ip = request.client.host if request.client else "unknown"
        logger.info(f"Chat request from {client_ip}, query length: {len(req.query)}")
        
        result, cid, mem, cache_hit, first = await run_rag(req)

        # Persist conversation turn
        if user and mem["turns"]:
            await append_turn_to_db(user, cid, mem["turns"][-1], db, first)

        response = ChatResponse(
            answer=result.get("answer", ""),
            citations=result.get("citations", []),
            latency=round(timer.elapsed(), 3),
            cache_hit=cache_hit,
            conversation_id=cid,
            messages=[ConversationMessage(**t) for t in mem["turns"]],
        )
        
        logger.info(f"Chat response generated: {timer.elapsed():.3f}s, cache_hit: {cache_hit}")
        return response

    except HTTPException:
        raise
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(400, f"Invalid request data: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in chat_endpoint: {e}")
        raise HTTPException(500, "Internal server error")

@app.post("/chat/stream")
async def chat_stream(
    req: ChatRequest,
    request: Request,
    user=Depends(get_current_user_optional),
    db=Depends(get_db)
):
    """Streaming chat endpoint with robust error handling."""
    if bot is None:
        raise HTTPException(503, "RAG engine not initialized")

    try:
        client_ip = request.client.host if request.client else "unknown"
        logger.info(f"Stream request from {client_ip}")

        # Validate and prepare data
        query = sanitize_query(req.query)
        cid = req.conversation_id or str(uuid.uuid4())
        
        validated_history = validate_history(req.history or [])
        mem = {"turns": validated_history}
        first = not mem["turns"] or not mem["turns"][0].get("assistant")


        # Rewrite and cache logic (similar to run_rag)
        try:
            q = query if first else await bot.rewrite_followup(query, mem["turns"])
            logger.info(f"Rewritten query: {q}")
        except Exception as e:
            logger.warning(f"Follow-up rewrite failed: {e}")
            q = query

        key_parts = [q] + [t.get("user", "") for t in mem["turns"][-2:]]
        key_src = "|".join(filter(None, key_parts))
        if len(key_src) > MAX_CACHE_KEY_LENGTH:
            key_src = key_src[:MAX_CACHE_KEY_LENGTH]
        
        cache_key = f"resp:{hashlib.sha256(key_src.encode('utf-8')).hexdigest()}"
        
        cached = None
        cache_hit = False
        
        if config.ENABLE_CACHING and not should_bypass(q):
            try:
                cached = await response_cache.get(cache_key)
                cache_hit = bool(cached)
            except Exception as e:
                logger.warning(f"Cache retrieval failed: {e}")

        # Prepare streaming
        if cache_hit:
            ranked_docs, prompt = [], None
        else:
            try:
                ranked_docs, prompt = await bot.prepare_stream(q)
            except Exception as e:
                logger.error(f"Stream preparation failed: {e}")
                raise HTTPException(500, "Failed to prepare stream")

        # Add placeholder turn
        mem["turns"].append({"user": query, "assistant": ""})

        async def events() -> AsyncGenerator[str, None]:
            t0 = time.perf_counter()
            first_latency = None
            buffer = ""
            
            try:
                # Welcome message
                if first:
                    welcome = "👋 Welcome to MediMaven.\n"
                    buffer += welcome
                    yield f"data: {json.dumps({'token': welcome}, ensure_ascii=False)}\n\n"

                # Stream tokens
                if cache_hit and cached:
                    # Stream cached response word by word
                    words = cached.get("answer", "").split(" ")
                    for word in words:
                        if first_latency is None:
                            first_latency = time.perf_counter() - t0
                        
                        token = word + " "
                        buffer += token
                        yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
                        await asyncio.sleep(0.01)  # Simulate streaming delay
                else:
                    # Stream from generator
                    try:
                        async for tok in bot.stream_generator(prompt):
                            if first_latency is None:
                                first_latency = time.perf_counter() - t0
                            
                            buffer += tok
                            yield f"data: {json.dumps({'token': tok}, ensure_ascii=False)}\n\n"
                    except Exception as e:
                        logger.error(f"Streaming generation failed: {e}")
                        error_msg = "Sorry, I encountered an error while generating the response."
                        yield f"data: {json.dumps({'token': error_msg}, ensure_ascii=False)}\n\n"
                        buffer = error_msg

                # Finalize response
                answer = postprocess(bot.clean_text(buffer)) if hasattr(bot, 'clean_text') else postprocess(buffer)
                
                citations = []
                if cache_hit and cached:
                    citations = cached.get("citations", [])
                else:
                    citations = [
                        {
                            "id": r.get("id"),
                            "source": r.get("source"),
                            "url": r.get("url"),
                            "rank": i + 1
                        }
                        for i, r in enumerate(ranked_docs[:5])
                    ]

                latency = round(first_latency or 0, 3)

                # Update memory
                turn_data = {
                    "user": query,
                    "assistant": answer,
                    "citations": citations,
                    "latency": latency,
                }
                
                mem["turns"][-1] = turn_data

                # Persist to database
                if user:
                    await append_turn_to_db(user, cid, turn_data, db, first)

                # Cache new response
                if not cache_hit and config.ENABLE_CACHING:
                    try:
                        await response_cache.set(cache_key, {
                            "answer": answer,
                            "citations": citations
                        })
                    except Exception as e:
                        logger.warning(f"Failed to cache response: {e}")

                # Final metadata
                meta = {
                    "done": True,
                    "answer": answer,
                    "citations": citations,
                    "latency": latency,
                    "conversation_id": cid,
                    "messages": mem["turns"],
                }
                yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

            except Exception as e:
                logger.error(f"Error in streaming events: {e}")
                error_response = {
                    "error": True,
                    "message": "An error occurred during streaming",
                    "conversation_id": cid
                }
                yield f"data: {json.dumps(error_response)}\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat_stream: {e}")
        raise HTTPException(500, "Internal server error")



@app.get("/chat/list")
async def chat_list(user=Depends(get_current_user), db=Depends(get_db)):
    """List user conversations with robust error handling."""
    try:
        # Start with a fresh transaction by rolling back any existing failed transaction
        try:
            await db.rollback()
        except Exception:
            pass  # Ignore rollback errors if no transaction exists
        
        # Use a simple, reliable query that works across different PostgreSQL setups
        query = select(
            Conversation.cid,
            Conversation.preview,
            Conversation.messages,
            Conversation.updated_at
        ).where(
            Conversation.user_id == user["sub"]
        ).order_by(
            Conversation.updated_at.desc()
        ).limit(MAX_THREADS_PER_USER)
        
        result = await db.execute(query)
        rows = result.fetchall()
        
        conversations = []
        for row in rows:
            try:
                # Handle messages safely
                messages = row.messages or []
                if isinstance(messages, str):
                    try:
                        messages = json.loads(messages)
                    except (json.JSONDecodeError, TypeError):
                        messages = []
                
                conversations.append({
                    "cid": row.cid,
                    "preview": row.preview or "No preview",
                    "messages": messages,
                    "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                    "message_count": len(messages) if isinstance(messages, list) else 0
                })
                
            except Exception as e:
                logger.warning(f"Error processing conversation {row.cid}: {e}")
                # minimal entry to avoid losing the conversation completely
                conversations.append({
                    "cid": row.cid,
                    "preview": "Error loading conversation",
                    "messages": [],
                    "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                    "message_count": 0
                })
        
        await db.commit()  # Ensure clean transaction state
        logger.info(f"Retrieved {len(conversations)} conversations for user {user['sub']}")
        return conversations

    except Exception as e:
        logger.error(f"Error listing conversations: {e}")
        
        # Ensure clean state for next request
        try:
            await db.rollback()
        except Exception:
            pass
        
        # Return empty list to keep frontend working
        return []

@app.post("/chat/end")
async def chat_end(payload: dict, user=Depends(get_current_user_optional)):
    """End conversation endpoint with validation."""
    try:
        cid = payload.get("conversation_id")
        logger.info(f"Ending conversation: {cid} for user {user['sub'] if user else 'anonymous'}")
        if not cid:
            raise HTTPException(400, "Missing conversation_id")
        
        if not validate_conversation_id(cid):
            raise HTTPException(400, "Invalid conversation_id format")
        
        logger.info(f"Conversation ended: {cid}")
        return {"status": "ended", "cid": cid}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ending conversation: {e}")
        raise HTTPException(500, "Failed to end conversation")

# Add error handlers
@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    logger.error(f"Internal server error: {exc}")
    return {"error": "Internal server error", "detail": str(exc)}

@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    logger.error(f"Validation error: {exc}")
    return {"error": "Validation error", "detail": str(exc)}
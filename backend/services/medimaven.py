from __future__ import annotations

import time, asyncio, weave, logging
from typing import Dict, List, Tuple, Optional

from backend.services.retrieve import Retriever
from backend.services.ltr import Ranker
from backend.services.generate import Generator, Backend
from backend.utils import Timer

logger = logging.getLogger(__name__)

class MediMaven(weave.Model):
    """Production-ready RAG engine with robust error handling"""
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"
        
    def __init__(self, llm_backend: Backend = Backend.VLLM):
        super().__init__()
        self._initialized = False
        
        try:
            # Initialize components with validation
            self.retriever = Retriever()
            self.ranker = Ranker(self.retriever)
            self.generator = Generator(llm_backend)
            
            # Optional punctuation model with fallback
            self.punctuate = None
            try:
                from deepmultilingualpunctuation import PunctuationModel
                self.punctuate = PunctuationModel()
            except Exception as e:
                logger.warning(f"Punctuation model unavailable: {e}")
            
            self._initialized = True
            logger.info("✅ MediMaven initialized successfully")
            
        except Exception as e:
            logger.error(f"MediMaven initialization failed: {e}")
            self._cleanup()
            raise RuntimeError(f"Failed to initialize MediMaven: {e}")
    
    @staticmethod
    def _prompt(query: str, ctx: List[Dict], n: int = 5) -> str:
        """Generate prompt with input validation"""
        if not query.strip():
            query = "Please provide medical information."
        
        # Use only valid context
        valid_ctx = [c for c in ctx[:n] if c.get('text')]
        
        passages = "\n\n".join(
            f"[{i+1}] {c['text'][:1000]}"  # Limit passage length
            for i, c in enumerate(valid_ctx)
        )
        
        return f"""You are MediMaven, a board-certified virtual physician.

Style & Structure:
• Start with a single-sentence summary.
• Then give concise, evidence-based detail.
• Remain friendly and reassuring.

Citation rules:
• When you use a passage, ALWAYS cite its number in square brackets — like [2].
• If none of the passages are useful, answer confidently from your knowledge.

---
Passages:
{passages}

Question: {query}

Answer:"""
    
    async def answer_rag(self, query: str, n_ctx: int = 5) -> Dict:
        """Complete RAG pipeline with comprehensive error handling"""
        if not self._initialized:
            return self._error_response("Service temporarily unavailable")
        
        t0 = Timer()
        
        try:
            # Validate query
            if not query or not query.strip():
                return self._error_response("Please provide a valid question")
            
            # Retrieve and rank documents
            ranked = await self.ranker.rerank_from_retriever(query)
            if not ranked:
                return {
                    "answer": "I couldn't find relevant information. Could you rephrase your question?",
                    "citations": [],
                    "latency_s": round(t0.elapsed(), 3),
                }
            
            # Generate answer
            prompt = self._prompt(query, ranked, n_ctx)
            reply = await self.generator.generate(prompt, max_new_tokens=256)
            
            if not reply or not reply.strip():
                reply = "I'm having trouble generating a response. Please try rephrasing your question."
            
            # Prepare citations
            citations = [
                {
                    "id": r.get("id", ""),
                    "source": r.get("source", ""),
                    "url": r.get("url", ""),
                    "rank": i + 1
                }
                for i, r in enumerate(ranked[:n_ctx])
            ]
            
            return {
                "answer": self.clean_text(reply),
                "citations": citations,
                "latency_s": round(t0.elapsed(), 3),
            }
            
        except Exception as e:
            logger.error(f"RAG pipeline failed: {e}")
            return self._error_response("I encountered an error. Please try again.")
    
    async def prepare_stream(self, query: str, n_ctx: int = 5) -> Tuple[List[Dict], str]:
        """Prepare streaming with error handling"""
        if not self._initialized:
            return [], self._prompt("", [], n_ctx)
        
        t0 = Timer()
        
        try:
            logger.info(f"🔎 Preparing stream for: {query[:100]}...")
            
            # Retrieve and rank
            ranked = await self.ranker.rerank_from_retriever(query)
            
            retrieval_time = t0.elapsed()
            logger.info(f"⏱️ Retrieval completed in {retrieval_time:.3f}s")
            
            # Generate prompt
            prompt = self._prompt(query, ranked, n_ctx)
            
            total_time = t0.elapsed()
            logger.info(f"⏱️ Stream preparation completed in {total_time:.3f}s")
            
            return ranked, prompt
            
        except Exception as e:
            logger.error(f"Stream preparation failed: {e}")
            return [], self._prompt(query, [], n_ctx)
    
    async def stream_generator(self, prompt: str, max_new_tokens: int = 128):
        """Token streaming with error handling"""
        if not self._initialized:
            yield "Service temporarily unavailable. "
            return
        
        try:
            logger.info("🔄 Starting token generation...")
            
            token_count = 0
            async for token in self.generator.stream(prompt, max_new_tokens):
                yield token
                token_count += 1
                
                # Safety limit
                if token_count > max_new_tokens * 2:
                    logger.warning("Token limit exceeded, stopping generation")
                    break
            
            logger.info(f"✅ Token generation completed ({token_count} tokens)")
            
        except Exception as e:
            logger.error(f"Token generation failed: {e}")
            yield "I encountered an error while generating the response. "
    
    async def rewrite_followup(self, query: str, history_tail: List[Dict]) -> str:
        """Rewrite follow-up queries with simplified logic"""
        if not self._initialized or not history_tail:
            return query
        
        try:
            # Limit history to last 2 turns for context
            recent_history = history_tail[-2:]
            context = "\n".join([
                f"User: {turn.get('user', '')}\nAssistant: {turn.get('assistant', '')[:200]}..."
                for turn in recent_history
                if turn.get('user') and turn.get('assistant')
            ])
            
            if not context:
                return query
            
            prompt = (
                "Rewrite the USER FOLLOW-UP question into a clear, self-contained question "
                "This question is FROM the USER to an ASSISTANT, "
                "that fully incorporates necessary context from the conversation history. "
                "Do NOT answer, clarify, or ask questions, ONLY rewrite the user's query that will be understandable by the assistant on its own., no side NOTES or whatever"
                "\n\n"
                "Conversation history:" f"{history_tail}\n\n"
                f"USER FOLLOW-UP: {query}\n"
                "STAND-ALONE QUESTION:"
            )
            
            result = await self.generator.generate(prompt, max_new_tokens=32)
            
            # Validate result
            if result and len(result.strip()) > 5 and len(result) < 200:
                logger.info(f"Query rewritten: '{query}' -> '{result}'")
                return result.strip()
            
            return query
            
        except Exception as e:
            logger.warning(f"Follow-up rewrite failed: {e}")
            return query
    
    def clean_text(self, text: str) -> str:
        """Text cleaning with fallback"""
        if not text or not text.strip():
            return text
        
        try:
            # Basic cleanup
            text = text.strip()
            
            # Use punctuation model if available
            if self.punctuate:
                text = self.punctuate.restore_punctuation(text)
            
            return text
            
        except Exception as e:
            logger.warning(f"Text cleaning failed: {e}")
            return text.strip()
    
    def _error_response(self, message: str) -> Dict:
        """Generate standardized error response"""
        return {
            "answer": message,
            "citations": [],
            "latency_s": 0.0,
        }
    
    def _cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'generator'):
                self.generator.close()
            if hasattr(self, 'retriever') and hasattr(self.retriever, 'qdrant_store'):
                self.retriever.qdrant_store.close()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    def close(self):
        """Public cleanup method"""
        self._cleanup()
        self._initialized = False
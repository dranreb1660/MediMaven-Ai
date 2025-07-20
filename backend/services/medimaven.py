
# ────────────────────────────────────────────────────────────────────────────
# src/backend/services/rag_modules/medimaven.py
# ────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import time, asyncio, weave
from typing import Dict, List, Tuple

from deepmultilingualpunctuation import PunctuationModel

from backend.services.retrieve import Retriever
from backend.services.ltr      import Ranker
from backend.services.generate import Generator, Backend
from backend.utils import Timer

# ────────────────────────────────────────────────────────────────────────────
class MediMaven(weave.Model):
    """MediMaven RAG engine – retrieval, ranking and LLM generation."""
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    # ------------------------------------------------------------------ init
    def __init__(self, llm_backend: Backend = Backend.VLLM):
        super().__init__()
        self.retriever  = Retriever()
        self.ranker     = Ranker(self.retriever)
        self.generator  = Generator(llm_backend)
        self.punctuate  = PunctuationModel()

    # ----------------------------------------------------------- prompt helper
    @staticmethod
    def _prompt(query: str, ctx: List[Dict], n: int = 5) -> str:
        passages = "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(ctx[:n]))
        return f"""
        You are MediMaven, a board-certified virtual physician.

        Style & Structure:
        • Start with a single-sentence summary.
        • Then give concise, evidence-based detail (bullet or short-paragraph).
        • Remain friendly and reassuring.

        Citation rules:
        • When you use a passage, ALWAYS cite its number in square brackets — like [2] — exactly where the supporting detail appears in your answer.
        • If none of the passages are useful, answer confidently from your own knowledge.
        • Never mention the absence of relevant passages.

        ---
        Passages:
        {passages}

        Question: {query}

        Answer:
        """

    # ------------------------------------------------------- non-stream answer
    async def answer_rag(self, query: str, n_ctx: int = 5) -> Dict:
        """Full RAG answer – returns dict ready for JSON route."""
        t0 = Timer()

        ranked = await self.ranker.rerank_from_retriever(query)
        if not ranked:
            return {
                "answer": "I couldn't find enough information. Try rephrasing.",
                "citations": [],
                "latency_s": round(t0.elapsed(), 3),
            }

        prompt = self._prompt(query, ranked, n_ctx)
        reply  = await self.generator.generate(prompt, max_new_tokens=256) or \
                 "I couldn't generate a response. Please try again."

        citations = [
            {"id": r["id"], "source": r.get("source"),
             "url": r.get("url"), "rank": i + 1}
            for i, r in enumerate(ranked[:n_ctx])
        ]
        return {
            "answer": self.clean_text(reply),
            "citations": citations,
            "latency_s": round(t0.elapsed(), 3),
        }

    # ------------------------------------------------------- prepare for stream
    async def prepare_stream(self, query: str, n_ctx: int = 5) -> Tuple[List[Dict], str]:
        """Return (ranked_docs, prompt) for the streaming route."""
        t0 = Timer()
        print(f"🔎 Retrieving n reranking ------------\nQuery: {query}")
        ranked = await self.ranker.rerank_from_retriever(query)

        ranked_time = t0.elapsed()
        print(f"⏱️ Retrieval + Ranking done--> took {ranked_time:.3f} seconds")

        prompt = self._prompt(query, ranked, n_ctx)
        prompting_time = t0.elapsed() - ranked_time
        print(f"⏱️ Prompting done--> took {prompting_time:.3f} seconds")

        return ranked, prompt

    # ------------------------------------------------------- token generator
    async def stream_generator(self, prompt: str, max_new_tokens: int = 128):
        """Async generator that yields ONLY new tokens."""
        print("Starting token generation...")
        async for tok in self.generator.stream(prompt,
                                               max_new_tokens=max_new_tokens):
            yield tok
        print("Token generation complete. ✅ ✅") 

    # ------------------------------------------------------- follow-up rewrite
    async def rewrite_followup(self, query, history_tail):
        """Rewrite a user follow-up into a stand-alone question."""
        print("Rewriting follow-up query...")
        t0 = Timer()
        prompt = (
            "Rewrite the USER FOLLOW-UP question into a clear, self-contained question "
            "This question is FROM the USER to an ASSISTANT, "
            "that fully incorporates necessary context from the conversation history. "
            "Do NOT answer, clarify, or ask questions, ONLY rewrite the user's query that will be understandable on its own., no side NOTES or whatever"
            "\n\n"
            "Conversation history:\n"
            f"{history_tail}\n\n"
            f"USER FOLLOW-UP: {query}\n"
            "STAND-ALONE QUESTION:"
        )
        txt = await self.generator.generate(prompt, max_new_tokens=32)
        rewrite_latency = t0.elapsed()
        print(f"Rewrite latency: {rewrite_latency} s")
        return txt.strip()

    # ----------------------------------------------------------- small helpers
    def clean_text(self, text: str) -> str:
        if not text.strip(): return text
        try:  return self.punctuate.restore_punctuation(text)
        except Exception: return text

    def close(self):
        if hasattr(self.generator, "engine"):
            fn = getattr(self.generator.engine, "shutdown",
                         getattr(self.generator.engine, "close", None))
            if fn: fn()
        self.retriever.qdrant_store.close()

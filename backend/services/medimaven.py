# ────────────────────────────────────────────────────────────────────────────
# src/backend/services/rag_modules/medimaven.py
# ────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import time, asyncio, weave
from typing import Dict, List

from deepmultilingualpunctuation import PunctuationModel

from backend.services.retrieve import Retriever
from backend.services.ltr      import Ranker
from backend.services.generate import Generator, Backend
from backend.utils import Timer
from pydantic import BaseModel, ConfigDict


class MediMaven(weave.Model):  # Make weave model
    
    """MediMaven RAG Engine
    This class orchestrates the retrieval, ranking, and generation of medical answers.
    It uses a retriever to fetch relevant passages, a ranker to order them,
    and a generator to produce the final answer.
    It also includes a punctuation model for text cleanup.
    """
    # model_config = ConfigDict(arbitrary_types_allowed=True)
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"  # Important if extra attrs are added later
    def __init__(self, llm_backend: Backend = Backend.VLLM):
        super().__init__()  # Weave init
        # 0) Heavy objects
        self.retriever = Retriever()   
        print("✅ Retriever loaded")

        self.ranker     = Ranker(self.retriever)
        print("✅ Ranker loaded")

        self.generator  = Generator(llm_backend)
        print(f"✅ Generator loaded ({llm_backend.name})")

        # optional text-cleanup model
        self.punctuate  = PunctuationModel()

    # ---------------------------------------------------------------------
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
        • When you use a passage, always cite its number in square brackets — like [2] — exactly where the supporting detail appears in your answer.
        • If none of the passages are useful, answer confidently from your own knowledge.
        • Never mention the absence of relevant passages.

        ---
        Passages:
        {passages}

        Question: {query}

        Remember: If a detail comes from a passage, cite with [number] at that point in the answer.
        If it comes from your own knowledge, do **not** cite.

        Answer:
        """

    # ---------------------------------------------------------------------
    @weave.op()
    async def answer(self, query: str, n_ctx: int = 5) -> Dict:
        t0 = Timer()
        print(f"\n🔎 Starting query: '{query}'")
        
        # Directly test dense search
        # print("🧪 Running direct dense search test...")
        # test_results = self.retriever.dense_search("diabetes", top_k=5)
        # print(f"🧪 Test results: {len(test_results)} documents")
        
        print('Retrieving and Ranking...')
        ranked = await self.ranker.rerank_from_retriever(query)
        # print(f"📊 Retrieved {len(ranked)} documents")
        ranked_time = t0.elapsed()
        print(f"⏱️ Retrieval + Ranking took {ranked_time:.3f} seconds")
        # Check if we have documents
        if not ranked:
            print("⚠️ WARNING: Ranker returned zero documents!")
            # Return safe response
            return {
                "answer": "I couldn't find relevant information for your query. Please try rephrasing or ask about a different medical topic.",
                "citations": [],
                "latency_s": round(ranked_time, 3),
            }
        
            
        print('Prompting...')
        
        prompt = self._prompt(query, ranked, n_ctx)
        prompting_time = t0.elapsed() - ranked_time
        print(f"⏱️ Prompting took {prompting_time:.3f} seconds")
        print("prompt: ", prompt)  # Print first 1000 chars for brevity
        print('Replying...')
        reply = await self.generator.generate(prompt, max_new_tokens=256)
        
        # Handle empty response
        if not reply.strip():
            print("⚠️ Warning: Received empty response from LLM")
            reply = "I couldn't generate a response. Please try again with a different question."
            
        gen_time = t0.elapsed() - ranked_time - prompting_time
        print(f"⏱️ Generation took {gen_time:.3f} seconds")
        print('done...')
        latency = t0.elapsed()
        print(f"Total latency: {latency:.3f} seconds")
        citations = [
            {"id": r["id"], "source": r.get("source"), "url": r.get("url"), "rank": i + 1}
            for i, r in enumerate(ranked[:n_ctx])
        ]
        print(f'answer: {reply}')
        
        return {
            "answer": self.clean_text(reply),
            "citations": citations,
            "latency_s": latency,
        }
    async def _prepare_prompt(self, query: str, n_ctx: int) -> str:
        """Async prepare prompt while retrieval runs"""
        return self._prompt(query, [], n_ctx)  # Start with empty context
    
    @weave.op()
    async def rewrite_followup(self, query, history_tail):
        """Return a stand-alone query for retrieval."""
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
        latency = t0.elapsed()
        print(f"⏱️ Query Rewriting took {latency:.3f} seconds")
        print(f"Rewritten query: {txt}")
        
        return txt

    # ---------------------------------------------------------------------
    def clean_text(self, text: str) -> str:
        """Add punctuation & capitalisation with empty check"""
        if not text.strip():
            return text
        try:
            return self.punctuate.restore_punctuation(text)
        except Exception as e:
            print(f"⚠️ Punctuation error: {str(e)}")
            return text  # Return original if punctuation fails

    # ---------------------------------------------------------------------
    def close(self):
        """Graceful shutdown (call from FastAPI /startup)."""
        if hasattr(self.generator, "engine"):
            # vLLM backend
            shutdown_fn = getattr(self.generator.engine, "shutdown", None) or getattr(
                self.generator.engine, "close", None
            )
            if shutdown_fn:
                shutdown_fn()
        self.retriever.qdrant_store.close()


# ────────────────────────────────────────────────────────────────────────────
# local smoke-test
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    async def main():
        bot = MediMaven(Backend.VLLM)
        

        res = await bot.answer("What causes type 2 diabetes?", n_ctx=5)
        print("Answer:", res["answer"])
        print("Citations:", res["citations"])
        print("Latency (s):", res["latency_s"])

        bot.close()

    asyncio.run(main())

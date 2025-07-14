# ──────────────────────────────────────────────────────────────────────────────
# generate.py – unified Generator class (Transformers | AWQ | vLLM)
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
from enum import Enum, auto
from pathlib import Path
import asyncio, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from awq import AutoAWQForCausalLM

from backend.utils import Timer, best_device
from backend.services.model_selector import pick_model

try:
    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

    HAS_VLLM = True
except ImportError:
    print("vllm not installed")
    HAS_VLLM = False

LLM_PATH, quant, dtype = pick_model()   # pick_model already uses config.FP16_DIR, etc.

# print(f"LLM_PATH --------: {LLM_PATH}")
class Backend(Enum):
    TRANSFORMERS = auto()
    AWQ          = auto()
    VLLM         = auto()

class Generator:
    """ Unified interface for LLM generation using different backends."""
    def __init__(self, backend: Backend = Backend.VLLM):
        self.backend = backend
        self.device = best_device()
        if backend == Backend.TRANSFORMERS:
            print('using Tranformers as llm backend')
            self.tok   = AutoTokenizer.from_pretrained(LLM_PATH, use_fast=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                LLM_PATH, torch_dtype=torch.float16,
                low_cpu_mem_usage=True, device_map="auto")
        elif backend == Backend.AWQ:
            print('using AWQ as llm backend')
            self.tok   = AutoTokenizer.from_pretrained(LLM_PATH, use_fast=True)
            self.model = AutoAWQForCausalLM.from_quantized(
                LLM_PATH, fuse_layers=True, device="cuda")
        elif backend == Backend.VLLM:
            print('using VLM as llm backend')

            if not HAS_VLLM:
                raise ImportError("pip install vllm")

            self.params = SamplingParams(repetition_penalty=1.1, 
                                         temperature=0.7, 
                                         top_p=0.95,
                                         skip_special_tokens=True)

            args = AsyncEngineArgs(
                model=LLM_PATH, task="generate", quantization=quant,
                dtype=dtype, max_model_len=4096, enforce_eager=True,
                disable_log_requests=True, disable_log_stats=True, 
                gpu_memory_utilization=0.8,  # 80% GPU memory
                
                )

            self.engine = AsyncLLMEngine.from_engine_args(args)

        else:
            raise ValueError(backend)

    # ---- public -----------------------------------------------------------
    async def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        if self.backend == Backend.VLLM:
            # Clone params to avoid mutation issues
            params = self.params.clone()
            params.max_tokens = max_new_tokens
            stream = self.engine.generate(prompt, params, request_id="request-1")
            text = ""
            async for o in stream:
                text = o.outputs[0].text
            return text.strip()

        # For other backends
        toks = self.tok(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **toks, 
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            num_return_sequences=1,
            pad_token_id=self.tok.eos_token_id  # Ensure proper termination
        )
        decoded = self.tok.decode(
            out[0][toks["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # Fallback for empty response
        if not decoded:
            print("⚠️ Empty response, using simple fallback")
            return "I'm having trouble generating a response. Could you rephrase your question?"
            
        return decoded

    async def stream(self, prompt: str, max_new_tokens: int = 128):
        """Yield only the NEW part of each incremental chunk."""
        if self.backend != Backend.VLLM:
            # Fallback for non-streaming backends
            yield await self.generate(prompt, max_new_tokens)
            return

        params = self.params.clone()
        params.max_tokens = max_new_tokens
        prev = ""
        async for chunk in self.engine.generate(prompt, params, request_id="request-1"):
            cur = chunk.outputs[0].text
            delta = cur[len(prev):]
            prev = cur
            if delta:
                yield delta # Yield only new tokens
                
                
    
    def close(self):
        if self.backend == Backend.VLLM:
            self.engine.shutdown()  
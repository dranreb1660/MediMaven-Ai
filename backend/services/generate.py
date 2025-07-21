from __future__ import annotations
from enum import Enum, auto
import asyncio, torch, uuid, logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from awq import AutoAWQForCausalLM
from backend.utils import Timer, best_device
from backend.services.model_selector import pick_model

logger = logging.getLogger(__name__)

try:
    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
    HAS_VLLM = True
except ImportError:
    logger.warning("vllm not installed, falling back to transformers")
    HAS_VLLM = False

LLM_PATH, quant, dtype = pick_model()

class Backend(Enum):
    TRANSFORMERS = auto()
    AWQ = auto()
    VLLM = auto()

class Generator:
    """Unified LLM interface with fixed async handling"""
    
    def __init__(self, backend: Backend = Backend.VLLM):
        self.backend = backend
        self.device = best_device()
        self._initialized = False
        self.tok = None
        self.model = None
        self.engine = None
        
        try:
            self._initialize_backend()
            self._initialized = True
            logger.info(f"✅ Generator initialized with {backend.name}")
            
        except Exception as e:
            logger.error(f"Generator initialization failed: {e}")
            self._cleanup()
            raise RuntimeError(f"Failed to initialize {backend.name} backend: {e}")
    
    def _initialize_backend(self):
        """Initialize the selected backend with proper error handling"""
        if self.backend == Backend.VLLM and HAS_VLLM:
            self._init_vllm()
        elif self.backend == Backend.AWQ:
            self._init_awq()
        else:
            self._init_transformers()
    
    def _init_vllm(self):
        """Initialize vLLM with conservative settings"""
        self.params = SamplingParams(
            repetition_penalty=1.1, temperature=0.7, top_p=0.95,
            skip_special_tokens=True
        )
        
        args = AsyncEngineArgs(
            model=LLM_PATH, task="generate", quantization=quant,
            dtype=dtype, max_model_len=4096, enforce_eager=True,
            disable_log_requests=True, disable_log_stats=True,
            gpu_memory_utilization=0.75,  # Conservative for production
            max_num_seqs=8,  # Limit concurrent sequences
        )
        
        self.engine = AsyncLLMEngine.from_engine_args(args)
    
    def _init_awq(self):
        """Initialize AWQ with error handling"""
        self.tok = AutoTokenizer.from_pretrained(LLM_PATH, use_fast=True)
        if not self.tok.pad_token:
            self.tok.pad_token = self.tok.eos_token
            
        self.model = AutoAWQForCausalLM.from_quantized(
            LLM_PATH, fuse_layers=True, device="cuda"
        )
    
    def _init_transformers(self):
        """Initialize transformers with memory optimization"""
        self.tok = AutoTokenizer.from_pretrained(LLM_PATH, use_fast=True)
        if not self.tok.pad_token:
            self.tok.pad_token = self.tok.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_PATH, torch_dtype=torch.float16,
            low_cpu_mem_usage=True, device_map="auto"
        )
    
    async def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        """Generate with robust error handling and fallbacks"""
        if not self._initialized:
            return "Generator not available. Please try again later."
        
        if not prompt or not prompt.strip():
            return "Please provide a valid question."
        
        # Truncate overly long prompts
        if len(prompt) > 8000:
            prompt = prompt[:8000] + "..."
        
        try:
            if self.backend == Backend.VLLM and self.engine:
                return await self._generate_vllm(prompt, max_new_tokens)
            else:
                return await self._generate_local(prompt, max_new_tokens)
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "I'm experiencing technical difficulties. Please try rephrasing your question."
    
    async def _generate_vllm(self, prompt: str, max_new_tokens: int) -> str:
        """Fixed vLLM generation with proper async handling"""
        try:
            params = self.params.clone()
            params.max_tokens = max_new_tokens
            request_id = str(uuid.uuid4())
            
            # Get the async generator
            stream = self.engine.generate(prompt, params, request_id=request_id)
            
            # Proper timeout handling for async iteration
            text = ""
            try:
                async def _iterate_with_timeout():
                    nonlocal text
                    async for output in stream:
                        text = output.outputs[0].text
                    return text
                
                # Apply timeout to the entire iteration
                result = await asyncio.wait_for(_iterate_with_timeout(), timeout=30.0)
                return result.strip() or "I couldn't generate a proper response."
                
            except asyncio.TimeoutError:
                logger.error("vLLM generation timed out")
                return "Response generation timed out. Please try a shorter question."
                
        except Exception as e:
            logger.error(f"vLLM generation error: {e}")
            return "I'm experiencing technical difficulties. Please try rephrasing your question."
    
    async def _generate_local(self, prompt: str, max_new_tokens: int) -> str:
        """Local model generation with memory management"""
        def _sync_generate():
            toks = self.tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
            toks = toks.to(self.device)
            
            with torch.no_grad():
                out = self.model.generate(
                    **toks, max_new_tokens=max_new_tokens,
                    do_sample=True, temperature=0.7,
                    pad_token_id=self.tok.eos_token_id,
                    use_cache=True
                )
            
            # Clear input tokens from output
            new_tokens = out[0][toks["input_ids"].shape[1]:]
            decoded = self.tok.decode(new_tokens, skip_special_tokens=True).strip()
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return decoded or "I couldn't generate a proper response."
        
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_generate)
    
    async def stream(self, prompt: str, max_new_tokens: int = 128):
        """Fixed streaming generation"""
        if not self._initialized or self.backend != Backend.VLLM:
            # Fallback: yield complete response
            response = await self.generate(prompt, max_new_tokens)
            words = response.split()
            for word in words:
                yield word + " "
                await asyncio.sleep(0.01)
            return
        
        try:
            params = self.params.clone()
            params.max_tokens = max_new_tokens
            request_id = str(uuid.uuid4())
            
            # Get the async generator
            stream = self.engine.generate(prompt, params, request_id=request_id)
            
            prev_text = ""
            try:
                # Proper async iteration without timeout wrapper
                async for chunk in stream:
                    current_text = chunk.outputs[0].text
                    delta = current_text[len(prev_text):]
                    if delta:
                        yield delta
                    prev_text = current_text
                    
            except Exception as e:
                logger.error(f"Streaming iteration failed: {e}")
                yield "I'm experiencing difficulties with streaming. "
                
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield "I'm experiencing difficulties with streaming. "
    
    def _cleanup(self):
        """Safe cleanup of resources"""
        try:
            if hasattr(self, 'engine') and self.engine:
                if hasattr(self.engine, 'shutdown'):
                    self.engine.shutdown()
                    
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self.model = None
            self.tok = None
            self.engine = None
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    def close(self):
        """Public cleanup method"""
        self._cleanup()
        self._initialized = False
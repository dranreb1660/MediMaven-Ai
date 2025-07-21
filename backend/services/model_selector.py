import torch
import logging
from pathlib import Path
from backend.app import config

logger = logging.getLogger(__name__)

FP16_DIR = config.FP16_DIR
AWQ_DIR = config.AWQ_DIR
MIN_FP16_MEM = config.MIN_FP16_GIB * 2**30

def pick_model() -> tuple[str, str | None, str]:
    """
    Select optimal model configuration based on available hardware.
    
    Returns (model_dir, quantization, dtype)
      • quantization = None  → use fp16/bf16 weights
      • quantization = "awq" → use 4-bit AWQ quantization
    """
    try:
        # CPU/MPS fallback
        if not torch.cuda.is_available():
            logger.info("🖥  CUDA not available, using AWQ quantization")
            return str(AWQ_DIR), "awq", "half"
        
        # Get GPU properties with error handling
        try:
            props = torch.cuda.get_device_properties(0)
            gpu_mem = props.total_memory
            cc_major = props.major
            name = props.name
        except Exception as e:
            logger.warning(f"Failed to get GPU properties: {e}")
            return str(AWQ_DIR), "awq", "half"
        
        logger.info(f"🖥  GPU: {name} ({gpu_mem/2**30:.1f} GiB, CC {cc_major})")
        
        # Validate model directories exist
        fp16_exists = Path(FP16_DIR).exists()
        awq_exists = Path(AWQ_DIR).exists()
        
        if not fp16_exists and not awq_exists:
            raise FileNotFoundError("Neither FP16 nor AWQ model directories found")
        
        # Decision logic with safety checks
        use_fp16 = (
            gpu_mem >= MIN_FP16_MEM and 
            cc_major >= 8 and 
            fp16_exists
        )
        
        if use_fp16:
            logger.info(f"💾 Using FP16 weights ({gpu_mem/2**30:.1f} GiB available)")
            return str(FP16_DIR), None, "half"
        else:
            if not awq_exists:
                if fp16_exists:
                    logger.warning("AWQ not available, falling back to FP16")
                    return str(FP16_DIR), None, "half"
                else:
                    raise FileNotFoundError("No model weights available")
            
            logger.info(f"💾 Using AWQ quantized weights ({gpu_mem/2**30:.1f} GiB available)")
            return str(AWQ_DIR), "awq", "half"
            
    except Exception as e:
        logger.error(f"Model selection failed: {e}")
        # Final fallback
        if Path(AWQ_DIR).exists():
            logger.info("Using AWQ as fallback")
            return str(AWQ_DIR), "awq", "half"
        elif Path(FP16_DIR).exists():
            logger.info("Using FP16 as fallback")
            return str(FP16_DIR), None, "half"
        else:
            raise RuntimeError("No model weights available for fallback")
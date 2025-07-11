import torch
from backend.app import config
FP16_DIR = config.FP16_DIR
AWQ_DIR  = config.AWQ_DIR
MIN_FP16_MEM = config.MIN_FP16_GIB * 2**30                   # 22 GiB ≈ fits A10/A100

def pick_model() -> tuple[str, str | None, str]:
    """
    Decide at runtime which weight folder + quantisation flag to use.

    Returns (model_dir, quantization, dtype)
      • quantization = None  → load fp16/bf16
      • quantization = "awq" → use 4-bit AWQ
    """
    if not torch.cuda.is_available():
        # CPU / MPS fallback → keep quantised to save RAM
        return str(AWQ_DIR), "awq", "half"

    props = torch.cuda.get_device_properties(0)
    gpu_mem = props.total_memory          # bytes
    cc_major = props.major                # 8 = Ampere, 7 = Turing, 9 = Hopper
    name = props.name
    print(f"🖥  GPU: {name}  ({gpu_mem/2**30:.1f} GiB, CC {cc_major})")

    # prefer fp16 on Ampere/Hopper with ≥22 GiB
    if gpu_mem >= MIN_FP16_MEM and cc_major >= 8:
        print(f"💾 Using fp16 weights on {name} ({gpu_mem/2**30:.1f} GiB)" )
        return str(FP16_DIR), None, "half"          # pure fp16 safetensors
    # else fallback to AWQ (works on any 8 GB+ GPU)
    print(f"💾 Using AWQ quantised weights on {name} ({gpu_mem/2**30:.1f} GiB)")
    return str(AWQ_DIR), "awq", "half"

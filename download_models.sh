#!/usr/bin/env bash
set -euo pipefail

echo "📦  MediMaven model bootstrap - Fixed version"

MODEL_ROOT=${MODEL_DIR:-/app/models/v1.1}
mkdir -p "$MODEL_ROOT"

python3 - "$MODEL_ROOT" <<'PY'
import os, sys
from huggingface_hub import snapshot_download
import shutil

def validate_model(model_dir, expected_files):
    """Check if all expected files exist and are non-empty"""
    if not os.path.exists(model_dir):
        return False
    
    for file in expected_files:
        file_path = os.path.join(model_dir, file)
        if not os.path.exists(file_path):
            print(f"  ❌ Missing: {file}")
            return False
        if os.path.getsize(file_path) == 0:
            print(f"  ❌ Empty file: {file}")
            return False
    
    return True

root = sys.argv[1]
token = os.getenv("HF_TOKEN") if os.getenv("HF_TOKEN") else None

# Define models with their expected critical files
models = {
    "llama3_8b_awq": {
        "repo": "dranreb1660/medimaven-llama3-8b-awq",
        "expected_files": ["config.json", "tokenizer_config.json", "model.safetensors.index.json"],
        "priority": 1  # High priority - needed for LLM
    },
    "ltr_cross-encoder": {
        "repo": "dranreb1660/medimaven-reranker-bge-cross-encoder",
        "expected_files": ["config.json", "model.safetensors"],
        "priority": 2  # Critical - needed for ranking
    },
    "ltr_lambdamart": {
        "repo": "dranreb1660/medimaven-ltr-lambdamart",
        "expected_files": ["ltr_lambdamart.txt"],
        "priority": 2  # Critical - needed for ranking
    },
    "llama3_8b_fp16": {
        "repo": "dranreb1660/medimaven-llama3-8b-fp16", 
        "expected_files": ["config.json", "tokenizer_config.json"],
        "priority": 3  # Low priority - only needed for high-memory GPUs
    }
}

# Sort models by priority
sorted_models = sorted(models.items(), key=lambda x: x[1]["priority"])

for subdir, model_info in sorted_models:
    local_dir = os.path.join(root, subdir)
    repo = model_info["repo"]
    expected_files = model_info["expected_files"]
    
    if validate_model(local_dir, expected_files):
        print(f"✅  {subdir} - validation passed")
        # Run post-processing if needed
        if "post_process" in model_info:
            model_info["post_process"](local_dir)
    else:
        print(f"⬇️  {repo}@v1.1 → {local_dir}")
        # Remove incomplete directory if it exists
        if os.path.exists(local_dir):
            shutil.rmtree(local_dir)
        
        try:
            snapshot_download(
                repo, 
                revision="v1.1",
                local_dir=local_dir,
                token=token
            )
            
            # Run post-processing if needed
            if "post_process" in model_info:
                model_info["post_process"](local_dir)
            
            # Validate after download
            if validate_model(local_dir, expected_files):
                print(f"✅  {subdir} - download and validation successful")
            else:
                print(f"❌  {subdir} - download completed but validation failed")
                
        except Exception as e:
            print(f"❌  Failed to download {subdir}: {str(e)}")
            # For critical models, this is a fatal error
            if model_info["priority"] <= 2:
                print(f"💥  Critical model {subdir} failed to download!")
                # Continue with other models but mark as incomplete

print("🎉  Model bootstrap completed")
PY
#!/bin/bash

# Check required env vars
REQUIRED_VARS=("WANDB_API_KEY" "HF_TOKEN")
for var in "${REQUIRED_VARS[@]}"; do
  if [[ -z "${!var}" ]]; then
    echo "❌ Environment variable $var not set. Exiting."
    exit 1
  fi
done

# Auto-download models if missing
if [ ! -d "/app/models/llama_finetuned/gptqmodel_4bit" ]; then
  echo "⬇️ Models missing, downloading now..."
  bash download_models.sh
fi

echo "🔐 Starting auxiliary services..."
python start_services.py

echo "🚀 Launching FastAPI..."
exec uvicorn src.backend.app.main:app --host 0.0.0.0 --port 8000

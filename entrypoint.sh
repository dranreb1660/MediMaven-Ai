#!/usr/bin/env bash
set -euo pipefail

# ─── 1) Sanity checks ────────────────────────────────────────────
# HF_TOKEN is now optional since models are public
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "🔑 HF_TOKEN provided - using authenticated access"
else
  echo "ℹ️  No HF_TOKEN - using public access"
fi

# WANDB optional – warn only
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "⚠️  WANDB_API_KEY not set – monitoring disabled"
fi

# ─── 2) Download (if missing) ────────────────────────────────────
if [ ! -d "${MODEL_DIR:-/app/models/v1.1}/llama3_8b_awq" ]; then
  echo "⬇️  First-run: downloading models…"
  bash /app/download_models.sh
fi

# ─── 3) Start aux services (login, seeding, etc.) ────────────────
python /app/start_services.py

# ─── 4) Launch FastAPI (vLLM inside) ─────────────────────────────
exec uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
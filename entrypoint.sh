#!/bin/bash

echo "🔐 Starting service setup..."
python start_services.py

echo "✅ Setup done. Launching FastAPI..."
exec uvicorn src.backend.app.main:app --host 0.0.0.0 --port 8000 
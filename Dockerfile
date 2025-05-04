########################################
# 1️⃣ builder stage – install deps
########################################
FROM python:3.10-slim AS builder

ARG TORCH_CUDA_VERSION=cu122          # match wheel version you want


ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/app

# ---- system libs -----------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget curl build-essential git && \
    rm -rf /var/lib/apt/lists/*

# ---- python deps -----------------------------------------------------------
COPY reqs.txt .
RUN pip install -r reqs.txt && \
    pip install torch torchvision torchaudio
RUN pip install -v gptqmodel --no-build-isolation

########################################
# 2️⃣ runtime image (tiny)
########################################
FROM python:3.10-slim

WORKDIR /app

# copy venv + compiled wheels from builder
COPY --from=builder /usr/local /usr/local

# remove any lingering wandb credentials
RUN rm -f /root/.netrc

# copy only application code
COPY src/ ./src
COPY pipelines/ ./pipelines
COPY entrypoint.sh .
COPY start_services.py .

RUN chmod +x /app/entrypoint.sh

EXPOSE 8000
HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]

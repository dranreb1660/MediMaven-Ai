# syntax=docker/dockerfile:1.7   ────────────────────────────────────────────
# Multi-arch build (linux/amd64 + linux/arm64)
#
# • Stage 1  ➜  builder: CUDA runtime + Python + full venv + compilers
# • Stage 2  ➜  runtime: CUDA runtime + Python + copy venv + app code
# ---------------------------------------------------------------------------

############################
# 1️⃣  Builder stage
############################
ARG CUDA_IMAGE=nvidia/cuda:12.6.1-cudnn-runtime-ubuntu22.04
ARG TARGETPLATFORM=linux/amd64
FROM --platform=$TARGETPLATFORM ${CUDA_IMAGE} AS builder

## ---- System deps (compiler, Python) --------------------------
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        build-essential git curl && \
    ln -s /usr/bin/python3.11 /usr/local/bin/python && \
    ln -s /usr/bin/pip3 /usr/local/bin/pip && \
    rm -rf /var/lib/apt/lists/*

## ---- Python virtual-env --------------------------------------
ENV VENV_PATH=/opt/venv
RUN python -m venv $VENV_PATH
ENV PATH=$VENV_PATH/bin:$PATH

## ---- Copy requirements first to leverage Docker cache -------
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip wheel packaging
# Install PyTorch first (required for flash-attn)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Install flash-attn wheel directly with force-reinstall to bypass version checks
RUN pip install --force-reinstall --no-deps https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3%2Bcu12torch2.5cxx11abiTRUE-cp311-cp311-linux_x86_64.whl

RUN pip install --upgrade -r /tmp/requirements.txt

# Clean up build artifacts and reduce venv size
RUN pip cache purge

############################
# 2️⃣  Runtime stage
############################
FROM --platform=$TARGETPLATFORM ${CUDA_IMAGE} AS runtime

# Install Python 3.11 in runtime stage (CRITICAL FIX #1)
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv \
        && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.11 /usr/local/bin/python

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_DIR=/app/models/v1.1 \
    VENV_PATH=/opt/venv \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_TOKEN="" \
    WANDB_API_KEY=""

# ---- Copy pre-built virtual-env -------------------------------
COPY --from=builder $VENV_PATH $VENV_PATH
ENV PATH=$VENV_PATH/bin:$PATH

# ---- App directory & code (CRITICAL FIX #2: Correct structure) ----
WORKDIR /app
# Copy to correct backend structure that the code expects
COPY ./backend ./backend
COPY ./data/final/chunk_embeddings.npy data/final/
COPY ./data/final/kb_chunks.parquet data/final/
COPY download_models.sh entrypoint.sh start_services.py ./
RUN chmod +x download_models.sh entrypoint.sh

# ---- Expose FastAPI port --------------------------------------
EXPOSE 8000

# ---- Entrypoint ------------------------------------------------
ENTRYPOINT ["bash", "/app/entrypoint.sh"]

# Stage 1: Install dependencies
FROM python:3.10-slim AS base

WORKDIR /app

# Install system and CUDA dependencies
RUN apt-get update && \
    apt-get install -y wget build-essential git curl && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-12-4 && \
    rm -rf /var/lib/apt/lists/* && \
    rm -f cuda-keyring_1.1-1_all.deb

# Print CUDA version for sanity check
RUN nvcc --version && nvidia-smi || echo "CUDA not available in build phase"

# Environment variables
ENV CUDA_HOME=/usr/local/cuda-12.4
ENV TORCH_CUDA_ARCH_LIST="8.9"

# Install Python deps early for caching

COPY reqs.txt .
RUN pip install --no-cache-dir -r reqs.txt
RUN pip install --no-cache-dir torch torchvision torchaudio

# Install your local model package last (less caching likely)
RUN pip install -v gptqmodel --no-build-isolation

# Stage 2: Runtime image
FROM python:3.10-slim

WORKDIR /app

# Copy Python + CUDA environment from stage 1
COPY --from=base /usr/local /usr/local
COPY --from=base /app /app

# Copy the rest of the codebase
COPY . .

# Add health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl --fail http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Entrypoint
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]

#!/usr/bin/env bash

# Check for Docker Buildx
if ! docker buildx version &> /dev/null; then
    echo "Docker Buildx is not installed."
    exit 1
fi

# Create a new builder if needed
docker buildx create --use || true

# Build and push multi-arch image
docker buildx build \
  --platform linux/amd64 \
  -t phade160/medimaven-rag-api:1.1.0 \
  --push .
  
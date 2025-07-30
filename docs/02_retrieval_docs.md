# 🔍 MediMaven Retrieval Pipeline Documentation

![CI](https://img.shields.io/badge/Built_with-Docker-blue) ![AWS](https://img.shields.io/badge/Cloud-AWS-%23FF9900) ![License](https://img.shields.io/badge/License-Apache%202.0-green)

**Goal**: Return the most relevant medical passages in < 100 ms P95, ready for LLM grounding.

## Table of Contents

1.  [Architecture Overview](#architecture-overview)
2.  [Retrieval Components](#retrieval-components)
3.  [Data Ingestion](#data-ingestion)
4.  [Reciprocal Rank Fusion (RRF)](#reciprocal-rank-fusion-rrf)
5.  [Performance Metrics](#performance-metrics)
6.  [Monitoring & Alerts](#monitoring--alerts)
7.  [Future Work](#future-work)

## 🏢 Architecture Overview

| Layer | Purpose | Tech |
|-------|---------|------|
| **BM25Store** | High-recall lexical search | `rank_bm25` |
| **QdrantStore (in-memory)** | 768-D semantic search | Qdrant container (no persistent volume) |
| **RRF Fusion** | Blend dense + sparse results | Custom implementation |
| **(optional) LTR Ranker** | LambdaMART + Cross-Encoder | LightGBM & `bge-reranker-large` |

---

## 1 · Architecture

```mermaid
graph TD
  QBM25[BM25 scores] -->|top 200| RRF
  QDense[Dense vectors<br>Qdrant HNSW] -->|top 300| RRF
  RRF -->|top 60 fused| LTR[LambdaMART + CE<br>(optional)]
  LTR --> API
```

## 2 · Data ingestion
```python

from datasets import load_dataset
from qdrant_client import QdrantClient
from backend.services.vector_stores import QdrantStore, BM25Store

chunks = load_dataset("bernard-kyei/medimaven-qa-data",
                      "kb_chunks", split="train")

# Build / rebuild stores (runs in CI and local)
bm25 = BM25Store(pkl_path="indexes/bm25.pkl",
                 chunks_df=chunks, force_rebuild=False)

qd   = QdrantStore(collection_name="medimaven_chunks",
                   embeddings_path="embeds.npy",
                   metadata_path="meta.parquet",
                   storage_path=None,        # in-memory
                   force_rebuild=True)
```

## 4 · Reciprocal Rank Fusion (RRF)

RRF blends the two result sets—dense embeddings and BM25 scores—into a single, robust ranking:

$$
\text{score}_{\mathrm{RRF}}(d)\;=\;
\sum_{s \in \{\text{dense},\;\text{bm25}\}}
\frac{w_s}{k_{\mathrm{rrf}} + \mathrm{rank}_s(d)}
$$

Implementation lives in `Retriever.fuse()` with defaults:

```python
k_rrf   = 60
w_dense = 2   # weight for semantic (dense) rankings
w_sparse = 1  # weight for BM25 sparse rankings
```
**Why RRF?**

- Quick to compute (pure Python).

- Robust to noisy scores—no need to calibrate BM25 vs cosine scales.

- Empirically boosts nDCG@5 by ≈ +0.09 over the best single channel in our v1.1 evaluation.

## 5 · End-to-end latency
| Step                      | P95 (ms) | Notes              |
| ------------------------- | -------- | ------------------ |
| BM25 tokenization + score | 22       | in-RAM NumPy       |
| Qdrant HNSW search        | 41       | `ef_search = 64`   |
| RRF fusion                | 2        | pure Python        |
| **Total retrieval**       | **≈ 65** | measured on M3-Pro |

## 6 · Monitoring & alerts
Weights & Biases — query logs, recall@10 dashboard.

Prometheus — Qdrant QPS, HNSW memory.

PagerDuty alert if P95 > 120 ms or recall dips > 5 %.

## 7 · Future work
ColBERT-v2 for contextual sparse vectors.

Approx-BM25 via Qdrant 2.0 PRF filters.

Online hard-negative mining using low-confidence user queries.
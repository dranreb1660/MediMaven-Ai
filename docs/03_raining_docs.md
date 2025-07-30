# 🩺 MediMaven Model Training Documentation

![CI](https://img.shields.io/badge/Built_with-Docker-blue) ![AWS](https://img.shields.io/badge/Cloud-AWS-%23FF9900) ![License](https://img.shields.io/badge/License-Apache%202.0-green)

This document provides a comprehensive overview of the training procedures for the MediMaven project. It is intended for machine learning engineers, researchers, and anyone interested in understanding the model training and fine-tuning processes.

## Table of Contents

1.  [Introduction](#introduction)
2.  [Learning-to-Rank (LTR) Models](#learning-to-rank-ltr-models)
    *   [LambdaMART](#lambdamart)
    *   [BGE Cross-Encoder](#bge-cross-encoder)
3.  [Large Language Model (LLM) Fine-Tuning](#large-language-model-llm-fine-tuning)
    *   [Llama-3 8B QLoRA Fine-tuning](#llama-3-8b-qlora-fine-tuning)
    *   [Llama-3 8B AWQ Quantization](#llama-3-8b-awq-quantization)

## 🚀 Introduction

The MediMaven v1.1 project employs a sophisticated multi-stage approach to answer medical questions using state-of-the-art deep learning techniques. The system combines a robust retrieval-augmented generation (RAG) pipeline that first retrieves relevant medical documents using hybrid search, then re-ranks them with cascaded learning-to-rank models, and finally generates evidence-based answers using a fine-tuned Llama 3 8B model.

The training process consists of three main stages:
1. **Data Processing**: Converting raw medical content into structured Q&A pairs with hard negatives
2. **Learning-to-Rank Training**: Training both traditional ML (LambdaMART) and deep learning (BGE Cross-Encoder) ranking models
3. **Language Model Fine-tuning**: Domain adaptation of Llama 3 8B using QLoRA and subsequent AWQ quantization for production deployment

## 🎯 Learning-to-Rank (LTR) Models

The LTR models are responsible for ranking the retrieved medical documents to ensure the most relevant documents are presented to the user. We train two types of LTR models: a LambdaMART model and a BGE cross-encoder.

### LambdaMART

**Notebook**: `04a_ltr_train_lambdamart.ipynb`

The LambdaMART model is a gradient-boosted decision tree model designed for ranking tasks using LightGBM. It serves as the first stage in our cascade ranking approach, providing fast initial re-ranking of retrieved documents.

| Step | Description |
| :--- | :--- |
| **Data Loading** | Hard negatives dataset (`hard_negatives.parquet`) with **634,350** query-document pairs |
| **Feature Engineering** | Three key features extracted: `bm25_score`, `dense_cosine`, `chunk_length` |
| **Data Splitting** | Training: **570,915** examples, Validation: **63,435** examples (90/10 split) |
| **Model Training** | LightGBM with LambdaRank objective, optimized for NDCG evaluation metric |
| **Performance** | Optimized for speed and initial relevance filtering (top 100 candidates) |
| **Model Saving** | Serialized model saved for production RAG pipeline usage |

### BGE Cross-Encoder

**Notebook**: `04b_ltr_train_colbert.ipynb`

The BGE cross-encoder is a deep learning model that performs the second stage of re-ranking. It fine-tunes a pre-trained `BAAI/bge-reranker-base` model on triplets of (query, positive document, negative document) to learn a highly accurate relevance scoring function.

| Step | Description |
| :--- | :--- |
| **Data Loading** | **634,350** hard negative triplets from `hard_negatives.parquet` |
| **Data Splitting** | Training: **570,914** examples, Validation: **63,436** examples (90/10 split) |
| **Model Fine-tuning** | Fine-tuned for 1 epoch on the training set. Achieved a training loss of **0.5028** at step 500. |
| **Evaluation** | Achieved an **NDCG@10** of **0.85+** on the validation set, demonstrating high ranking quality. |
| **Model Saving** | The fine-tuned model is saved and logged to Weights & Biases as a versioned artifact. |

## 🧠 Large Language Model (LLM) Fine-Tuning

The LLM generates final answers based on ranked documents using the Meta-Llama-3-8B-Instruct model.

### Llama-3 8B QLoRA Fine-tuning

**Notebook**: `05_llama3_qlora.ipynb`

QLoRA (Quantized Low-Rank Adaptation) is used to fine-tune the `meta-llama/Meta-Llama-3-8B-Instruct` model on the medical dataset in a memory-efficient manner. This allows the model to be trained on a single consumer-grade GPU.

| Step | Description |
| :--- | :--- |
| **Data Preparation** | **143,280** examples from the merged QA and knowledge base datasets, formatted for instruction-based fine-tuning. |
| **Data Splitting** | Training: **128,952** examples, Validation: **14,328** examples (90/10 split). |
| **Model Fine-tuning** | Fine-tuned for 1 epoch with the following configuration:<br>• **Quantization**: 4-bit NormalFloat (NF4) with BitsAndBytes.<br>• **Attention**: Flash Attention 2 for improved performance.<br>• **LoRA**: `r=64`, `lora_alpha=32`.<br>• **Final Validation Loss**: **0.4711**. |
| **Model Saving** | The resulting PEFT adapter is saved as the `llama3_qlora_adapter` artifact in Weights & Biases. |

### Llama-3 8B AWQ Quantization

**Notebook**: `07_llama3_awq_quantization.ipynb`

AWQ (Activation-aware Weight Quantization) is used to quantize the fine-tuned Llama 3 model to 4-bit for high-performance inference. This significantly reduces the model's memory footprint and improves inference speed without a substantial drop in accuracy.

| Step | Description |
| :--- | :--- |
| **Model Loading** | The base `meta-llama/Meta-Llama-3-8B-Instruct` model is loaded along with the fine-tuned QLoRA adapter. |
| **Quantization** | The model is quantized using the AWQ algorithm with a group size of 128. A calibration dataset of 1,024 samples is used to minimize quantization error. |
| **Model Saving** | The 4-bit quantized model is saved and prepared for deployment in the RAG pipeline. |

---

This concludes the overview of the model training and fine-tuning processes for the MediMaven project. For more detailed information, please refer to the individual training notebooks.

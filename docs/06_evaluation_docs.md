# 📊 MediMaven v1.1 Model Evaluation Documentation

![CI](https://img.shields.io/badge/Built_with-Docker-blue) ![AWS](https://img.shields.io/badge/Cloud-AWS-%23FF9900) ![License](https://img.shields.io/badge/License-Apache%202.0-green)

This document provides a summary of the key evaluation metrics for the models trained in the MediMaven v1.1 project. It is intended to provide a quantitative measure of the performance of the learning-to-rank and language models.

## Table of Contents

1.  [Learning-to-Rank (LTR) Evaluation](#learning-to-rank-ltr-evaluation)
    *   [BGE Cross-Encoder](#bge-cross-encoder)
    *   [LambdaMART](#lambdamart)
2.  [Language Model (LLM) Fine-Tuning Evaluation](#language-model-llm-fine-tuning-evaluation)
    *   [Llama-3 8B QLoRA Fine-tuning](#llama-3-8b-qlora-fine-tuning)

## 🎯 Learning-to-Rank (LTR) Evaluation

The performance of the LTR models is critical for ensuring that the most relevant documents are passed to the language model. We evaluate our LTR models using the Normalized Discounted Cumulative Gain (NDCG), a standard metric for ranking quality.

### BGE Cross-Encoder

**Notebook**: [`04b_ltr_train_bge-ce.ipynb`](../notebooks/v1.1/04b_ltr_train_bge_ce.ipynb)

The BGE cross-encoder is the final stage of our re-ranking pipeline. Its ability to accurately rank documents is measured by its NDCG@10 score on the validation set.

| Metric | Value | Description |
| :--- | :--- | :--- |
| **NDCG@10** | **0.85+** | The NDCG score at a cutoff of 10 documents, indicating very high ranking quality. |
| **Training Loss** | **0.5028** | The cross-entropy loss at step 500 of the fine-tuning process. |

### LambdaMART

**Notebook**: [`04b_ltr_train_bge-ce.ipynb`](../notebooks/v1.1/04a_ltr_train_lambdamart.ipynb)

The LambdaMART model provides the initial, fast re-ranking of documents. It is trained to optimize the NDCG metric directly, ensuring that the top candidates passed to the cross-encoder are of high quality.

| Metric | Description |
| :--- | :--- |
| **Objective** | **LambdaRank** with **NDCG** evaluation metric. |
| **Purpose** | To quickly filter and rank the top ~100 documents for the cross-encoder. |

## 🧠 Language Model (LLM) Fine-Tuning Evaluation

The performance of the fine-tuned Llama 3 model is evaluated by its ability to learn the patterns in the medical Q&A dataset. This is measured by the validation loss during the fine-tuning process.

### Llama-3 8B QLoRA Fine-tuning

**Notebook**: [`05_llama3_qlora.ipynb`](../notebooks/v1.1/05_llama3_qlora.ipynb)

The validation loss indicates how well the model is generalizing to unseen data. A lower loss indicates better performance.

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Validation Loss** | **0.4711** | The final cross-entropy loss on the validation set after one epoch of fine-tuning. |

---

This concludes the overview of the model evaluation for the MediMaven project. For more detailed information, please refer to the individual training notebooks.

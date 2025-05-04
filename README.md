# 🩺 MediMaven — Production‑grade Medical RAG Assistant

![CI](https://img.shields.io/badge/Built_with-Docker-blue) ![AWS](https://img.shields.io/badge/Cloud-AWS-%23FF9900) ![License](https://img.shields.io/badge/License-Apache%202.0-green)

  - <a href="https://www.medimaven-ai.com">Web client</a> 
  - <a href="https://api.medimaven-ai.com/docs">Swagger Ui</a> 
  - <a href="docs/infra-runbook.md">Run‑book</a>
  - [Jump to Demo](#✨-demo)
## 🚀 Overview
**MediMaven** is an **end-to-end medical Q&A chatbot** that uses **Large Language Models (LLMs)** and **Retrieval-Augmented Generation (RAG)** to provide **accurate, real-time** responses about a wide range of general health topics. Built with **PyTorch & Hugging Face**, **Learning-to-Rank (LTR)**, and **Airflow ETL**, it’s designed to **ingest and analyze** medical data from trusted sources (MedQuAD, iCliniq, etc.), **fine-tune** a domain-specific LLM, and **deploy** on **AWS** for a **scalable, production-ready** solution.

> **Disclaimer**: This chatbot is for **educational and informational** purposes only, and **should not** replace professional medical advice.  

---

### ✨ Highlights

* **Accurate** – Llama‑3.1 8B GPTQ + FAISS + XGBoost LTR (↑ 30 % MRR vs baseline)
* **Cheap** – single g4dn.xlarge Spot instance, auto‑stop at 15 min CPU < 10 %
* **Scalable** – ALB + HTTPS + CloudFront cache for static SPA
* **Observable** – ALB logs → S3, container logs → CloudWatch, budget alarm $100

---

## 📌 Key Features

1. **Broad Medical Coverage**  
   - Answers questions about diseases, symptoms, treatments, and public health, using **high-quality data** (MedQuAD, iCliniq, Mayo Clinic, CDC).

2. **Retrieval-Augmented Generation (RAG)**  
   - Retrieves **top relevant passages** from an indexed knowledge base (FAISS/Pinecone) and **generates** contextual answers with a **PyTorch LLM**.

3. **Learning-to-Rank (LTR)**  
   - Improves **relevance** by re-ranking retrieved results using **XGBoost** or a **PyTorch-based** ranking model, adapting to user intent.

4. **Fine-Tuning & Experiment Tracking**  
   - Custom-train **Llama 2 / GPT-4** on medical Q&A data, with hyperparameter tracking via **Weights & Biases** (W&B).

5. **ETL & EDA Pipelines**  
   - **Airflow** DAGs to **extract**, **transform**, and **load** data from multiple sources, plus **pandas**-based EDA to visualize domain distribution, topics, and data quality.

6. **FastAPI Backend + react + Vite + Tailwind css Frontend**  
   - Real-time API with `/chat` endpoints for user queries, plus an intuitive **react** interface for multi-turn conversations.

7. **AWS Deployment**
   - Multi stage multi architecture Docker image for GPU inference. FastAPI RAG services run on **EC2** Spot instances behind an **ALB** with **HTTPS (ACM**) and **Route 53 DNS**, with automatic start/stop and **CloudWatch/S3** logging for observability.


## 📂 Project Structure

```bash
MediMaven/
├── Dockerfile                   # Multi‑stage GPU build for inference
├── docker-compose*.yml          # Local & prod compose configs
├── Docs/                        # In‑depth markdowns: ETL, infra, model, run‑book
│   ├── etl.md
│   ├── model.md
│   ├── infra-runbook.md
│   └── demo.gif
├── airflow_etl/                 # Airflow DAGs & plugins for data ingestion + EDA
├── data/                        # Raw & processed data used in ETL pipelines
│   ├── raw/
│   └── processed/
├── pipelines/                   # Python pipelines: ETL, embeddings, LTR, RAG inference
│   ├── etl_pipeline.py
│   ├── embedding_pipeline.py
│   ├── ltr_training_pipeline.py
│   └── rag_inference_pipeline.py
├── src/                         # Application source code
│   ├── backend/                 # FastAPI app, retrieval, ranking, schemas
│   └── frontend/                # React + Tailwind SPA (chat client)
├── requirements.txt             # Python dependencies
├── setup_*.sh                   # Download models, init scripts
└── tests/                       # Unit & smoke tests

```


> **TL;DR:** End‑to‑end Retrieval‑Augmented‑Generation system (LLM + FAISS + GPU) that answers medical questions with cited context — tuned, containerised, and **cost‑optimised to run on Spot GPU with automatic start/stop**.

---

## ✨ Demo

| Interface | URL | Notes     |
|-----------|-----|-------    |
| **Swagger UI** | `https://api.medimaven-ai.com/docs` | FastAPI backend |
| **Web Client** | `https://www.medimaven-ai.com` | React + Tailwind + Streamed tokens |
| **cURL** | `curl -X POST https://api.medimaven-ai.com/chat -d '{"query":"What causes migraine?"}' -H "Content-Type: application/json"` | JSON → JSON |
|**Own GPU**| `t https://raw.githubusercontent.com/dranreb1660/MediMaven-Ai/main/download_models.sh && t https://raw.githubusercontent.com/dranreb1660/MediMaven-Ai/main/docker-compose.prod.yml` <br>`chmod +x download_models.sh` <br> `docker compose -f docker-compose.prod.yml` <br> open `http://localhost:8000/docs` or on cloud--> `http://<your_ip>:8000/docs` | Requires GPU access and Nvidia drivers


![demo-gif](docs/demo.gif)

---

## 🏗 Architecture
![architecture-png](docs/arch.gif)



---
# 🧩 Tech stack
| Layer             | Technology                                                               | Reason                          |
| ----------------- | ------------------------------------------------------------------------ | ------------------------------- |
| **LLM**           | GPTQ Llama‑2 (4‑bit) via 🤗 TGI                                          | 2 × faster, fits 24 GB VRAM     |
| **Retrieval**     | **FAISS** flat IP + XGBoost LTR                                          | Low‑latency & higher relevance  |
| **Serving**       | Docker Compose on **nvidia‑cuda:12.4** runtime                           | One‑command local or cloud      |
| **Cloud**         | AWS Spot (g4dn.xlarge / a10g), **ALB**, **S3**, **Route 53**, **Lambda** | Cheapest always‑on illusion     |
| **Observability** | CloudWatch logs + S3 ALB logs                                            | Root‑cause & cost insight       |
| **CI**            | GitHub Actions → multi‑arch buildx                                       | ARM (M‑series) & x86 containers |

---

# 📝 Run‑book / Ops
 See docs/infra-runbook.md for:

- Start/stop Spot instance

 - Restoring EBS gp3 30 GB

- Rotating HF / W&B tokens

- Interpreting CloudWatch alarms

---


# 📜 License
Apache‑2.0 — free for personal or commercial use (citation appreciated).
---

# 🙋‍♂️ Author
### **Bernard Kyei-Mensah**
>**ML/AI Engineer** passionate about shipping LLMs that don’t break the bank.
>- Linkdin: @dranreb1660 
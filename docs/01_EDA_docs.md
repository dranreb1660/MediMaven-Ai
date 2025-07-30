# 📊 MediMaven v1.1 — Exploratory Data Analysis (EDA) & Data Engineering Blueprint

![CI](https://img.shields.io/badge/Built_with-Docker-blue) ![AWS](https://img.shields.io/badge/Cloud-AWS-%23FF9900) ![License](https://img.shields.io/badge/License-Apache%202.0-green)

> **TL;DR** v1.1 scales **MediMaven-QA** from a single curated corpus to a **multi-source, citation-preserving knowledge base**  
> (⚙︎ 150 k QA pairs / 28 M tokens) with automated Airflow ETL, strict PHI scrubbing, and end-to-end experiment tracking.  
> The result is recruiter-ready proof of modern ML-Ops, data governance, and domain-specific NLP excellence.

## Table of Contents

1.  [Dataset Link](#dataset-link)
2.  [Dataset Evolution](#dataset-evolution)
3.  [Data Inventory & Schema](#data-inventory--schema)
4.  [Source & Domain Distribution](#source--domain-distribution)
5.  [Text Statistics](#text-statistics)
6.  [Data Quality & Governance](#data-quality--governance)
7.  [ETL & Experiment Tracking Pipeline](#etl--experiment-tracking-pipeline)
8.  [Insights for Model & Product](#insights-for-model--product)
9.  [Next Iteration (v1.2 Roadmap)](#next-iteration-v12-roadmap)
10. [Quick Load Snippet](#quick-load-snippet)
11. [Citation](#citation)

## 🔗 Dataset Link

**Hugging Face Dataset**: [dranreb1660/medimaven-qa-data](https://huggingface.co/datasets/dranreb1660/medimaven-qa-data)

---

## 1️⃣ Dataset Evolution  

| Version | Sources | Rows | Tokens | Highlights |
|---------|---------|------|--------|------------|
| **v1.0** | MedQuad & iCliniq | ≈ 65 k | ~ 7 M | Curated Q&A only |
| **v1.1** | Mayo Clinic, NHS.uk, WebMD *(scraped)* + synthetic lay/symptom Q&A | **70 243 chunks** / **143 221 QA** | **~ 28 M** tokens (~ 390 t/chunk) | Adds symptom-narrative questions (51 %) & richer metadata |

*Key upgrade* — A **Scrapy + Splash** crawler feeds spaCy sentence-aware chunking. GPT-4o-mini then auto-generates two QA pairs per chunk for **<$0.05 / 1 k pairs**.

---

## 2️⃣ Data Inventory & Schema  

| Config  | Granularity | Rows | Primary Fields |
|---------|-------------|------|----------------|
| `kb_chunks` | 400-token context windows | 70 243 | `id`, `url`, `title`, `section`, `source`, `text`, `n_tokens` |
| `qa_wide`   | list-of-dict QA per chunk | 70 018 | `chunk_id`, `qa` (*list[dict]*) |
| `qa_long`   | exploded QA pairs | 143 221 | `chunk_id`, `question`, `answer` |

All configs ship under **CC-BY-4.0** and include machine-generated annotations.

---

## 3️⃣ Source & Domain Distribution  

* **4 host domains** (Mayo Clinic, NHS, WebMD, CDC blog) — balanced to avoid single-source bias.  
* Each domain is tagged in `source`, enabling per-domain evaluation and ablation studies.

---

## 4️⃣ Text Statistics  

| Metric | Value |
|--------|------:|
| **Total tokens** | ~ 28 M |
| **Avg. tokens / chunk** | 390 |
| **QA pairs / chunk** | 2.0 |
| **% symptom-narrative Qs** | 51 % |

Lay questions trend ~ 25 words; first-person symptom narratives ~ 40 words — offering coverage of both concise and conversational user intents.

---

## 5️⃣ Data Quality & Governance  

| Check | Action |
|-------|--------|
| **Duplicates** | SHA-256 hashing on `question` & `answer`. |
| **PHI removal** | Regex + spaCy NER scrub for names, locations, dates. |
| **Missing values** | Critical fields enforced by crawler (`url`, `title`). |
| **Licensing** | All content from public-domain or CC-compatible pages; dataset released under **CC-BY-4.0**. |

---

## 6️⃣ ETL & Experiment Tracking Pipeline  

```mermaid
flowchart TD
    C[Scrapy/Splash Crawl] --> D[Sentence-Aware Chunking<br>(spaCy)]
    D --> |metadata| S[SQLite / Parquet Staging]
    D --> |text|  L[GPT-4o-mini QA Generation]
    L --> M[Merge & Validate]
    M --> V[Weights & Biases Artifact Registry]
    V --> T[Training / Evaluation]
```
## 7️⃣ Insights for Model & Product
- High lexical overlap (Jaccard ≈ 0.41) between Q & A ⇒ reranking benefits from context-aware features (e.g., reciprocal-rank fusion).

- Longer answers for symptom narratives ⇒ prioritize memory-efficient context windows (4-bit GPTQ, sliding attention).

- Domain skew minimal ⇒ cross-domain generalization without aggressive re-weighting.

## 8️⃣ Next Iteration (v1.2 Roadmap)
- Active-learning loop — feed low-confidence user queries back into GPT-4o to create hard negatives.

- Clinical specialty tags — auto-classify chunks into ICD-10 top-level codes for smarter routing.

- Bias audit — evaluate answer tone across gender/age terms and mitigate detected skew.

## 9️⃣ Quick Load Snippet (python)
```python
from datasets import load_dataset

chunks  = load_dataset("bernard-kyei/medimaven-qa-data", "kb_chunks", split="train")
qa_long = load_dataset("bernard-kyei/medimaven-qa-data", "qa_long",  split="train")

```
----

# 📜 Citation

```bibtex

@misc{KyeiMensah2025MediMavenQA,
  author  = {Kyei-Mensah, Bernard},
  title   = {MediMaven-QA: A Citation-Preserving Medical Q\A Dataset with Symptom Narratives},
  year    = {2025},
  url     = {https://huggingface.co/datasets/dranreb1660/medimaven-qa-data},
  note    = {Version 1.0}
}
```
# ⚡️ MediMaven 5-Minute Quickstart

**Goal**: Spin up the complete MediMaven stack (API + in-memory vector DB + UI) on your laptop or a small GPU VM.

## Table of Contents

1.  [Stack Components](#stack-components)
2.  [Clone & Configure](#clone--configure)
3.  [Start the Stack](#start-the-stack)
4.  [Smoke Test](#smoke-test)
5.  [Shut Down](#shut-down)

---

## 🐳 Stack Components

| Stage | Container | Port | Notes |
| :--- | :--- | :--- | :--- |
| Qdrant (vector DB) | `qdrant/qdrant:latest` | **6333** | **In-memory** collection seeded at startup—no external volume |
| API (FastAPI + vLLM) | `medimaven-api:1.1` | **8000** | RAG orchestration & Auth0 |
| UI (React/Vite) | `medimaven-ui:1.1` | **5173** | Pastel chat interface |

---

## 📋 Clone & Configure

To get started, clone the repository and configure the environment variables:

```bash
git clone https://github.com/bernard-kyei/medimaven.git
cd medimaven
cp .env.example .env   # fill HF_TOKEN, OPENAI_API_KEY, AUTH0 secrets
```

## 🚀 Starting the Stack

Once you've configured the environment variables, you can start the application with a single command:

```bash
docker compose -f docker-compose.prod.yml up -d --pull
```

## 💨 Smoke Test

To ensure everything is working correctly, you can run a quick smoke test:

```bash
curl http://localhost:8000/healthz                   # → {"status":"ok"}

curl -X POST http://localhost:8000/chat \\
     -H "Authorization: Bearer <token>" \\
     -d '{"query": "What are early symptoms of Lyme disease?"}'
```

## 🛑 Shutting Down

To stop the application and remove the containers, run the following command:

```bash
docker compose down
```

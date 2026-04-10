# Adaptive LLM Orchestration System

## Quick Start

```bash
# 1. Copy environment file
cp .env.example .env
# Fill in your API keys

# 2. Start with Docker
docker-compose up --build

# 3. Open browser
open http://localhost:8000
```

## Manual Setup

```bash
pip install -r requirements.txt
redis-server &
uvicorn app.main:app --reload --port 8000
```

## Architecture
- `app/router.py` — Task classification & model routing
- `app/executor.py` — Unified model invocation
- `app/validator.py` — Output validation & confidence scoring
- `app/debugger.py` — Failure analysis & retry logic
- `app/orchestrator.py` — Pipeline coordination
- `app/logger.py` — Structured logging
- `app/rag.py` — FAISS-based context retrieval
- `app/main.py` — FastAPI entry point
- `frontend/index.html` — UI

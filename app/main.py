"""
main.py — FastAPI entry point.

Endpoints:
  POST /execute  — Run the full orchestration pipeline
  GET  /logs     — Retrieve recent structured logs
  POST /debug    — Manually trigger debug analysis on a failed output
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.debugger import debug_and_retry
from app.executor import execute
from app.logger import logger
from app.models import (
    DebugRequest,
    ModelProvider,
    OrchestratorResult,
    TaskRequest,
    TaskType,
)
from app.orchestrator import run_pipeline
from app.validator import validate_output

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    await logger.log("system_startup", message="Adaptive LLM Orchestrator starting")
    yield
    await logger.log("system_shutdown", message="Orchestrator shutting down")


app = FastAPI(
    title="Adaptive LLM Orchestration System",
    description="Self-healing multi-model pipeline with RAG-based debugging",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve frontend directory relative to this file so it works regardless
# of where uvicorn is launched from.
_FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

# Mount the entire frontend folder at root-level paths so that
# style.css  → http://localhost:8000/style.css
# app.js     → http://localhost:8000/app.js
app.mount("/frontend", StaticFiles(directory=_FRONTEND_DIR), name="frontend")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def serve_ui():
    return FileResponse(os.path.join(_FRONTEND_DIR, "index.html"))


@app.get("/style.css", include_in_schema=False)
async def serve_css():
    return FileResponse(os.path.join(_FRONTEND_DIR, "style.css"), media_type="text/css")


@app.get("/app.js", include_in_schema=False)
async def serve_js():
    return FileResponse(os.path.join(_FRONTEND_DIR, "app.js"), media_type="application/javascript")


@app.post("/execute", response_model=OrchestratorResult)
async def execute_task(request: TaskRequest) -> OrchestratorResult:
    """
    Run the full adaptive orchestration pipeline.

    - Routes task to optimal model
    - Validates output
    - Retries/escalates on failure
    - Logs all decisions
    """
    try:
        result = await run_pipeline(request)
        return result
    except Exception as exc:  # noqa: BLE001
        await logger.log("unhandled_error", error=str(exc), task_preview=request.task[:100])
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc


@app.get("/logs")
async def get_logs(limit: int = Query(default=50, ge=1, le=500)) -> list[dict[str, Any]]:
    """
    Return recent structured log entries from Redis.
    Falls back to empty list if Redis is unavailable.
    """
    return await logger.get_recent(limit=limit)


@app.post("/debug")
async def manual_debug(request: DebugRequest) -> dict[str, Any]:
    """
    Manually trigger the debugging pipeline on a known-bad output.
    """
    from app.router import _classify_task  # noqa: PLC0415

    task_type = _classify_task(request.task)
    validation = validate_output(request.task, request.failed_output, task_type)

    if validation.is_valid:
        validation.is_valid = False
        validation.issues.append("Manually triggered debug")

    provider = ModelProvider.GROQ
    if request.model_used:
        try:
            provider = ModelProvider(request.model_used)
        except ValueError:
            pass

    start = time.monotonic()
    recovered, attempts, success = await debug_and_retry(
        task=request.task,
        task_type=task_type,
        initial_provider=provider,
        validation_result=validation,
    )
    elapsed_ms = (time.monotonic() - start) * 1000

    return {
        "success": success,
        "recovered_output": recovered,
        "attempts": len(attempts),
        "strategies_used": [a.strategy for a in attempts],
        "final_validation": (
            attempts[-1].validation.model_dump() if attempts and attempts[-1].validation else None
        ),
        "latency_ms": round(elapsed_ms, 2),
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import redis.asyncio as aioredis

from app.config import config

# Configure standard Python logger
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_std_logger = logging.getLogger("orchestrator")

# Ensure log directory exists
os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)

# Redis key for recent logs
REDIS_LOG_KEY = "orchestrator:logs"
REDIS_LOG_MAX = 500  # Keep last 500 entries


class StructuredLogger:
    """
    Writes JSON-line logs to disk and pushes them to Redis for API retrieval.
    All I/O is non-blocking via asyncio file writes and aioredis.
    """

    def __init__(self) -> None:
        self._redis: Optional[aioredis.Redis] = None
        self._file_lock = asyncio.Lock()

    async def _get_redis(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = await aioredis.from_url(
                config.REDIS_URL, decode_responses=True
            )
        return self._redis

    def _build_entry(self, event: str, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **data,
        }

    async def log(self, event: str, **data: Any) -> None:
        entry = self._build_entry(event, data)
        serialized = json.dumps(entry)

        # 1. Write to disk
        async with self._file_lock:
            try:
                with open(config.LOG_FILE, "a") as f:
                    f.write(serialized + "\n")
            except OSError as exc:
                _std_logger.error("Failed to write log to disk: %s", exc)

        # 2. Push to Redis (non-fatal if Redis is unavailable)
        try:
            redis = await self._get_redis()
            pipe = redis.pipeline()
            pipe.lpush(REDIS_LOG_KEY, serialized)
            pipe.ltrim(REDIS_LOG_KEY, 0, REDIS_LOG_MAX - 1)
            await pipe.execute()
        except Exception as exc:  # noqa: BLE001
            _std_logger.warning("Redis log push failed: %s", exc)

        # 3. Also emit to standard logger
        _std_logger.info("[%s] %s", event, json.dumps(data))

    async def get_recent(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return the most recent log entries from Redis."""
        try:
            redis = await self._get_redis()
            raw = await redis.lrange(REDIS_LOG_KEY, 0, limit - 1)
            return [json.loads(r) for r in raw]
        except Exception as exc:  # noqa: BLE001
            _std_logger.warning("Could not fetch logs from Redis: %s", exc)
            return []

    async def log_request(self, request_id: str, task: str) -> None:
        await self.log(
            "request_received",
            request_id=request_id,
            task_preview=task[:120],
        )

    async def log_route(
        self,
        request_id: str,
        task_type: str,
        complexity: str,
        model_provider: str,
    ) -> None:
        await self.log(
            "route_decision",
            request_id=request_id,
            task_type=task_type,
            complexity=complexity,
            model_provider=model_provider,
        )

    async def log_execution(
        self,
        request_id: str,
        model_provider: str,
        model_name: str,
        latency_ms: float,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        await self.log(
            "execution",
            request_id=request_id,
            model_provider=model_provider,
            model_name=model_name,
            latency_ms=round(latency_ms, 2),
            success=success,
            error=error,
        )

    async def log_validation(
        self,
        request_id: str,
        is_valid: bool,
        confidence: float,
        issues: list[str],
    ) -> None:
        await self.log(
            "validation",
            request_id=request_id,
            is_valid=is_valid,
            confidence=round(confidence, 4),
            issues=issues,
        )

    async def log_debug(
        self,
        request_id: str,
        retry_count: int,
        strategy: str,
        success: bool,
    ) -> None:
        await self.log(
            "debug_attempt",
            request_id=request_id,
            retry_count=retry_count,
            strategy=strategy,
            success=success,
        )

    async def log_final(
        self,
        request_id: str,
        total_latency_ms: float,
        retries: int,
        success: bool,
    ) -> None:
        await self.log(
            "request_complete",
            request_id=request_id,
            total_latency_ms=round(total_latency_ms, 2),
            retries=retries,
            success=success,
        )


# Module-level singleton
logger = StructuredLogger()

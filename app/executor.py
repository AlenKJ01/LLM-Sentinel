"""
executor.py — Unified model invocation with token tracking, cost estimation,
              and structured error categorisation.

Enhancements:
  - Estimated cost per execution (USD, per 1M token pricing)
  - ErrorCategory attached to every failed ExecutionResult
  - No silent failures: all exceptions are categorised and surfaced
"""

import logging
import time
from typing import Optional

import httpx
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import config
from app.models import ErrorCategory, ExecutionResult, ModelProvider

logger = logging.getLogger("orchestrator")

_groq_client   = None
_gemini_client = None


def _get_groq():
    global _groq_client
    if _groq_client is None:
        from langchain_groq import ChatGroq
        _groq_client = ChatGroq(
            api_key=config.GROQ_API_KEY,
            model=config.GROQ_MODEL,
            temperature=0.2,
            max_retries=1,
        )
    return _groq_client


def _get_gemini():
    global _gemini_client
    if _gemini_client is None:
        from langchain_google_genai import ChatGoogleGenerativeAI
        _gemini_client = ChatGoogleGenerativeAI(
            google_api_key=config.GEMINI_API_KEY,
            model=config.GEMINI_MODEL,
            temperature=0.2,
        )
    return _gemini_client


# ── Cost table (USD per 1M tokens, input+output blended estimate) ───────────
_COST_PER_1M_TOKENS: dict[ModelProvider, float] = {
    ModelProvider.GROQ:           0.59,   # LLaMA 3.3 70B on Groq
    ModelProvider.GEMINI:         0.075,  # Gemini 2.0 Flash
    ModelProvider.OLLAMA_CODER:   0.0,    # local
    ModelProvider.OLLAMA_GENERAL: 0.0,
    ModelProvider.OLLAMA_LLAMA:   0.0,
}

MODEL_DISPLAY_NAMES = {
    ModelProvider.OLLAMA_CODER:   "Qwen2.5-Coder (Local)",
    ModelProvider.OLLAMA_GENERAL: "Mistral (Local)",
    ModelProvider.OLLAMA_LLAMA:   "LLaMA 3.2 (Local)",
    ModelProvider.GROQ:           "LLaMA 3.3 70B via Groq",
    ModelProvider.GEMINI:         "Gemini 2.0 Flash",
}

_SYSTEM_PROMPT = (
    "You are a highly capable AI assistant. "
    "Respond accurately and concisely. "
    "For code, include only the code block and a brief explanation."
)

# Strict instruction block prepended to EVERY task before sending to model.
# This is non-negotiable and cannot be overridden by user input.
_STRICT_INSTRUCTIONS = r"""
STRICT RULES (MANDATORY):

1. USE ONLY PYTHON STANDARD LIBRARY
- Allowed: urllib, json, time, re, logging, os, random, socket, etc.
- FORBIDDEN: requests, httpx, numpy, pandas, etc.

2. OUTPUT STRUCTURE MUST MATCH EXACTLY:
### Root Causes of Failure
### Step-by-Step Improvements
### Python Implementation
### Optional Enhancements

3. PRODUCTION-READY CODE REQUIREMENTS:
- Must include timeout handling (urllib.request.urlopen(timeout=...))
- Must include retry logic ONLY for:
  • network errors (URLError)
  • timeouts (socket.timeout)
- MUST NOT retry invalid data or parsing failures
- Must include exponential backoff WITH jitter:
  time.sleep(base * 2**n + random.uniform(0, 1))

4. VALIDATION REQUIREMENTS:
- Validate required fields: user_id, email
- Validate types AND non-empty values
- Validate email using regex
- Handle nested fields safely using .get()
- DO NOT fake validation (no json.dumps → json.loads trick)

5. CACHING REQUIREMENTS:
- Implement cache using dict + timestamp
- Enforce TTL (default 60s)
- If cache expired → DO NOT use it
- If API fails → use valid cache
- If no cache → return SAFE DEFAULT with consistent schema:
  {"user_id": "default_user_id", "email": "default@example.com"}

6. EDGE CASE HANDLING (MANDATORY):
- Empty response → detect BEFORE parsing
- Invalid JSON → catch JSONDecodeError
- Unexpected data types → validate and correct
- Missing nested fields → use safe defaults

7. MODULAR DESIGN (STRICT):
- fetch_api_data() → ONLY API call
- validate_data() → ONLY validation
- clean_data() → ONLY transformations
- get_data_with_fallback() → orchestration
- NO mixing responsibilities

8. ERROR HANDLING:
- Use specific exceptions ONLY:
  socket.timeout → TIMEOUT
  urllib.error.URLError → NETWORK_ERROR
  json.JSONDecodeError → PARSE_ERROR
- NO bare except
- Log structured messages

9. HARD PROHIBITIONS:
- DO NOT retry invalid data
- DO NOT skip TTL checks
- DO NOT return partial schema
- DO NOT omit required imports

10. IMPORT CORRECTNESS (MANDATORY):
- ALL used modules MUST be explicitly imported
- If using socket.timeout → MUST include "import socket"
- If using urllib.error → MUST import it explicitly
- Missing imports = INVALID RESPONSE

11. CACHE CORRECTNESS:
- Cache MUST persist across function calls
- DO NOT create cache inside function scope
- Cache must be defined at module level

12. DATA CLEANING REQUIREMENT:
- Invalid data MUST be corrected, not just rejected
- Always return valid structured output

13. FINAL CORRECTNESS RULES (MANDATORY):
- ALL used modules MUST be imported (e.g., socket, re, urllib.error)
- ALWAYS check for empty response BEFORE json.loads()
- Validation MUST correct invalid data (not just return None)
- clean_data() MUST perform actual transformation OR be removed
- NEVER retry when validation fails

ANY VIOLATION = INVALID RESPONSE
"""


def _categorise_error(exc: Exception) -> ErrorCategory:
    msg = str(exc).lower()
    if "timeout" in msg or "timed out" in msg:
        return ErrorCategory.TIMEOUT
    if "quota" in msg or "rate limit" in msg or "resource_exhausted" in msg or "429" in msg:
        return ErrorCategory.QUOTA_EXCEEDED
    if "not found" in msg or "404" in msg or "decommissioned" in msg:
        return ErrorCategory.MODEL_UNAVAILABLE
    if "modulenotfounderror" in msg or "importerror" in msg:
        return ErrorCategory.API_ERROR
    if "json" in msg or "parse" in msg:
        return ErrorCategory.PARSING_ERROR
    return ErrorCategory.API_ERROR


def _estimate_cost(provider: ModelProvider, tokens: int) -> float:
    rate = _COST_PER_1M_TOKENS.get(provider, 0.0)
    return round((tokens / 1_000_000) * rate, 8)


async def _invoke_ollama(model_name: str, task: str) -> tuple[str, int]:
    payload = {
        "model": model_name,
        "prompt": f"System: {_SYSTEM_PROMPT}{_STRICT_INSTRUCTIONS}\n\nUser: {task}",
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 2048},
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json=payload,
        )
        response.raise_for_status()
        data   = response.json()
        text   = data.get("response", "").strip()
        tokens = data.get("eval_count", len(text.split()))
        return text, tokens


async def _invoke_langchain(client, task: str) -> tuple[str, int]:
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT + _STRICT_INSTRUCTIONS),
        HumanMessage(content=task),
    ]
    result      = await client.ainvoke(messages)
    text        = result.content.strip()
    usage       = getattr(result, "usage_metadata", {})
    token_count = usage.get("total_tokens", len(text.split())) if usage else len(text.split())
    return text, token_count


async def execute(
    task: str,
    provider: ModelProvider,
    system_prompt_override: Optional[str] = None,
) -> ExecutionResult:
    # 🔥 Quality boost injection (improves output depth)
    quality_boost = """
    Ensure the solution is deeply reasoned and not generic.
    Avoid superficial explanations.
    Focus on real-world robustness, correctness, and edge cases.
    """

    effective_task = quality_boost + "\n\n" + task
    if system_prompt_override:
        effective_task = f"{quality_boost}\n\n[CONTEXT]\n{system_prompt_override}\n\n[TASK]\n{task}"

    start          = time.monotonic()
    error: Optional[str]          = None
    error_category: Optional[ErrorCategory] = None
    output         = ""
    tokens: Optional[int]         = None

    model_name_map = {
        ModelProvider.OLLAMA_CODER:   config.OLLAMA_CODER_MODEL,
        ModelProvider.OLLAMA_GENERAL: config.OLLAMA_GENERAL_MODEL,
        ModelProvider.OLLAMA_LLAMA:   config.OLLAMA_LLAMA_MODEL,
        ModelProvider.GROQ:           config.GROQ_MODEL,
        ModelProvider.GEMINI:         config.GEMINI_MODEL,
    }

    try:
        if provider == ModelProvider.OLLAMA_CODER:
            output, tokens = await _invoke_ollama(config.OLLAMA_CODER_MODEL, effective_task)
        elif provider == ModelProvider.OLLAMA_GENERAL:
            output, tokens = await _invoke_ollama(config.OLLAMA_GENERAL_MODEL, effective_task)
        elif provider == ModelProvider.OLLAMA_LLAMA:
            output, tokens = await _invoke_ollama(config.OLLAMA_LLAMA_MODEL, effective_task)
        elif provider == ModelProvider.GROQ:
            if not config.GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY is not configured")
            output, tokens = await _invoke_langchain(_get_groq(), effective_task)
        elif provider == ModelProvider.GEMINI:
            if not config.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY is not configured")
            output, tokens = await _invoke_langchain(_get_gemini(), effective_task)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    except httpx.ConnectError as exc:
        error          = f"Cannot reach {provider} — is Ollama running? ({exc})"
        error_category = ErrorCategory.MODEL_UNAVAILABLE
        logger.warning("[executor] ConnectError provider=%s: %s", provider.value, exc)

    except httpx.TimeoutException as exc:
        error          = f"Timeout calling {provider} ({exc})"
        error_category = ErrorCategory.TIMEOUT
        logger.warning("[executor] Timeout provider=%s: %s", provider.value, exc)

    except Exception as exc:
        error          = f"{type(exc).__name__}: {exc}"
        error_category = _categorise_error(exc)
        logger.warning("[executor] Error provider=%s category=%s: %s",
                       provider.value, error_category.value, exc)

    latency_ms     = (time.monotonic() - start) * 1000
    tokens         = tokens or 0
    estimated_cost = _estimate_cost(provider, tokens)

    return ExecutionResult(
        output=output,
        model_provider=provider,
        model_name=MODEL_DISPLAY_NAMES.get(provider, model_name_map[provider]),
        latency_ms=latency_ms,
        tokens_used=tokens,
        error=error,
        error_category=error_category,
        estimated_cost_usd=estimated_cost,
    )

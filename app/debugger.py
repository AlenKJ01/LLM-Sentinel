"""
debugger.py — Validation-feedback-aware retry with intelligent fallback.

Enhancements:
  - Validation feedback injected into each retry prompt (issue-specific)
  - Early stop when confidence stops improving (no-improvement threshold)
  - Retry history tracked with delta confidence per attempt
  - Intelligent fallback model selection based on failure type
  - All errors categorised and logged; no silent failures
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from app.config import config
from app.models import ErrorCategory, ExecutionResult, ModelProvider, TaskType, ValidationResult
from app.rag import FailureRecord, rag_store
from app.validator import validate_output

logger = logging.getLogger("orchestrator")

# ── Improvement threshold ───────────────────────────────────────────────────
# If confidence does not improve by at least this amount, abort early
_MIN_IMPROVEMENT = 0.05


@dataclass
class RetryAttempt:
    attempt_number: int
    strategy: str
    provider: ModelProvider
    prompt: str
    result: Optional[ExecutionResult] = None
    validation: Optional[ValidationResult] = None
    confidence_delta: float = 0.0   # NEW: improvement vs previous attempt
    failure_reason: Optional[str] = None  # NEW: categorised failure


# ── Fallback selection ──────────────────────────────────────────────────────

# Maps (task_type, error_category) → preferred fallback provider
_SMART_FALLBACK: dict[tuple, ModelProvider] = {
    (TaskType.CODE,      ErrorCategory.VALIDATION_ERROR):  ModelProvider.GEMINI,
    (TaskType.CODE,      ErrorCategory.PARSING_ERROR):     ModelProvider.GROQ,
    (TaskType.CODE,      ErrorCategory.TIMEOUT):           ModelProvider.GROQ,
    (TaskType.REASONING, ErrorCategory.VALIDATION_ERROR):  ModelProvider.GEMINI,
    (TaskType.REASONING, ErrorCategory.QUOTA_EXCEEDED):    ModelProvider.GROQ,
    (TaskType.GENERAL,   ErrorCategory.QUOTA_EXCEEDED):    ModelProvider.GROQ,
    (TaskType.MATH,      ErrorCategory.VALIDATION_ERROR):  ModelProvider.GEMINI,
}

_ESCALATION_CHAIN: list[ModelProvider] = [
    ModelProvider.GROQ,
    ModelProvider.OLLAMA_GENERAL,
    ModelProvider.GEMINI,
]


def _categorise_error(error: Optional[str]) -> ErrorCategory:
    """Classify an error string into a structured ErrorCategory."""
    if not error:
        return ErrorCategory.UNKNOWN
    lower = error.lower()
    if "timeout" in lower or "timed out" in lower:
        return ErrorCategory.TIMEOUT
    if "quota" in lower or "rate limit" in lower or "resource_exhausted" in lower or "429" in lower:
        return ErrorCategory.QUOTA_EXCEEDED
    if "not found" in lower or "404" in lower or "decommissioned" in lower:
        return ErrorCategory.MODEL_UNAVAILABLE
    if "modulenotfounderror" in lower or "importerror" in lower:
        return ErrorCategory.API_ERROR
    if "json" in lower or "parse" in lower or "decode" in lower:
        return ErrorCategory.PARSING_ERROR
    if "syntax" in lower or "validation" in lower:
        return ErrorCategory.VALIDATION_ERROR
    return ErrorCategory.API_ERROR


def _smart_fallback(
    task_type: TaskType,
    error_category: ErrorCategory,
    exclude: set[ModelProvider],
    attempt: int,
) -> ModelProvider:
    """
    Choose the best fallback based on task type and failure type.
    Avoids providers that have already failed.
    """
    preferred = _SMART_FALLBACK.get((task_type, error_category))
    if preferred and preferred not in exclude:
        return preferred

    # Fall back to escalation chain, skipping excluded providers
    for provider in _ESCALATION_CHAIN:
        if provider not in exclude:
            return provider

    # Last resort: return GROQ regardless
    return ModelProvider.GROQ


def _build_retry_prompt(
    original_task: str,
    attempt_number: int,
    issues: list[str],
    suggestions: list[str],
    checks: dict,
    rag_context: Optional[str],
    previous_confidence: float,
) -> str:
    """
    Aggressive, validation-feedback-aware retry prompt.
    Each attempt injects specific constraint violations as hard requirements.
    """
    issues_block = "\n".join(f"  [{i+1}] {issue}" for i, issue in enumerate(issues)) or "  [1] Unspecified quality failure"

    # Build targeted per-violation instructions from the checks dict
    hard_fixes: list[str] = []
    constraint_violations: list[str] = []

    # Pull constraint-level violations first (highest priority)
    constraints = checks.get("constraints", {})
    if constraints.get("stdlib_only") is False:
        constraint_violations.append(
            "► USE ONLY Python standard library. Remove ALL external imports "
            "(requests, numpy, pandas, etc.). This is an absolute prohibition."
        )
    if constraints.get("sections_complete") is False:
        constraint_violations.append(
            "► Include ALL four section headers using EXACT text (case-sensitive, with ###):\n"
            "  ### Root Causes of Failure\n"
            "  ### Step-by-Step Improvements\n"
            "  ### Python Implementation\n"
            "  ### Optional Enhancements\n"
            "  NO alternatives: not 'Root Causes', not 'Implementation', not 'Improvements'."
        )
    if constraints.get("retry_present") is False:
        constraint_violations.append(
            "► Add EXPLICIT retry logic: implement a loop with attempt counter, "
            "delay/backoff, and max_attempts. This must be visible in the code."
        )
    if constraints.get("fallback_present") is False:
        constraint_violations.append(
            "► Add EXPLICIT fallback logic: every failure path must have an "
            "alternative handler (except block, backup strategy, or default value)."
        )
    if constraints.get("modular") is False:
        constraint_violations.append(
            "► Refactor into MODULAR code: define at least 2 separate, focused functions. "
            "A single monolithic script is NOT acceptable."
        )
    if constraints.get("production_quality") is False:
        constraint_violations.append(
            "► This is a production-ready task. Add: try/except error handling, "
            "retry logic with backoff, and a fallback strategy. ALL are required."
        )
    if constraints.get("code_quality") is False:
        constraint_violations.append(
            "► Wrap ALL json.loads() calls and HTTP requests in try/except blocks. "
            "Unhandled exceptions from external I/O are forbidden."
        )

    # Pull code-level issues
    if checks.get("syntax", {}).get("issues"):
        hard_fixes.append("► Fix ALL Python syntax errors listed in the issues above.")
    if checks.get("depth", {}).get("issues"):
        hard_fixes.append(
            "► Remove ALL placeholders (# TODO, # implement this, pass). "
            "Every function must have a complete, working implementation."
        )
    cq = checks.get("code_output_quality", {})
    if cq.get("issues"):
        for cq_issue in cq["issues"][:4]:
            tag = cq_issue.split("[")[1].split("]")[0] if "[" in cq_issue else "code_quality"
            fix_map = {
                "fake_validation":      "► Replace json.dumps/loads self-validation with real field/type/format checks (user_id non-empty string, email regex match).",
                "fake_cache":           "► Replace hardcoded dict cache with: _cache = {}  and entries of {data, timestamp}. Check time.time() - entry['ts'] < TTL.",
                "missing_timeout":      "► Add timeout=10 to every urllib.request.urlopen() call.",
                "missing_jitter":       "► Add jitter: time.sleep(base_delay * 2**attempt + random.uniform(0, 1))",
                "broad_except":         "► Replace 'except Exception' with specific types: socket.timeout, urllib.error.URLError, json.JSONDecodeError, ValueError.",
                "missing_email_validation": "► Add email validation: re.fullmatch(r'^[\\w.-]+@[\\w.-]+\\.\\w+$', value)",
                "empty_response":       "► After reading response body, check: if not body.strip(): log error and use fallback.",
                "unsafe_dict_access":   "► Replace data['key']['nested'] with data.get('key', {}).get('nested', default).",
            }
            if tag in fix_map:
                hard_fixes.append(fix_map[tag])
    if checks.get("completeness", {}).get("issues"):
        hard_fixes.append("► Provide a SIGNIFICANTLY more detailed and complete response.")

    constraint_block = ("\nCONSTRAINT VIOLATIONS (EACH IS A HARD REJECTION REASON):\n" +
                        "\n".join(constraint_violations)) if constraint_violations else ""

    fixes_block = ("\nADDITIONAL CODE FIXES REQUIRED:\n" +
                   "\n".join(hard_fixes)) if hard_fixes else ""

    rag_block = (f"\n\nRELEVANT PAST FIXES (apply these patterns):\n{rag_context}") if rag_context else ""

    prompt = f"""╔══════════════════════════════════════════════════════════════╗
║           ⚠  CRITICAL FAILURE — RESPONSE REJECTED  ⚠         ║
╚══════════════════════════════════════════════════════════════╝

ATTEMPT {attempt_number} | Previous confidence score: {previous_confidence:.0%} (BELOW THRESHOLD)

The previous response has been REJECTED due to {len(issues)} violation(s):
{issues_block}
{constraint_block}
{fixes_block}
{rag_block}

══════════════════════════════════════════════════════════════
STRICT RULES FOR THIS RETRY (ALL are mandatory):
  1. Do NOT repeat any of the violations listed above
  2. Do NOT use any disallowed libraries or patterns
  3. Address EVERY violation explicitly — no exceptions
  4. Provide a COMPLETE and CORRECT solution
  5. If any violation remains, the response will be rejected AGAIN

OUTPUT FORMAT REQUIRED:
  ### Root Causes of Failure
  (if applicable — explain WHY the previous approach was wrong)

  ### Step-by-Step Improvements
  (numbered list of concrete fixes)

  ### Python Implementation
  (complete, runnable code — no placeholders, no TODOs)

  ### Optional Enhancements
  (edge cases, performance improvements)

══════════════════════════════════════════════════════════════
ORIGINAL TASK (respond to this in full compliance):
{original_task}
══════════════════════════════════════════════════════════════
"""
    return prompt


# ── Public API ───────────────────────────────────────────────────────────────

async def debug_and_retry(
    task: str,
    task_type: TaskType,
    initial_provider: ModelProvider,
    validation_result: ValidationResult,
    max_retries: Optional[int] = None,
) -> tuple[str, list[RetryAttempt], bool]:
    """
    Attempt recovery from a failed validation with intelligent retry strategy.

    Returns: (best_output, attempts, success)
    """
    from app.executor import execute  # local import to avoid circular

    max_retries      = max_retries or config.MAX_RETRIES
    attempts: list[RetryAttempt] = []
    best_output      = ""
    best_confidence  = validation_result.confidence
    tried_providers: set[ModelProvider] = {initial_provider}
    current_provider = initial_provider
    prev_confidence  = validation_result.confidence

    # RAG context retrieved once
    rag_query  = f"{task} {' '.join(validation_result.issues[:3])}"
    rag_context = rag_store.build_context_prompt(rag_query, k=5)

    # Start with the original validation's checks
    current_checks      = validation_result.checks
    current_issues      = validation_result.issues
    current_suggestions = validation_result.suggestions

    for attempt_num in range(1, max_retries + 1):

        # ── Determine strategy and provider ──────────────────────────────────
        if attempt_num == 1:
            strategy = "improved_prompt"
            provider = current_provider
        elif attempt_num == max_retries:
            strategy = "gemini_escalation"
            provider = ModelProvider.GEMINI
        else:
            strategy  = "smart_model_switch"
            last_error = attempts[-1].result.error if attempts and attempts[-1].result else None
            error_cat  = _categorise_error(last_error)
            provider   = _smart_fallback(task_type, error_cat, tried_providers, attempt_num)

        tried_providers.add(provider)

        prompt = _build_retry_prompt(
            original_task=task,
            attempt_number=attempt_num,
            issues=current_issues,
            suggestions=current_suggestions,
            checks=current_checks,
            rag_context=rag_context,
            previous_confidence=prev_confidence,
        )

        attempt = RetryAttempt(
            attempt_number=attempt_num,
            strategy=strategy,
            provider=provider,
            prompt=prompt,
        )

        # ── Execute ───────────────────────────────────────────────────────────
        execution = await execute(prompt, provider)
        attempt.result = execution

        if execution.error:
            error_cat = _categorise_error(execution.error)
            attempt.failure_reason = f"{error_cat.value}: {execution.error[:120]}"
            attempt.validation = ValidationResult(
                is_valid=False,
                confidence=0.0,
                issues=[f"[{error_cat.value}] {execution.error[:200]}"],
                suggestions=["Switch to a different provider"],
            )
            logger.warning(
                "[debugger] attempt=%d strategy=%s provider=%s error_category=%s error=%s",
                attempt_num, strategy, provider.value, error_cat.value, execution.error[:100],
            )
            attempts.append(attempt)
            continue

        # ── Validate ──────────────────────────────────────────────────────────
        new_validation       = validate_output(task, execution.output, task_type)
        attempt.validation   = new_validation
        delta                = new_validation.confidence - prev_confidence
        attempt.confidence_delta = round(delta, 4)

        logger.info(
            "[debugger] attempt=%d strategy=%s provider=%s confidence=%.3f delta=%.3f",
            attempt_num, strategy, provider.value, new_validation.confidence, delta,
        )

        if new_validation.confidence > best_confidence:
            best_output     = execution.output
            best_confidence = new_validation.confidence

        # Update context for next attempt
        current_issues      = new_validation.issues
        current_suggestions = new_validation.suggestions
        current_checks      = new_validation.checks
        prev_confidence     = new_validation.confidence

        attempts.append(attempt)

        # ── Early stop: success ───────────────────────────────────────────────
        if new_validation.is_valid:
            _store_recovery(task, task_type, validation_result, strategy, "fixed")
            return best_output, attempts, True

        # ── Early stop: no improvement ────────────────────────────────────────
        if attempt_num > 1 and delta < _MIN_IMPROVEMENT:
            logger.info(
                "[debugger] early stop — improvement %.4f below threshold %.4f at attempt %d",
                delta, _MIN_IMPROVEMENT, attempt_num,
            )
            break

    _store_recovery(task, task_type, validation_result, "all_strategies", "escalated")
    return best_output, attempts, False


def _store_recovery(
    task: str,
    task_type: TaskType,
    validation_result: ValidationResult,
    fix_applied: str,
    resolution: str,
) -> None:
    try:
        record = FailureRecord(
            task_preview=task[:200],
            task_type=task_type.value,
            error_summary="; ".join(validation_result.issues[:3]),
            failed_output="",
            fix_applied=fix_applied,
            resolution=resolution,
        )
        rag_store.add_record(record)
    except Exception as exc:
        logger.warning("[debugger] Failed to store RAG recovery record: %s", exc)

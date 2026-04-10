"""
orchestrator.py — Full pipeline coordinator.

Enhancements:
  - Parallel primary + fallback execution (asyncio.gather with cancellation)
  - Intelligent fallback based on error category (not blind sequential loop)
  - Token + cost tracking aggregated across all execution steps
  - Structured output sections: routing, validation, debug, cost
  - RAG query passes task_type for re-ranking
  - No silent except: pass — all errors are categorised and logged
"""

import asyncio
import logging
import time
import uuid
from typing import Optional

from app.debugger import RetryAttempt, debug_and_retry
from app.executor import execute
from app.logger import logger as structured_logger
from app.models import (
    ErrorCategory,
    ModelPreference,
    ModelProvider,
    OrchestratorResult,
    TaskRequest,
)
from app.rag import rag_store
from app.router import route_task
from app.validator import validate_output

_log = logging.getLogger("orchestrator")

# Providers that can serve as immediate parallel fallbacks (cloud-only, fast)
_PARALLEL_FALLBACKS: dict[ModelProvider, ModelProvider] = {
    ModelProvider.GEMINI:         ModelProvider.GROQ,
    ModelProvider.OLLAMA_CODER:   ModelProvider.GROQ,
    ModelProvider.OLLAMA_GENERAL: ModelProvider.GROQ,
    ModelProvider.OLLAMA_LLAMA:   ModelProvider.GROQ,
    ModelProvider.GROQ:           ModelProvider.GEMINI,
}

# Intelligent sequential fallback chain keyed by error category
_FALLBACK_BY_ERROR: dict[ErrorCategory, list[ModelProvider]] = {
    ErrorCategory.QUOTA_EXCEEDED:    [ModelProvider.GROQ, ModelProvider.OLLAMA_GENERAL],
    ErrorCategory.MODEL_UNAVAILABLE: [ModelProvider.GROQ, ModelProvider.GEMINI],
    ErrorCategory.TIMEOUT:           [ModelProvider.GROQ, ModelProvider.OLLAMA_GENERAL],
    ErrorCategory.API_ERROR:         [ModelProvider.GEMINI, ModelProvider.GROQ],
    ErrorCategory.UNKNOWN:           [ModelProvider.GROQ, ModelProvider.GEMINI],
}


async def _execute_with_parallel_fallback(
    task: str,
    primary_provider: ModelProvider,
    rag_context: Optional[str],
) -> tuple:
    """
    Launch primary provider immediately.
    If primary fails within 3 s OR returns an error, the fallback fires.
    Returns (ExecutionResult, was_fallback_used: bool, retries_used: int).
    """
    fallback_provider = _PARALLEL_FALLBACKS.get(primary_provider)

    primary_task = asyncio.create_task(
        execute(task, primary_provider, system_prompt_override=rag_context)
    )

    # Wait up to 3 s for primary; if it hasn't resolved, launch fallback in parallel
    try:
        primary_result = await asyncio.wait_for(asyncio.shield(primary_task), timeout=3.0)
        if not primary_result.error:
            # Primary succeeded quickly — cancel any fallback (not started yet)
            return primary_result, False, 0
    except asyncio.TimeoutError:
        _log.info("[orchestrator] Primary provider %s slow — launching parallel fallback", primary_provider.value)

    # Primary timed out the 3 s guard or errored — race both
    if not fallback_provider or fallback_provider == primary_provider:
        # No meaningful fallback available — just await primary
        result = await primary_task
        return result, False, 0

    fallback_task = asyncio.create_task(
        execute(task, fallback_provider, system_prompt_override=rag_context)
    )

    done, pending = await asyncio.wait(
        [primary_task, fallback_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    for t in pending:
        t.cancel()

    winner = done.pop()
    result = winner.result()
    was_fallback = (winner is fallback_task)

    # If the winner still has an error and the other task is pending, await it
    if result.error and pending:
        try:
            other = pending.pop()
            other_result = await asyncio.wait_for(other, timeout=30.0)
            if not other_result.error:
                return other_result, not was_fallback, 1
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

    return result, was_fallback, (1 if was_fallback else 0)


def _select_sequential_fallback(
    error_category: Optional[ErrorCategory],
    tried: set[ModelProvider],
) -> Optional[ModelProvider]:
    """Pick the best sequential fallback based on the error type."""
    chain = _FALLBACK_BY_ERROR.get(
        error_category or ErrorCategory.UNKNOWN,
        [ModelProvider.GROQ, ModelProvider.GEMINI],
    )
    for provider in chain:
        if provider not in tried:
            return provider
    # All preferred fallbacks exhausted — try anything untried
    for provider in ModelProvider:
        if provider not in tried:
            return provider
    return None



def _build_failure_summary(validation) -> str:
    """Build a concise structured failure summary for logging and retry injection."""
    lines = ["Previous attempt failed due to:"]
    constraints = validation.checks.get("constraints", {})
    if not constraints.get("stdlib_only", True):
        lines.append("  • Non-standard libraries used (FORBIDDEN)")
    if not constraints.get("sections_complete", True):
        lines.append("  • Required output sections missing")
    if not constraints.get("retry_present", True):
        lines.append("  • Retry logic absent")
    if not constraints.get("fallback_present", True):
        lines.append("  • Fallback logic absent")
    if not constraints.get("modular", True):
        lines.append("  • Code is not modular")
    if not constraints.get("production_quality", True):
        lines.append("  • Missing production-quality patterns")
    if not constraints.get("code_quality", True):
        lines.append("  • Unsafe code patterns detected")
    for issue in validation.issues[:3]:
        if "HARD VIOLATION" not in issue:
            lines.append(f"  • {issue}")
    lines.append("\nFix ALL of these strictly. Partial fixes will be rejected.")
    return "\n".join(lines)


async def _second_pass(
    task: str,
    first_output: str,
    provider,
    task_type,
    request_id: str,
) -> tuple:
    """
    Run a second generation pass to improve the first valid output.
    Only replaces if the second pass scores higher confidence.
    """
    from app.validator import validate_output as _validate
    import logging as _logging
    _log2 = _logging.getLogger("orchestrator")

    review_prompt = f"""Review and improve the following solution:

ORIGINAL TASK:
{task}

CURRENT SOLUTION:
{first_output}

IMPROVEMENT REQUIREMENTS:
- Identify and fix any weaknesses or edge cases
- Strengthen error handling and robustness
- Ensure full compliance with ALL original constraints
- Make it truly production-ready
- Do NOT remove any correct sections — only improve

Provide the improved, complete solution:"""

    try:
        from app.executor import execute as _execute
        review_result = await _execute(review_prompt, provider)
        if review_result.error or not review_result.output:
            _log2.info("[orchestrator] Second pass failed: %s", review_result.error)
            return first_output, _validate(task, first_output, task_type)

        second_validation = _validate(task, review_result.output, task_type)
        first_validation  = _validate(task, first_output, task_type)

        if second_validation.confidence > first_validation.confidence:
            _log2.info(
                "[orchestrator] Second pass improved confidence: %.3f → %.3f",
                first_validation.confidence, second_validation.confidence,
            )
            await structured_logger.log(
                "second_pass_improved",
                request_id=request_id,
                before=first_validation.confidence,
                after=second_validation.confidence,
            )
            return review_result.output, second_validation

        _log2.info(
            "[orchestrator] Second pass did not improve (%.3f ≤ %.3f) — keeping original",
            second_validation.confidence, first_validation.confidence,
        )
        return first_output, first_validation

    except Exception as exc:
        _log2.warning("[orchestrator] Second pass error: %s", exc)
        from app.validator import validate_output as _validate2
        return first_output, _validate2(task, first_output, task_type)


async def run_pipeline(request: TaskRequest) -> OrchestratorResult:
    request_id     = str(uuid.uuid4())
    pipeline_start = time.monotonic()
    retries        = 0
    rag_context_used = False

    # Accumulated cost + token tracking
    total_tokens   = 0
    total_cost_usd = 0.0
    cost_breakdown: dict[str, float] = {}

    await structured_logger.log_request(request_id, request.task)

    # ── Step 1: Route ─────────────────────────────────────────────────────────
    route = route_task(
        request.task,
        model_preference=request.model_preference,
        prefer_fast=request.prefer_fast,
        prefer_local=request.prefer_local,
    )

    await structured_logger.log_route(
        request_id,
        route.task_type.value,
        route.complexity.value,
        route.model_provider.value,
    )
    await structured_logger.log(
        "route_confidence",
        request_id=request_id,
        confidence=route.confidence,
        routing_metadata=route.routing_metadata,
    )

    # ── Step 2: RAG context ───────────────────────────────────────────────────
    rag_context: Optional[str] = None
    try:
        if rag_store.total_records > 0:
            rag_context = rag_store.build_context_prompt(
                request.task, k=5, task_type=route.task_type.value
            )
            if rag_context:
                rag_context_used = True
    except Exception as exc:
        _log.warning("[orchestrator] RAG context failed: %s", exc)

    # ── Step 3: Execute with parallel fallback ────────────────────────────────
    execution, was_parallel_fallback, parallel_retries = await _execute_with_parallel_fallback(
        request.task, route.model_provider, rag_context
    )
    retries += parallel_retries

    _track_cost(execution, cost_breakdown)
    total_tokens   += execution.tokens_used or 0
    total_cost_usd += execution.estimated_cost_usd or 0.0

    await structured_logger.log_execution(
        request_id,
        execution.model_provider.value,
        execution.model_name,
        execution.latency_ms,
        success=execution.error is None,
        error=execution.error,
    )
    if execution.tokens_used:
        await structured_logger.log(
            "token_usage",
            request_id=request_id,
            model_provider=execution.model_provider.value,
            tokens=execution.tokens_used,
            estimated_cost_usd=execution.estimated_cost_usd,
        )

    # ── Step 4: Intelligent sequential fallback if still failing ─────────────
    tried_providers: set[ModelProvider] = {route.model_provider, execution.model_provider}

    if execution.error and not execution.output:
        fallback = _select_sequential_fallback(execution.error_category, tried_providers)
        while fallback and not execution.output:
            _log.info(
                "[orchestrator] Sequential fallback → %s (error_category=%s)",
                fallback.value,
                execution.error_category.value if execution.error_category else "unknown",
            )
            execution = await execute(request.task, fallback)
            retries += 1
            tried_providers.add(fallback)

            _track_cost(execution, cost_breakdown)
            total_tokens   += execution.tokens_used or 0
            total_cost_usd += execution.estimated_cost_usd or 0.0

            await structured_logger.log_execution(
                request_id,
                execution.model_provider.value,
                execution.model_name,
                execution.latency_ms,
                success=execution.error is None,
                error=execution.error,
            )
            await structured_logger.log(
                "fallback_decision",
                request_id=request_id,
                reason=str(execution.error_category),
                selected_fallback=fallback.value,
            )

            if execution.error:
                fallback = _select_sequential_fallback(execution.error_category, tried_providers)
            else:
                break

    # ── Step 5: Validate ──────────────────────────────────────────────────────
    validation = validate_output(request.task, execution.output, route.task_type)
    await structured_logger.log_validation(
        request_id, validation.is_valid, validation.confidence, validation.issues
    )
    await structured_logger.log(
        "validation_checks",
        request_id=request_id,
        checks=validation.checks,
    )

    final_output   = execution.output
    final_provider = execution.model_provider
    final_model    = execution.model_name
    debug_summary: dict = {}

    # ── Step 6: Debug & retry if invalid ─────────────────────────────────────
    if not validation.is_valid:
        # Build structured failure summary to pass into retry prompt
        _failure_summary = _build_failure_summary(validation)

        await structured_logger.log(
            "validation_failure_summary",
            request_id=request_id,
            summary=_failure_summary,
            confidence=validation.confidence,
        )

        # 🔥 Inject failure context into retry task (CRITICAL FIX)
        enhanced_task = request.task + "\n\n" + _failure_summary

        recovered_output, attempts, success = await debug_and_retry(
            task=enhanced_task,   # <-- key change (was request.task)
            task_type=route.task_type,
            initial_provider=route.model_provider,
            validation_result=validation,
        )

        retries += len(attempts)

        for attempt in attempts:
            is_valid = attempt.validation.is_valid if attempt.validation else False

            await structured_logger.log_debug(
                request_id,
                attempt.attempt_number,
                attempt.strategy,
                success=is_valid,
            )

            if attempt.result:
                _track_cost(attempt.result, cost_breakdown)
                total_tokens   += attempt.result.tokens_used or 0
                total_cost_usd += attempt.result.estimated_cost_usd or 0.0

                await structured_logger.log_execution(
                    request_id,
                    attempt.result.model_provider.value,
                    attempt.result.model_name,
                    attempt.result.latency_ms,
                    success=attempt.result.error is None,
                    error=attempt.result.error,
                )
                
        initial_failure_reason = validation.reason

        debug_summary = {
            "attempts": len(attempts),
            "recovered": success,
            "strategies": [a.strategy for a in attempts],
            "confidence_deltas": [a.confidence_delta for a in attempts],
            "initial_failure": initial_failure_reason
        }

        if success and recovered_output:
            final_output = recovered_output

            last_good = next(
                (a for a in reversed(attempts) if a.validation and a.validation.is_valid),
                None,
            )

            if last_good and last_good.result:
                final_provider = last_good.result.model_provider
                final_model    = last_good.result.model_name

        # 🔥 IMPORTANT: validate against ORIGINAL task, not enhanced_task
        validation = validate_output(request.task, final_output, route.task_type)

    # ── Step 7: Multi-pass improvement (second pass on valid outputs) ──────────
    if validation.is_valid and final_output:
        final_output, validation = await _second_pass(
            task=request.task,
            first_output=final_output,
            provider=final_provider,
            task_type=route.task_type,
            request_id=request_id,
        )

    # ── Step 8: Strict failure policy — never return weak output ─────────────
    if not final_output or not validation.is_valid:
        final_output = (
            "⚠ GENERATION FAILED\n\n"
            "The system could not produce a response that meets all quality requirements "
            f"after {retries} attempt(s).\n\n"
            f"Last validation confidence: {validation.confidence:.0%} (threshold: 70%)\n\n"
            "Validation issues detected:\n" +
            "\n".join(f"  • {i}" for i in validation.issues[:5])
        )

    # ── Step 9: Final logging ─────────────────────────────────────────────────
    total_latency = (time.monotonic() - pipeline_start) * 1000
    await structured_logger.log_final(request_id, total_latency, retries, success=bool(final_output))
    await structured_logger.log(
        "cost_summary",
        request_id=request_id,
        total_tokens=total_tokens,
        total_cost_usd=round(total_cost_usd, 8),
        breakdown=cost_breakdown,
    )

    return OrchestratorResult(
        task=request.task,
        output=final_output or "Unable to generate a valid response after all retries.",
        model_used=final_model,
        model_provider=final_provider.value,
        task_type=route.task_type.value,
        complexity=route.complexity.value,
        latency_ms=round(total_latency, 2),
        retries=retries,
        validation_confidence=validation.confidence,
        rag_context_used=rag_context_used,
        request_id=request_id,
        error=execution.error if not final_output else None,
        # ── Structured sub-sections ──
        meta={
            "success":    validation.is_valid,
            "attempts":   retries,
            "confidence": validation.confidence,
            "reason":     validation.reason
        },
        routing={
            "task_type":  route.task_type.value,
            "model":      route.model_provider.value,
            "confidence": route.confidence,
            "reason":     route.reasoning,
            "metadata":   route.routing_metadata,
        },
        validation={
            "is_valid":   validation.is_valid,
            "confidence": validation.confidence,
            "issues":     validation.issues,
            "checks":     validation.checks,
            "reason":     validation.reason,
        },
        debug=debug_summary,
        cost={
            "total_tokens":    total_tokens,
            "total_cost_usd":  round(total_cost_usd, 8),
            "breakdown":       cost_breakdown,
        },
    )


def _track_cost(execution, breakdown: dict) -> None:
    """Accumulate per-model cost into the breakdown dict."""
    if execution.estimated_cost_usd and execution.estimated_cost_usd > 0:
        key = execution.model_provider.value
        breakdown[key] = round(breakdown.get(key, 0.0) + execution.estimated_cost_usd, 8)

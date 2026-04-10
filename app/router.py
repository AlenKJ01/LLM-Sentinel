"""
router.py — Classifies tasks and routes them to the best model.

Enhancements:
  - Confidence score (0–1) on every routing decision
  - Structured routing_metadata with keyword hit counts
  - General-task penalty prevents misclassification of factual queries
  - Explicit reason string per route
"""

import re
from typing import Optional

from langchain import embeddings

from app.config import config
from app.models import (
    ComplexityLevel,
    ModelPreference,
    ModelProvider,
    RouteDecision,
    TaskType,
)

from sentence_transformers import SentenceTransformer
import numpy as np

_model = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

# ── Keyword sets ────────────────────────────────────────────────────────────

_CODE_KEYWORDS = frozenset([
    "code", "function", "class", "implement", "debug", "fix", "refactor",
    "sql", "query", "algorithm", "script", "program", "def ", "import ",
    "syntax", "compile", "runtime", "exception", "stacktrace", "unittest",
    "pytest", "api", "endpoint", "schema", "json", "yaml", "dockerfile",
    "bash", "shell", "regex", "parse", "coding", "programming",
])

_REASONING_KEYWORDS = frozenset([
    "explain", "why", "how does", "compare", "analyze", "evaluate",
    "pros and cons", "tradeoffs", "design", "architecture", "strategy",
    "plan", "recommend", "should i", "best practice", "difference between",
    "what is the impact", "summarize", "critique", "assess",
])

_MATH_KEYWORDS = frozenset([
    "calculate", "solve", "equation", "integral", "derivative", "matrix",
    "probability", "statistics", "proof", "theorem", "formula", "compute",
    "math", "arithmetic",
])

# Signals that indicate a general knowledge / factual query
_GENERAL_SIGNALS = frozenset([
    "capital", "country", "city", "what is", "tell me about",
    "information about", "history of", "who is", "where is",
    "report", "essay", "describe", "overview", "summary of",
    "facts about", "write about", "explain what",
])

TASK_PROTOTYPE_TEXTS = {
    TaskType.CODE: "write python code function class implementation debugging",
    TaskType.REASONING: "analyze explain reasoning system design compare evaluate",
    TaskType.MATH: "solve equation math calculation probability statistics",
    TaskType.GENERAL: "general question explanation facts information overview"
}

_PROTOTYPE_EMBEDDINGS = None

def get_prototype_embeddings():
    global _PROTOTYPE_EMBEDDINGS

    if _PROTOTYPE_EMBEDDINGS is None:
        model = get_model()
        _PROTOTYPE_EMBEDDINGS = {
            k: model.encode(v, normalize_embeddings=True)
            for k, v in TASK_PROTOTYPE_TEXTS.items()
        }

    return _PROTOTYPE_EMBEDDINGS

def _semantic_classify(task: str) -> tuple[TaskType, float]:
    """
    Semantic classification using embeddings
    """
    query_emb = get_model.encode(task, normalize_embeddings=True)

    scores = {}
    embeddings = get_prototype_embeddings()

    for task_type, proto_emb in embeddings.items():
        score = float(np.dot(query_emb, proto_emb))
        scores[task_type] = score

    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    return best_type, best_score 

def _classify_task(task: str) -> tuple[TaskType, dict[str, float]]:
    """
    Hybrid classification:
    - Keyword-based (fast, deterministic)
    - Embedding-based (semantic understanding)

    Returns (TaskType, hit_counts_dict)
    """

    lower = task.lower()

    # ── Keyword scoring ─────────────────────────────────────────
    code_hits   = sum(1 for kw in _CODE_KEYWORDS if kw in lower)
    reason_hits = sum(1 for kw in _REASONING_KEYWORDS if kw in lower)
    math_hits   = sum(1 for kw in _MATH_KEYWORDS if kw in lower)
    general_hits = sum(1 for kw in _GENERAL_SIGNALS if kw in lower)

    # Code block → strong signal
    if re.search(r"```|\n {4}", task):
        code_hits += 3

    # Penalize code if general dominates
    if general_hits >= 2:
        code_hits = max(0, code_hits - 3)

    hit_counts = {
        "code": code_hits,
        "reasoning": reason_hits,
        "math": math_hits,
        "general": general_hits,
    }

    # ── Keyword decision ───────────────────────────────────────
    scores = {
        TaskType.CODE:      code_hits,
        TaskType.REASONING: reason_hits,
        TaskType.MATH:      math_hits,
    }

    keyword_winner, top_score = max(scores.items(), key=lambda x: x[1])
    keyword_type = keyword_winner if top_score > 0 else TaskType.GENERAL

    # ── Semantic classification (NEW) ──────────────────────────
    try:
        query_emb = get_model.encode(task, normalize_embeddings=True)

        semantic_type, semantic_scores = _semantic_classify(task)
        embeddings = get_prototype_embeddings()
        for task_type, proto_emb in embeddings.items():
            score = float(np.dot(query_emb, proto_emb))
            semantic_scores[task_type] = score

        semantic_type = max(semantic_scores, key=semantic_scores.get)
        semantic_score = semantic_scores[semantic_type]

    except Exception:
        # fallback if embedding fails
        semantic_type = keyword_type
        semantic_score = 0.0

    # ── Hybrid decision logic ──────────────────────────────────
    if semantic_score > 0.55:
        final_type = semantic_type
    elif top_score >= 2:
        final_type = keyword_type
    else:
        final_type = semantic_type  # fallback to semantic if weak keywords

    # ── Add metadata for debugging/metrics ─────────────────────
    hit_counts.update({
        "semantic_score": float(round(semantic_score, 3)),
        "semantic_type": semantic_type.value,
        "keyword_type": keyword_type.value,
    })

    return final_type, hit_counts

def _estimate_complexity(task: str, task_type: TaskType) -> tuple[ComplexityLevel, float]:
    word_count   = len(task.split())
    lower        = task.lower()
    length_score = min(word_count / 500, 1.0)

    multi_step_hits = sum(
        1 for phrase in [
            "step by step", "multiple", "complex", "large scale",
            "production", "distributed", "secure", "optimized",
            "end-to-end", "full implementation", "complete system",
        ]
        if phrase in lower
    )
    multi_score = min(multi_step_hits / 5, 1.0)

    type_weights = {
        TaskType.CODE:      0.6,
        TaskType.REASONING: 0.5,
        TaskType.MATH:      0.55,
        TaskType.GENERAL:   0.3,
        TaskType.UNKNOWN:   0.3,
    }
    raw = 0.4 * length_score + 0.35 * multi_score + 0.25 * type_weights[task_type]

    if raw >= config.COMPLEXITY_HIGH_THRESHOLD:
        return ComplexityLevel.HIGH, raw
    if raw >= config.COMPLEXITY_LOW_THRESHOLD:
        return ComplexityLevel.MEDIUM, raw
    return ComplexityLevel.LOW, raw


def _compute_routing_confidence(
    task_type: TaskType,
    hit_counts: dict[str, float],
    complexity: ComplexityLevel,
    is_user_override: bool,
) -> float:
    """
    Confidence score rules:
      - User explicit override → 1.0 (deterministic)
      - Strong keyword dominance (winning type ≥ 3 hits, others ≤ 1) → 0.85–0.95
      - Moderate dominance → 0.65–0.80
      - Weak signal (general fallback or tie) → 0.45–0.60
    """
    if is_user_override:
        return 1.0

    winning_hits = hit_counts.get(task_type.value, 0)
    other_hits = sum(
        v for k, v in hit_counts.items()
        if isinstance(v, (int, float)) and k not in [task_type.value, "general"]
    )
    margin = winning_hits - other_hits

    if task_type == TaskType.GENERAL:
        # General is a fallback — inherently lower confidence
        return round(min(0.55 + hit_counts.get("general", 0) * 0.03, 0.65), 3)

    if winning_hits >= 4 and margin >= 3:
        base = 0.90
    elif winning_hits >= 2 and margin >= 1:
        base = 0.75
    elif winning_hits >= 1:
        base = 0.60
    else:
        base = 0.45

    # Complexity adds certainty (high complexity = more distinctive task)
    complexity_bonus = {"low": 0.0, "medium": 0.02, "high": 0.04}[complexity.value]

    semantic_score = float(hit_counts.get("semantic_score", 0))
    semantic_bonus = 0.05 if semantic_score > 0.6 else 0.0

    final_score = base + complexity_bonus + semantic_bonus

    return round(min(final_score, 0.97), 3)


def _auto_select_model(
    task_type: TaskType,
    complexity: ComplexityLevel,
) -> tuple[ModelProvider, str]:
    """
    Routing matrix:
      CODE:      LOW/MED → Qwen2.5-Coder | HIGH → Gemini
      MATH/REASONING: HIGH → Gemini | LOW/MED → Groq
      GENERAL:   HIGH → Gemini | MED → Mistral | LOW → Groq
    """
    if task_type == TaskType.CODE:
        if complexity == ComplexityLevel.HIGH:
            return ModelProvider.GEMINI, "High-complexity code requires deep reasoning — routed to Gemini"
        return ModelProvider.OLLAMA_CODER, "Code task — routed to Qwen2.5-Coder (specialist model)"

    if task_type in (TaskType.MATH, TaskType.REASONING):
        if complexity == ComplexityLevel.HIGH:
            return ModelProvider.GEMINI, "Complex reasoning/math — routed to Gemini"
        return ModelProvider.GROQ, "Reasoning/math task — routed to Groq (LLaMA 3.3 70B, fast + capable)"

    # GENERAL
    if complexity == ComplexityLevel.HIGH:
        return ModelProvider.GEMINI, "Complex general task — routed to Gemini"
    if complexity == ComplexityLevel.MEDIUM:
        return ModelProvider.OLLAMA_GENERAL, "General task — routed to Mistral (local, balanced)"
    return ModelProvider.GROQ, "Simple general task — routed to Groq (fastest path)"


_PREFERENCE_MAP: dict[ModelPreference, tuple[ModelProvider, str]] = {
    ModelPreference.GROQ:    (ModelProvider.GROQ,           "User override: Groq — LLaMA 3.3 70B (Fast responses)"),
    ModelPreference.GEMINI:  (ModelProvider.GEMINI,         "User override: Gemini 2.0 Flash (Complex reasoning)"),
    ModelPreference.CODER:   (ModelProvider.OLLAMA_CODER,   "User override: Qwen2.5-Coder (Code & debugging)"),
    ModelPreference.GENERAL: (ModelProvider.OLLAMA_GENERAL, "User override: Mistral (General tasks)"),
    ModelPreference.LLAMA:   (ModelProvider.OLLAMA_LLAMA,   "User override: LLaMA 3.2 (General/code fallback)"),
}


def route_task(
    task: str,
    model_preference: ModelPreference = ModelPreference.AUTO,
    prefer_fast: bool = False,
    prefer_local: bool = False,
) -> RouteDecision:
    """
    Classify, score, and route a task.
    Returns RouteDecision with confidence score and structured metadata.
    """
    task_type, hit_counts = _classify_task(task)
    complexity, complexity_raw = _estimate_complexity(task, task_type)

    is_user_override = model_preference != ModelPreference.AUTO or prefer_fast or prefer_local

    # 1. Explicit model preference from UI
    if model_preference != ModelPreference.AUTO:
        provider, reason = _PREFERENCE_MAP[model_preference]
        return RouteDecision(
            task_type=task_type,
            complexity=complexity,
            model_provider=provider,
            reasoning=reason,
            confidence=1.0,
            routing_metadata={
                "source": "user_override",
                "preference": model_preference.value,
                "keyword_hits": hit_counts,
                "complexity_score": round(complexity_raw, 3),
            },
        )

    # 2. Legacy fast/local flags
    if prefer_fast:
        return RouteDecision(
            task_type=task_type,
            complexity=complexity,
            model_provider=ModelProvider.GROQ,
            reasoning="prefer_fast flag set — routed to Groq for minimum latency",
            confidence=1.0,
            routing_metadata={"source": "prefer_fast_flag", "keyword_hits": hit_counts},
        )
    if prefer_local:
        provider = ModelProvider.OLLAMA_CODER if task_type == TaskType.CODE else ModelProvider.OLLAMA_GENERAL
        return RouteDecision(
            task_type=task_type,
            complexity=complexity,
            model_provider=provider,
            reasoning="prefer_local flag set — routed to Ollama for local execution",
            confidence=1.0,
            routing_metadata={"source": "prefer_local_flag", "keyword_hits": hit_counts},
        )

    # 3. Automatic routing
    provider, reason = _auto_select_model(task_type, complexity)
    confidence = _compute_routing_confidence(task_type, hit_counts, complexity, is_user_override=False)

    return RouteDecision(
        task_type=task_type,
        complexity=complexity,
        model_provider=provider,
        reasoning=reason,
        confidence=confidence,
        routing_metadata={
            "source": "auto",
            "keyword_hits": hit_counts,
            "complexity_score": round(complexity_raw, 3),
            "word_count": len(task.split()),
        },
    )

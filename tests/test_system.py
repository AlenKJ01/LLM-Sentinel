"""
tests/test_system.py — Integration + unit tests for the orchestrator.

Run with:  pytest tests/ -v
"""

import asyncio
import json
import pytest

from app.models import (
    ComplexityLevel,
    ModelProvider,
    TaskRequest,
    TaskType,
)
from app.router import route_task, _classify_task, _estimate_complexity
from app.validator import validate_output


# ─────────────────────────────────────────────────────────────────────────────
# Router tests
# ─────────────────────────────────────────────────────────────────────────────

class TestClassification:
    def test_code_task_detected(self):
        task = "Write a Python function to implement quicksort"
        assert _classify_task(task) == TaskType.CODE

    def test_reasoning_task_detected(self):
        task = "Explain the pros and cons of microservices vs monolith"
        assert _classify_task(task) == TaskType.REASONING

    def test_math_task_detected(self):
        task = "Calculate the integral of x^2 from 0 to 5"
        assert _classify_task(task) == TaskType.MATH

    def test_general_fallback(self):
        task = "Tell me about the weather"
        assert _classify_task(task) == TaskType.GENERAL

    def test_code_block_boosts_code_score(self):
        task = "What does this do?\n```\nfor i in range(10): print(i)\n```"
        assert _classify_task(task) == TaskType.CODE


class TestComplexity:
    def test_short_task_is_low_complexity(self):
        task = "Hello world in Python"
        level, score = _estimate_complexity(task, TaskType.CODE)
        assert level == ComplexityLevel.LOW

    def test_long_complex_task_is_high(self):
        task = (
            "Design and implement a production-ready distributed rate limiter "
            "using Redis with sliding window algorithm, token bucket fallback, "
            "full test coverage, and step-by-step documentation. "
            "The system must handle multiple tenants with different rate limits "
            "and support hot-reload of configuration." * 2
        )
        level, score = _estimate_complexity(task, TaskType.CODE)
        assert level == ComplexityLevel.HIGH


class TestRouteDecision:
    def test_code_routes_to_ollama_coder_by_default(self):
        decision = route_task("Write a Python function to binary search a list")
        assert decision.model_provider == ModelProvider.OLLAMA_CODER
        assert decision.task_type == TaskType.CODE

    def test_prefer_fast_routes_to_groq(self):
        decision = route_task("Summarise this article", prefer_fast=True)
        assert decision.model_provider == ModelProvider.GROQ

    def test_prefer_local_routes_to_ollama(self):
        decision = route_task("What is 2+2?", prefer_local=True)
        assert decision.model_provider in (ModelProvider.OLLAMA_CODER, ModelProvider.OLLAMA_GENERAL)

    def test_high_complexity_reasoning_routes_to_gemini(self):
        task = (
            "Analyze and compare the architectural tradeoffs of event-driven "
            "microservices versus request-response monoliths for a complex "
            "financial system with strict consistency requirements. "
            "Provide recommendations based on best practices." * 3
        )
        decision = route_task(task)
        assert decision.model_provider == ModelProvider.GEMINI


# ─────────────────────────────────────────────────────────────────────────────
# Validator tests
# ─────────────────────────────────────────────────────────────────────────────

class TestValidator:
    def test_empty_output_fails(self):
        result = validate_output("Write code", "")
        assert not result.is_valid
        assert result.confidence == 0.0

    def test_short_output_fails(self):
        result = validate_output("Explain machine learning in detail", "ML is good")
        assert not result.is_valid or result.confidence < 0.6

    def test_valid_general_response(self):
        result = validate_output(
            "What is machine learning?",
            "Machine learning is a subset of artificial intelligence that enables "
            "systems to learn from data and improve their performance over time "
            "without being explicitly programmed. It uses statistical techniques "
            "to build mathematical models and make predictions.",
        )
        assert result.is_valid
        assert result.confidence >= 0.5

    def test_code_with_syntax_error_flagged(self):
        result = validate_output(
            "Write a Python function",
            "```python\ndef foo(:\n    pass\n```",
            task_type=TaskType.CODE,
        )
        assert not result.is_valid
        assert any("Syntax error" in i for i in result.issues)

    def test_valid_python_code_passes(self):
        result = validate_output(
            "Write a Python function to add two numbers",
            "```python\ndef add(a: int, b: int) -> int:\n    \"\"\"Return sum of a and b.\"\"\"\n    return a + b\n```\n\nThis function takes two integers and returns their sum.",
            task_type=TaskType.CODE,
        )
        assert result.is_valid
        assert result.confidence > 0.6

    def test_refusal_detected(self):
        result = validate_output(
            "Explain recursion",
            "I cannot provide information as an AI language model.",
        )
        assert not result.is_valid

    def test_truncation_detected(self):
        result = validate_output(
            "Explain something",
            "The answer is complex and involves many steps etc.",
        )
        assert any("truncated" in i.lower() for i in result.issues)


# ─────────────────────────────────────────────────────────────────────────────
# RAG tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRAG:
    def test_add_and_retrieve_record(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FAISS_STORE_PATH", str(tmp_path))
        from app.rag import RAGStore, FailureRecord  # fresh import with patched path
        store = RAGStore()

        record = FailureRecord(
            task_preview="Write a binary search function",
            task_type="code",
            error_summary="Syntax error on line 3",
            failed_output="",
            fix_applied="improved_prompt",
            resolution="fixed",
        )
        store.add_record(record)

        results = store.retrieve_similar("binary search implementation", k=1)
        assert len(results) == 1
        assert results[0][0].task_type == "code"

    def test_empty_store_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FAISS_STORE_PATH", str(tmp_path / "empty"))
        from app.rag import RAGStore
        store = RAGStore()
        results = store.retrieve_similar("anything")
        assert results == []


# ─────────────────────────────────────────────────────────────────────────────
# Example API request payloads (for documentation / curl usage)
# ─────────────────────────────────────────────────────────────────────────────

EXAMPLE_REQUESTS = {
    "code_task": {
        "url": "POST http://localhost:8000/execute",
        "body": {
            "task": "Write a Python class implementing a thread-safe LRU cache with O(1) get and put operations.",
            "prefer_fast": False,
            "prefer_local": True,
            "metadata": {"source": "test"}
        }
    },
    "reasoning_task": {
        "url": "POST http://localhost:8000/execute",
        "body": {
            "task": "Compare Redis and Memcached for session storage in a high-traffic web application. Include tradeoffs and a recommendation.",
            "prefer_fast": True,
            "prefer_local": False,
            "metadata": {}
        }
    },
    "complex_reasoning": {
        "url": "POST http://localhost:8000/execute",
        "body": {
            "task": "Design a fault-tolerant event sourcing architecture for a financial ledger system that handles 10,000 transactions per second with exactly-once semantics.",
            "prefer_fast": False,
            "prefer_local": False,
            "metadata": {"priority": "high"}
        }
    },
    "manual_debug": {
        "url": "POST http://localhost:8000/debug",
        "body": {
            "task": "Write a function to reverse a linked list",
            "failed_output": "def reverse(lst): return lst[::-1]",
            "error": "Wrong data structure used",
            "model_used": "groq"
        }
    },
    "get_logs": {
        "url": "GET http://localhost:8000/logs?limit=20",
        "body": None
    }
}


if __name__ == "__main__":
    print("=== Example API Requests ===\n")
    for name, req in EXAMPLE_REQUESTS.items():
        print(f"[{name}]")
        print(f"  {req['url']}")
        if req["body"]:
            print(f"  Body: {json.dumps(req['body'], indent=4)}")
        print()

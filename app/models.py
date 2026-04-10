from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class TaskType(str, Enum):
    CODE      = "code"
    REASONING = "reasoning"
    GENERAL   = "general"
    MATH      = "math"
    UNKNOWN   = "unknown"


class ModelProvider(str, Enum):
    OLLAMA_CODER   = "ollama_coder"
    OLLAMA_GENERAL = "ollama_general"
    OLLAMA_LLAMA   = "ollama_llama"
    GROQ           = "groq"
    GEMINI         = "gemini"


class ComplexityLevel(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


class ModelPreference(str, Enum):
    AUTO    = "auto"
    GROQ    = "groq"
    GEMINI  = "gemini"
    CODER   = "ollama_coder"
    GENERAL = "ollama_general"
    LLAMA   = "ollama_llama"


# ── NEW: error categories for structured failure tracking ──────────────────
class ErrorCategory(str, Enum):
    API_ERROR        = "api_error"
    VALIDATION_ERROR = "validation_error"
    PARSING_ERROR    = "parsing_error"
    TIMEOUT          = "timeout"
    QUOTA_EXCEEDED   = "quota_exceeded"
    MODEL_UNAVAILABLE = "model_unavailable"
    UNKNOWN          = "unknown"


class TaskRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=8000)
    metadata: dict[str, Any] = Field(default_factory=dict)
    model_preference: ModelPreference = Field(default=ModelPreference.AUTO)
    prefer_fast: bool = Field(default=False)
    prefer_local: bool = Field(default=False)


class DebugRequest(BaseModel):
    task: str
    failed_output: str
    error: Optional[str] = None
    model_used: Optional[str] = None


# ── ENHANCED: RouteDecision now includes confidence + structured metadata ──
class RouteDecision(BaseModel):
    task_type: TaskType
    complexity: ComplexityLevel
    model_provider: ModelProvider
    reasoning: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    routing_metadata: dict[str, Any] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    is_valid: bool
    confidence: float = Field(ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    checks: dict[str, Any] = Field(default_factory=dict)
    reason: Optional[str] = None


class ExecutionResult(BaseModel):
    model_config = {"protected_namespaces": ()}

    output: str
    model_provider: ModelProvider
    model_name: str
    latency_ms: float
    tokens_used: Optional[int] = None
    error: Optional[str] = None
    # NEW
    error_category: Optional[ErrorCategory] = None
    estimated_cost_usd: Optional[float] = None


# ── ENHANCED: OrchestratorResult with full structured sections ─────────────
class OrchestratorResult(BaseModel):
    model_config = {"protected_namespaces": ()}

    task: str
    output: str
    model_used: str
    model_provider: str
    task_type: str
    complexity: str
    latency_ms: float
    retries: int
    validation_confidence: float
    rag_context_used: bool
    request_id: str
    error: Optional[str] = None

    # NEW: structured sub-sections
    routing: dict[str, Any] = Field(default_factory=dict)
    validation: dict[str, Any] = Field(default_factory=dict)
    debug: dict[str, Any] = Field(default_factory=dict)
    cost: dict[str, Any] = Field(default_factory=dict)

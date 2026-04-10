"""
validator.py — Strict, task-aware validation with hard constraint enforcement.

Validation layers:
  1.  Empty / too-short check
  2.  Refusal detection
  3.  Truncation detection
  4.  HARD CONSTRAINT GATE (immediate rejection on any violation):
        - stdlib-only enforcement
        - required section headers
        - retry logic presence
        - fallback logic presence
        - modularity requirement
        - production-quality depth
        - unsafe code patterns
  5.  Code syntax check (hard fail on error)
  6.  Code structure checks
  7.  Depth check (rejects shallow/placeholder outputs)
  8.  Completeness heuristic
  9.  Relevance
  10. Required sections (reasoning/general tasks)

Confidence threshold raised to 0.70 — only complete outputs pass.
"""

import ast
import re
from typing import Optional

from app.models import TaskType, ValidationResult

# ── Thresholds ──────────────────────────────────────────────────────────────
MIN_OUTPUT_CHARS    = 20
MIN_CONFIDENCE_VALID = 0.70   # RAISED from 0.50 — strict acceptance only

# ── Patterns ────────────────────────────────────────────────────────────────
_TRUNCATION_PATTERNS = [
    r"\.\.\.$",
    r"etc\.\s*$",
    r"\[TRUNCATED\]",
    r"\(continued\)",
]

_REFUSAL_PATTERNS = [
    r"i (cannot|can't|am unable to)",
    r"as an ai (language model|assistant)",
    r"i don'?t have (access|the ability)",
]

_NON_STDLIB_LIBS = frozenset([
    "requests", "httpx", "aiohttp", "flask", "fastapi", "django",
    "numpy", "pandas", "scipy", "sklearn", "tensorflow", "torch",
    "langchain", "openai", "anthropic", "boto3", "sqlalchemy",
    "celery", "redis", "pymongo", "pydantic",
])

_STDLIB_ONLY_SIGNALS = frozenset([
    "standard library", "stdlib", "no external", "built-in only",
    "no dependencies", "no pip", "without installing",
])

# Required section headers for structured output tasks
# Exact section headers the model MUST produce (case-insensitive match).
# These match the _STRICT_INSTRUCTIONS in executor.py — must stay in sync.
_REQUIRED_SECTIONS = [
    r"###\s*Root Causes of Failure",
    r"###\s*Step-by-Step Improvements",
    r"###\s*Python Implementation",
    r"###\s*Optional Enhancements",
]

# Signals that a task explicitly requests modular code
_MODULARITY_SIGNALS = frozenset([
    "modular", "separate functions", "separate modules", "clean architecture",
    "production-ready", "production ready", "well-structured",
])

# Signals that production-readiness is claimed/required
_PRODUCTION_SIGNALS = frozenset([
    "production", "production-ready", "robust", "fault-tolerant", "enterprise",
])


# ── Reusable layer functions (unchanged from previous version) ───────────────

def _check_empty(output: str) -> tuple[bool, str]:
    if not output or len(output.strip()) < MIN_OUTPUT_CHARS:
        return False, "Output is empty or too short"
    return True, ""


def _check_refusal(output: str) -> tuple[bool, str]:
    lower = output.lower()
    for pat in _REFUSAL_PATTERNS:
        if re.search(pat, lower):
            return False, f"Model refused to answer (pattern: {pat})"
    return True, ""


def _check_truncation(output: str) -> tuple[bool, float, str]:
    lower = output.lower().strip()
    for pat in _TRUNCATION_PATTERNS:
        if re.search(pat, lower):
            return False, 0.2, "Output appears truncated"
    return True, 0.0, ""


def _check_code_syntax(output: str) -> tuple[bool, float, list[str]]:
    code_blocks = re.findall(r"```(?:python)?\n?(.*?)```", output, re.DOTALL)
    if not code_blocks:
        if re.search(r"\bdef \w+|class \w+|import \w+", output):
            code_blocks = [output]
    if not code_blocks:
        return True, 0.0, []

    issues = []
    for block in code_blocks:
        try:
            ast.parse(block.strip())
        except SyntaxError as exc:
            issues.append(f"Syntax error in code block: {exc.msg} (line {exc.lineno})")

    return (len(issues) == 0), (-0.3 if issues else 0.1), issues


def _check_code_structure(task: str, output: str) -> tuple[float, list[str]]:
    task_lower = task.lower()
    issues = []
    delta  = 0.0

    if "function" in task_lower or "def " in task_lower:
        if not re.search(r"\bdef \w+", output):
            issues.append("Task requires a function but no 'def' found in output")
            delta -= 0.15
        else:
            delta += 0.05

    if "class" in task_lower:
        if not re.search(r"\bclass \w+", output):
            issues.append("Task requires a class but no 'class' definition found")
            delta -= 0.10

    if any(kw in task_lower for kw in ["test", "unittest", "pytest"]):
        if not re.search(r"assert|unittest|pytest|def test_", output):
            issues.append("Task involves testing but no test constructs found")
            delta -= 0.10

    return delta, issues


def _check_retry_fallback_modularity(task: str, output: str) -> tuple[float, list[str]]:
    task_lower = task.lower()
    is_system_task = any(kw in task_lower for kw in [
        "retry", "fallback", "production", "robust", "fault-tolerant",
        "error handling", "resilient", "pipeline",
    ])
    if not is_system_task:
        return 0.0, []

    issues = []
    delta  = 0.0

    if "retry" in task_lower and not re.search(r"retry|retries|attempt|backoff", output.lower()):
        issues.append("Task requires retry logic but none detected in output")
        delta -= 0.10

    if "fallback" in task_lower and not re.search(r"fallback|except|alternative", output.lower()):
        issues.append("Task requires fallback logic but none detected")
        delta -= 0.10

    def_count = len(re.findall(r"\bdef \w+", output))
    if def_count >= 3:
        delta += 0.05

    return delta, issues


def _check_completeness(task: str, output: str) -> tuple[float, list[str]]:
    task_words   = len(task.split())
    output_words = len(output.split())
    issues       = []

    if task_words > 50 and output_words < 30:
        issues.append("Response is very short for a complex task")
        return -0.15, issues
    if output_words >= max(task_words * 0.5, 80):
        return 0.10, []
    return 0.0, []


def _check_relevance(task: str, output: str) -> float:
    task_tokens   = set(re.findall(r"\b\w{4,}\b", task.lower()))
    output_tokens = set(re.findall(r"\b\w{4,}\b", output.lower()))
    if not task_tokens:
        return 0.0
    overlap = len(task_tokens & output_tokens) / len(task_tokens)
    return round((overlap - 0.5) * 0.2, 4)


def _check_required_sections(task: str, output: str) -> tuple[float, list[str]]:
    task_lower   = task.lower()
    output_lower = output.lower()

    required = set(re.findall(r"\b[a-z]{5,}\b", task_lower)) - {
        "please", "write", "about", "should", "would", "could", "their",
        "which", "where", "there", "these", "those", "other", "using",
        "given", "based", "above", "below",
    }
    if not required:
        return 0.0, []

    covered = sum(1 for topic in required if topic in output_lower)
    ratio   = covered / len(required)

    if ratio < 0.3:
        return -0.10, [f"Output covers only {int(ratio*100)}% of key topics from the task"]
    if ratio >= 0.6:
        return 0.05, []
    return 0.0, []


# ── NEW: Hard constraint gate ────────────────────────────────────────────────

def _check_constraints(task: str, output: str, task_type) -> tuple[bool, dict, list[str]]:
    """
    Hard constraint gate. ANY failure → immediate rejection.
    Returns (all_passed, constraints_dict, violation_messages).
    """
    task_lower   = task.lower()
    output_lower = output.lower()
    violations: list[str] = []

    # ── C1: stdlib-only ──────────────────────────────────────────────────────
    requires_stdlib = any(sig in task_lower for sig in _STDLIB_ONLY_SIGNALS)
    stdlib_ok = True
    if requires_stdlib:
        import_lines = re.findall(r"^\s*(?:import|from)\s+([\w.]+)", output, re.MULTILINE)
        used_top     = {line.split(".")[0] for line in import_lines}
        bad_libs     = used_top & _NON_STDLIB_LIBS
        if bad_libs:
            stdlib_ok = False
            violations.append(
                f"HARD VIOLATION [stdlib_only] — Non-standard libraries used: "
                f"{', '.join(sorted(bad_libs))}. Task forbids external dependencies."
            )

    # ── C2: required section headers (exact match) ──────────────────────────
    needs_sections = any(kw in task_lower for kw in [
        "root cause", "step-by-step", "implementation", "analysis", "report",
        "improvements", "debug", "explain and fix",
    ])
    sections_ok = True
    if needs_sections:
        # Map each pattern to its human-readable exact header name
        _SECTION_LABELS = {
            _REQUIRED_SECTIONS[0]: "### Root Causes of Failure",
            _REQUIRED_SECTIONS[1]: "### Step-by-Step Improvements",
            _REQUIRED_SECTIONS[2]: "### Python Implementation",
            _REQUIRED_SECTIONS[3]: "### Optional Enhancements",
        }
        missing_headers = [
            _SECTION_LABELS[pat]
            for pat in _REQUIRED_SECTIONS
            if not re.search(pat, output, re.IGNORECASE)
        ]
        if len(missing_headers) >= 2:
            sections_ok = False
            violations.append(
                f"HARD VIOLATION [sections_complete] — {len(missing_headers)} required section(s) missing.\n"
                f"  Missing headers (use EXACT text including ###):\n"
                + "\n".join(f"    {h}" for h in missing_headers) +
                "\n  Headers must match exactly — no abbreviations or variations."
            )
        elif len(missing_headers) == 1:
            sections_ok = False
            violations.append(
                f"HARD VIOLATION [sections_complete] — Missing required section: {missing_headers[0]}"
            )
            # Single missing section — soft penalty instead of hard rejection
            violations_soft = missing_headers  # logged but not a hard fail
            output_lower = output.lower()  # reassign to avoid lint warning

    # ── C3: retry logic ───────────────────────────────────────────────────────
    needs_retry = "retry" in task_lower or "retries" in task_lower
    retry_ok    = True
    if needs_retry:
        if not re.search(r"retry|retries|attempt|backoff|max_attempts", output_lower):
            retry_ok = False
            violations.append(
                "HARD VIOLATION [retry_present] — Task requires retry logic but none found. "
                "Implement explicit retry with attempt counter and backoff."
            )

    # ── C4: fallback logic ────────────────────────────────────────────────────
    needs_fallback = "fallback" in task_lower or "fault-tolerant" in task_lower
    fallback_ok    = True
    if needs_fallback:
        if not re.search(r"fallback|except\s|except:|alternative|backup", output_lower):
            fallback_ok = False
            violations.append(
                "HARD VIOLATION [fallback_present] — Task requires fallback logic but none found. "
                "Add explicit fallback/exception handling paths."
            )

    # ── C5: modularity ────────────────────────────────────────────────────────
    needs_modular = any(sig in task_lower for sig in _MODULARITY_SIGNALS)
    modular_ok    = True
    if needs_modular:
        def_count = len(re.findall(r"\bdef \w+", output))
        if def_count < 2:
            modular_ok = False
            violations.append(
                f"HARD VIOLATION [modular] — Task requires modular code but only {def_count} "
                "function(s) found. Split into at least 2 focused functions/classes."
            )

    # ── C6: production-quality depth ─────────────────────────────────────────
    claims_production = any(sig in task_lower for sig in _PRODUCTION_SIGNALS)
    production_ok     = True
    if claims_production:
        missing_prod = []
        if not re.search(r"try:|except\s|except:", output):
            missing_prod.append("error handling (try/except)")
        if not re.search(r"retry|retries|attempt", output_lower):
            missing_prod.append("retry logic")
        if not re.search(r"fallback|alternative|backup", output_lower):
            missing_prod.append("fallback strategy")
        if missing_prod:
            production_ok = False
            violations.append(
                f"HARD VIOLATION [production_quality] — Production-ready output must include: "
                f"{', '.join(missing_prod)}. All are missing."
            )

    # ── C7: unsafe code patterns ──────────────────────────────────────────────
    quality_ok = True
    if task_type is not None and task_type.value == "code":
        if "json.loads" in output and not re.search(r"try:.*json\.loads", output, re.DOTALL):
            quality_ok = False
            violations.append(
                "HARD VIOLATION [code_quality] — json.loads used without try/except. "
                "Wrap all JSON parsing in try/except to handle malformed input."
            )
        if re.search(r"requests\.|httpx\.|urllib", output_lower):
            if not re.search(r"try:", output):
                quality_ok = False
                violations.append(
                    "HARD VIOLATION [code_quality] — HTTP calls made without try/except "
                    "error handling. All network calls must handle exceptions."
                )

    constraints = {
        "stdlib_only":        stdlib_ok,
        "sections_complete":  sections_ok,
        "retry_present":      retry_ok,
        "fallback_present":   fallback_ok,
        "modular":            modular_ok,
        "production_quality": production_ok,
        "code_quality":       quality_ok,
    }
    return len(violations) == 0, constraints, violations


# ── NEW: Depth / shallow-output check ────────────────────────────────────────

def _check_code_output_quality(task: str, output: str) -> tuple[float, list[str]]:
    """
    Validates the QUALITY of code produced in the output.
    Catches 8 common issues from the production-readiness spec:

      1. Fake validation (json.dumps → json.loads on same object)
      2. Hardcoded "cache" dicts
      3. urllib.request.urlopen called without timeout=
      4. Missing jitter in retry backoff
      5. Bare except Exception (too broad)
      6. Missing email regex validation when required
      7. Empty response body not handled
      8. Nested dict access without .get() safety
    """
    issues: list[str] = []
    delta = 0.0

    # Only run on outputs that contain code
    has_code = bool(re.search(r"```|\bdef \w+|\bimport \b", output))
    if not has_code:
        return 0.0, []

    # 1. Fake validation: dumping then loading same variable
    if re.search(r"json\.dumps\(.*?\).*?json\.loads\(", output, re.DOTALL):
        issues.append(
            "CODE QUALITY [fake_validation] — json.dumps() then json.loads() on the same "
            "object is not real validation. Validate fields, types, and formats directly."
        )
        delta -= 0.15

    # 2. Hardcoded cache dict (literal dict used as "cache" with no timestamp)
    hardcoded_cache = re.search(
        r'(?:cache|cached_data|fallback_data)\s*=\s*\{[^}]{10,}\}', output
    )
    if hardcoded_cache and not re.search(r"time\.time\(\)|timestamp|ttl|TTL", output):
        issues.append(
            "CODE QUALITY [fake_cache] — Cache is a hardcoded dictionary with no TTL or "
            "timestamp. Use a dict + time.time() for real in-memory caching with expiry."
        )
        delta -= 0.15

    # 3. urlopen without timeout
    if re.search(r"urlopen\(", output) and not re.search(r"urlopen\(.*?timeout\s*=", output, re.DOTALL):
        issues.append(
            "CODE QUALITY [missing_timeout] — urllib.request.urlopen() called without "
            "timeout= parameter. Always pass timeout=10 (or similar) to prevent hanging."
        )
        delta -= 0.10

    # 4. Retry loop without jitter (has sleep + exponential but no random)
    has_retry  = bool(re.search(r"for.*attempt|while.*attempt|retry", output, re.IGNORECASE))
    has_sleep  = bool(re.search(r"time\.sleep\(", output))
    has_random = bool(re.search(r"random\.uniform|random\.random|jitter", output))
    if has_retry and has_sleep and not has_random:
        issues.append(
            "CODE QUALITY [missing_jitter] — Retry logic uses sleep() without jitter. "
            "Add random.uniform(0, 1) to backoff delay to prevent thundering herd."
        )
        delta -= 0.08

    # 5. Bare except Exception (too broad — masks real errors)
    if re.search(r"except\s+Exception\s*:", output):
        issues.append(
            "CODE QUALITY [broad_except] — 'except Exception:' catches too broadly. "
            "Use specific exceptions: socket.timeout, urllib.error.URLError, "
            "json.JSONDecodeError, ValueError, TypeError."
        )
        delta -= 0.08

    # 6. Email validation missing when user_id/email fields are involved
    task_lower = task.lower()
    if re.search(r"email|user_id|validate.*field", task_lower):
        if not re.search(r"re\.match|re\.fullmatch|re\.search.*@|@.*regex", output):
            issues.append(
                "CODE QUALITY [missing_email_validation] — Task involves email fields "
                "but no regex email validation found. Add: "
                r"re.fullmatch(r'^[\w.-]+@[\w.-]+\.\w+$', email)"
            )
            delta -= 0.08

    # 7. Empty response body not handled
    if re.search(r"urlopen|http\.client|urllib", output):
        if not re.search(r"\.read\(\)|response_body|raw_data", output):
            pass  # Can't tell without seeing full code — skip
        else:
            # They read the body — check they handle empty case
            if not re.search(r"if not.*body|if.*body.*==|len\(.*body\)|strip\(\)", output):
                issues.append(
                    "CODE QUALITY [empty_response] — Response body is read but empty "
                    "body is not checked. Add: if not body: raise/log and use fallback."
                )
                delta -= 0.06

    # 8. Nested dict access without .get() safety (e.g. data["key"]["nested"])
    unsafe_access = re.findall(r'\w+\["\w+"\]\["\w+"\]', output)
    if len(unsafe_access) >= 2:
        issues.append(
            f"CODE QUALITY [unsafe_dict_access] — {len(unsafe_access)} instance(s) of "
            "chained dict['key']['nested'] access without .get(). Use .get() with defaults "
            "to safely handle missing nested fields."
        )
        delta -= 0.06
    # 9. Retry must NOT wrap validation logic
    if re.search(r"for .*retry.*:.*validate", output, re.DOTALL | re.IGNORECASE):
        issues.append(
            "CODE QUALITY [retry_misuse] — Validation detected inside retry loop. "
            "Retry must wrap ONLY network calls, not validation."
        )
        delta -= 0.12

    # 10. Validation must not return None for invalid data
    if re.search(r"def validate.*?:.*return None", output, re.DOTALL):
        issues.append(
            "CODE QUALITY [reject_instead_of_fix] — Validation returns None. "
            "It must correct invalid data and return a valid structured output."
        )
        delta -= 0.12    

    # 11. Empty response handling required
    if re.search(r"urlopen|read\(", output) and not re.search(r"if not .*data|strip\(\)", output):
        issues.append(
            "CODE QUALITY [missing_empty_check] — No empty response check before parsing."
        )
        delta -= 0.10  

    # 12. clean_data must not be a no-op
    if re.search(r"def clean_data.*return data\s*$", output, re.MULTILINE):
        issues.append(
            "CODE QUALITY [fake_clean] — clean_data() does nothing. "
            "It must transform data or be removed."
        )
        delta -= 0.08   


    # Reward clean, well-structured code
    specific_exceptions = len(re.findall(
        r"except\s+(socket\.timeout|urllib\.error|json\.JSONDecodeError|ValueError|TypeError)",
        output
    ))
    if specific_exceptions >= 2:
        delta += 0.06  # Reward specific exception handling

    has_ttl     = bool(re.search(r"time\.time\(\).*ttl|ttl.*time\.time\(\)|TTL|cache_ttl", output, re.IGNORECASE))
    has_jitter  = bool(re.search(r"random\.uniform|jitter", output))
    if has_ttl:    delta += 0.04
    if has_jitter: delta += 0.04

    return round(delta, 4), issues


def _check_depth(task: str, output: str) -> tuple[float, list[str]]:
    """
    Reject shallow, generic, or placeholder-heavy outputs.
    Returns (confidence_delta, issues).
    """
    issues = []
    delta  = 0.0
    lower  = output.lower()

    placeholders = re.findall(
        r"#\s*(todo|fixme|implement\s+this|your\s+code\s+here|placeholder|pass)",
        lower
    )
    if placeholders:
        delta -= 0.20 * min(len(placeholders), 3)
        issues.append(
            f"Output contains {len(placeholders)} placeholder(s) — implementation is incomplete."
        )

    generic_phrases = [
        "you can use", "it depends", "for example purposes", "in a real application",
        "this is a simplified", "left as an exercise", "beyond the scope",
    ]
    generic_hits = sum(1 for p in generic_phrases if p in lower)
    if generic_hits >= 2:
        delta -= 0.10
        issues.append(
            f"Output uses {generic_hits} generic/evasive phrase(s) — lacks implementation depth."
        )

    depth_markers = [
        r"\berror\b", r"\bexception\b", r"\bedge case\b",
        r"\blog\b", r"\bvalidat", r"\btest\b",
    ]
    depth_hits = sum(1 for p in depth_markers if re.search(p, lower))
    if depth_hits >= 3:
        delta += 0.05

    return delta, issues

def _check_semantic_correctness(task: str, output: str) -> tuple[float, list[str]]:
    """
    Uses lightweight LLM check or heuristic to verify if output answers the task.
    """
    issues = []
    delta = 0.0

    task_tokens = set(task.lower().split())
    output_tokens = set(output.lower().split())

    overlap = len(task_tokens & output_tokens) / (len(task_tokens) + 1e-6)

    if overlap < 0.2:
        issues.append("Semantic mismatch: output does not address task properly")
        delta -= 0.20
    elif overlap > 0.5:
        delta += 0.10

    return delta, issues

def _check_hallucination(output: str) -> tuple[float, list[str]]:
    """
    Detects unsupported confident claims.
    """
    issues = []
    delta = 0.0

    suspicious_phrases = [
        "always", "never", "guaranteed", "100%", "definitely"
    ]

    hits = sum(1 for p in suspicious_phrases if p in output.lower())

    if hits >= 2:
        issues.append("Potential hallucination: overly confident claims detected")
        delta -= 0.15

    return delta, issues

def _check_internal_consistency(output: str) -> tuple[float, list[str]]:
    """
    Detect contradictions inside the response.
    """
    issues = []
    delta = 0.0

    if "however" in output.lower() and "therefore" in output.lower():
        issues.append("Possible contradiction in reasoning flow")
        delta -= 0.05

    return delta, issues

# ── Public API ────────────────────────────────────────────────────────────────

def validate_output(
    task: str,
    output: str,
    task_type: Optional[TaskType] = None,
) -> ValidationResult:

    issues: list[str]      = []
    suggestions: list[str] = []
    checks: dict           = {}
    confidence = 0.60

    # ── Layer 1: Empty ───────────────────────────────────────────
    ok, msg = _check_empty(output)
    checks["empty"] = {"passed": ok}
    if not ok:
        return ValidationResult(
            is_valid=False,
            confidence=0.0,
            issues=[msg],
            suggestions=["Retry with a more explicit prompt"],
            checks=checks,
            reason="empty_output"
        )

    # ── Layer 2: Refusal ─────────────────────────────────────────
    ok, msg = _check_refusal(output)
    checks["refusal"] = {"passed": ok}
    if not ok:
        return ValidationResult(
            is_valid=False,
            confidence=0.10,
            issues=[msg],
            suggestions=[
                "Rephrase task to avoid triggering safety filters",
                "Switch model"
            ],
            checks=checks,
            reason="model_refusal"   # ✅ FIXED
        )

    # ── Layer 3: Truncation ──────────────────────────────────────
    ok, penalty, msg = _check_truncation(output)
    checks["truncation"] = {"passed": ok, "penalty": penalty}
    if not ok:
        confidence -= penalty
        issues.append(msg)
        suggestions.append("Request continuation or increase max_tokens")

    # ── Layer 4: HARD CONSTRAINT GATE ────────────────────────────
    constraints_passed, constraints_dict, violations = _check_constraints(task, output, task_type)
    checks["constraints"] = constraints_dict

    if not constraints_passed:
        issues.extend(violations)
        return ValidationResult(
            is_valid=False,
            confidence=0.0,
            issues=issues,
            suggestions=[
                "Fix ALL HARD VIOLATIONS listed above. No partial fixes accepted.",
                "Every constraint must pass before output is accepted.",
            ],
            checks=checks,
            reason="constraint_violation"   # ✅ ADDED
        )

    # ── Layer 5: Code syntax ─────────────────────────────────────
    if task_type is not None and task_type.value == "code":

        syntax_ok, delta, syntax_issues = _check_code_syntax(output)
        checks["syntax"] = {"passed": syntax_ok, "delta": delta, "issues": syntax_issues}

        if not syntax_ok:
            issues.extend(syntax_issues)
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                issues=issues,
                suggestions=["Fix syntax errors. Code must parse cleanly."],
                checks=checks,
                reason="syntax_error"   # ✅ ADDED
            )

        confidence += delta

        struct_delta, struct_issues = _check_code_structure(task, output)
        checks["code_structure"] = {"delta": struct_delta, "issues": struct_issues}
        confidence += struct_delta
        issues.extend(struct_issues)

        rfm_delta, rfm_issues = _check_retry_fallback_modularity(task, output)
        checks["retry_fallback_modularity"] = {"delta": rfm_delta, "issues": rfm_issues}
        confidence += rfm_delta
        issues.extend(rfm_issues)

        cq_delta, cq_issues = _check_code_output_quality(task, output)
        checks["code_output_quality"] = {"delta": cq_delta, "issues": cq_issues}
        confidence += cq_delta
        issues.extend(cq_issues)

        if cq_issues:
            suggestions.extend([
                "Replace fake validation with real checks.",
                "Add timeout, jitter, and proper exception handling.",
            ])

    # ── Layer 6: Depth ───────────────────────────────────────────
    depth_delta, depth_issues = _check_depth(task, output)
    checks["depth"] = {"delta": depth_delta, "issues": depth_issues}
    confidence += depth_delta
    issues.extend(depth_issues)

    if depth_issues:
        suggestions.append("Remove placeholders and improve implementation depth.")

    # ── Layer 7: Completeness ────────────────────────────────────
    comp_delta, comp_warnings = _check_completeness(task, output)
    checks["completeness"] = {"delta": comp_delta, "issues": comp_warnings}
    confidence += comp_delta
    issues.extend(comp_warnings)

    # ── Layer 8: Relevance ───────────────────────────────────────
    rel_delta = _check_relevance(task, output)
    checks["relevance"] = {"delta": rel_delta}
    confidence += rel_delta

    # ── Layer 9: Semantic correctness ───────────────────────────
    sem_delta, sem_issues = _check_semantic_correctness(task, output)
    checks["semantic_correctness"] = {"delta": sem_delta, "issues": sem_issues}
    confidence += sem_delta
    issues.extend(sem_issues)

    # ── Layer 10: Hallucination detection ───────────────────────
    hall_delta, hall_issues = _check_hallucination(output)
    checks["hallucination"] = {"delta": hall_delta, "issues": hall_issues}
    confidence += hall_delta
    issues.extend(hall_issues)

    # ── Layer 11: Consistency check ─────────────────────────────
    cons_delta, cons_issues = _check_internal_consistency(output)
    checks["consistency"] = {"delta": cons_delta, "issues": cons_issues}
    confidence += cons_delta
    issues.extend(cons_issues)

    # ── Layer 12: Sections ──────────────────────────────────────
    if task_type is None or task_type.value in ("reasoning", "general"):
        sec_delta, sec_issues = _check_required_sections(task, output)
        checks["required_sections"] = {"delta": sec_delta, "issues": sec_issues}
        confidence += sec_delta
        issues.extend(sec_issues)

    # ── Final ───────────────────────────────────────────────────
    confidence = round(max(0.0, min(1.0, confidence)), 4)
    is_valid   = confidence >= MIN_CONFIDENCE_VALID

    reason = None

    if not is_valid:
        if issues:
            reason = issues[0][:120]
        else:
            reason = "unknown_failure"

    return ValidationResult(
        is_valid=is_valid,
        confidence=confidence,
        issues=issues,
        suggestions=suggestions,
        checks=checks,
        reason=reason
    )
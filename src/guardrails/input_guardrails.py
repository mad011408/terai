"""
Input guardrails for validating and sanitizing user inputs.
"""

from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import re
from datetime import datetime


class ValidationResult(Enum):
    """Result of input validation."""
    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"


@dataclass
class GuardrailResult:
    """Result from a guardrail check."""
    passed: bool
    result: ValidationResult
    message: str
    guardrail_name: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "result": self.result.value,
            "message": self.message,
            "guardrail_name": self.guardrail_name,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


class InputGuardrail(ABC):
    """Base class for input guardrails."""

    def __init__(self, name: str, description: str, enabled: bool = True):
        self.name = name
        self.description = description
        self.enabled = enabled

    @abstractmethod
    def check(self, input_text: str, context: Optional[Dict] = None) -> GuardrailResult:
        """Check input against this guardrail."""
        pass


class LengthGuardrail(InputGuardrail):
    """Validates input length."""

    def __init__(self, min_length: int = 1, max_length: int = 10000):
        super().__init__(
            name="length_guardrail",
            description="Validates input length bounds"
        )
        self.min_length = min_length
        self.max_length = max_length

    def check(self, input_text: str, context: Optional[Dict] = None) -> GuardrailResult:
        length = len(input_text)

        if length < self.min_length:
            return GuardrailResult(
                passed=False,
                result=ValidationResult.BLOCK,
                message=f"Input too short: {length} < {self.min_length}",
                guardrail_name=self.name,
                details={"length": length, "min": self.min_length}
            )

        if length > self.max_length:
            return GuardrailResult(
                passed=False,
                result=ValidationResult.BLOCK,
                message=f"Input too long: {length} > {self.max_length}",
                guardrail_name=self.name,
                details={"length": length, "max": self.max_length}
            )

        return GuardrailResult(
            passed=True,
            result=ValidationResult.PASS,
            message="Length check passed",
            guardrail_name=self.name,
            details={"length": length}
        )


class PromptInjectionGuardrail(InputGuardrail):
    """Detects prompt injection attempts."""

    def __init__(self):
        super().__init__(
            name="prompt_injection_guardrail",
            description="Detects prompt injection and jailbreak attempts"
        )
        self.suspicious_patterns = [
            r"ignore\s+(previous|all|above)\s+instructions",
            r"disregard\s+(your|all)\s+(instructions|rules|guidelines)",
            r"you\s+are\s+now\s+(?:a|an)\s+\w+",
            r"pretend\s+(?:you|to)\s+(?:are|be)",
            r"act\s+as\s+(?:if|though)",
            r"from\s+now\s+on",
            r"new\s+instructions",
            r"override\s+(?:your|previous)",
            r"forget\s+(?:everything|all|your)",
            r"system\s*:\s*",
            r"\[system\]",
            r"</?\s*system\s*>",
            r"<\|.*?\|>",
            r"###\s*(?:system|instruction)",
        ]
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.suspicious_patterns]

    def check(self, input_text: str, context: Optional[Dict] = None) -> GuardrailResult:
        matches = []

        for pattern in self.compiled_patterns:
            match = pattern.search(input_text)
            if match:
                matches.append(match.group())

        if matches:
            return GuardrailResult(
                passed=False,
                result=ValidationResult.BLOCK,
                message="Potential prompt injection detected",
                guardrail_name=self.name,
                details={"matches": matches[:5]}  # Limit to 5
            )

        return GuardrailResult(
            passed=True,
            result=ValidationResult.PASS,
            message="No prompt injection detected",
            guardrail_name=self.name
        )


class ContentFilterGuardrail(InputGuardrail):
    """Filters inappropriate content."""

    def __init__(self, blocked_terms: Optional[List[str]] = None):
        super().__init__(
            name="content_filter_guardrail",
            description="Filters inappropriate or harmful content"
        )
        self.blocked_terms = blocked_terms or []
        self.severity_patterns = {
            "high": [
                r"\b(bomb|weapon|explosive|attack|kill|murder)\b",
                r"\b(hack|exploit|malware|virus|trojan)\b",
            ],
            "medium": [
                r"\b(password|credential|secret|api.?key)\b",
            ]
        }

    def check(self, input_text: str, context: Optional[Dict] = None) -> GuardrailResult:
        input_lower = input_text.lower()
        found_terms = []

        # Check blocked terms
        for term in self.blocked_terms:
            if term.lower() in input_lower:
                found_terms.append(term)

        # Check severity patterns
        severity = None
        for level, patterns in self.severity_patterns.items():
            for pattern in patterns:
                if re.search(pattern, input_lower):
                    severity = level
                    break
            if severity:
                break

        if severity == "high":
            return GuardrailResult(
                passed=False,
                result=ValidationResult.BLOCK,
                message="High severity content detected",
                guardrail_name=self.name,
                details={"severity": severity}
            )

        if found_terms:
            return GuardrailResult(
                passed=False,
                result=ValidationResult.WARN,
                message=f"Blocked terms found: {found_terms}",
                guardrail_name=self.name,
                details={"blocked_terms": found_terms}
            )

        if severity == "medium":
            return GuardrailResult(
                passed=True,
                result=ValidationResult.WARN,
                message="Medium severity content detected",
                guardrail_name=self.name,
                details={"severity": severity}
            )

        return GuardrailResult(
            passed=True,
            result=ValidationResult.PASS,
            message="Content filter passed",
            guardrail_name=self.name
        )


class CodeInjectionGuardrail(InputGuardrail):
    """Detects code injection attempts."""

    def __init__(self):
        super().__init__(
            name="code_injection_guardrail",
            description="Detects SQL, XSS, and command injection attempts"
        )
        self.injection_patterns = {
            "sql": [
                r"(?:union\s+(?:all\s+)?select)",
                r"(?:;\s*(?:drop|delete|update|insert))",
                r"(?:'\s*or\s+'?\d+'?\s*=\s*'?\d+)",
                r"(?:--\s*$)",
            ],
            "xss": [
                r"<script[^>]*>",
                r"javascript\s*:",
                r"on\w+\s*=",
                r"<img[^>]+onerror",
            ],
            "command": [
                r";\s*(?:rm|cat|wget|curl|chmod)\s",
                r"\|\s*(?:bash|sh|python|perl)",
                r"\$\([^)]+\)",
                r"`[^`]+`",
            ]
        }

    def check(self, input_text: str, context: Optional[Dict] = None) -> GuardrailResult:
        detected = {}

        for injection_type, patterns in self.injection_patterns.items():
            for pattern in patterns:
                if re.search(pattern, input_text, re.IGNORECASE):
                    detected[injection_type] = True
                    break

        if detected:
            return GuardrailResult(
                passed=False,
                result=ValidationResult.BLOCK,
                message=f"Potential code injection detected: {list(detected.keys())}",
                guardrail_name=self.name,
                details={"injection_types": list(detected.keys())}
            )

        return GuardrailResult(
            passed=True,
            result=ValidationResult.PASS,
            message="No code injection detected",
            guardrail_name=self.name
        )


class InputValidator:
    """
    Orchestrates multiple input guardrails.
    """

    def __init__(self, guardrails: Optional[List[InputGuardrail]] = None):
        self.guardrails = guardrails or self._default_guardrails()
        self.validation_history: List[Dict[str, Any]] = []

    def _default_guardrails(self) -> List[InputGuardrail]:
        """Create default guardrails."""
        return [
            LengthGuardrail(min_length=1, max_length=50000),
            PromptInjectionGuardrail(),
            ContentFilterGuardrail(),
            CodeInjectionGuardrail(),
        ]

    def add_guardrail(self, guardrail: InputGuardrail) -> None:
        """Add a guardrail."""
        self.guardrails.append(guardrail)

    def remove_guardrail(self, name: str) -> bool:
        """Remove a guardrail by name."""
        for i, g in enumerate(self.guardrails):
            if g.name == name:
                self.guardrails.pop(i)
                return True
        return False

    def validate(self, input_text: str, context: Optional[Dict] = None,
                fail_fast: bool = True) -> Tuple[bool, List[GuardrailResult]]:
        """
        Validate input against all guardrails.

        Args:
            input_text: The input to validate
            context: Additional context for validation
            fail_fast: Stop on first failure if True

        Returns:
            Tuple of (passed, list of results)
        """
        results = []
        passed = True

        for guardrail in self.guardrails:
            if not guardrail.enabled:
                continue

            result = guardrail.check(input_text, context)
            results.append(result)

            if result.result == ValidationResult.BLOCK:
                passed = False
                if fail_fast:
                    break

        # Record validation
        self.validation_history.append({
            "timestamp": datetime.now().isoformat(),
            "input_preview": input_text[:100],
            "passed": passed,
            "results": [r.to_dict() for r in results]
        })

        return passed, results

    def validate_and_sanitize(self, input_text: str,
                             context: Optional[Dict] = None) -> Tuple[bool, str, List[GuardrailResult]]:
        """
        Validate and sanitize input.

        Returns:
            Tuple of (passed, sanitized_text, results)
        """
        passed, results = self.validate(input_text, context)

        if passed:
            # Basic sanitization
            sanitized = self._sanitize(input_text)
            return True, sanitized, results

        return False, "", results

    def _sanitize(self, text: str) -> str:
        """Basic input sanitization."""
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total = len(self.validation_history)
        passed = sum(1 for v in self.validation_history if v["passed"])

        return {
            "total_validations": total,
            "passed": passed,
            "blocked": total - passed,
            "pass_rate": passed / total if total > 0 else 0,
            "guardrails": [
                {"name": g.name, "enabled": g.enabled}
                for g in self.guardrails
            ]
        }


def validate_input(func: Callable) -> Callable:
    """Decorator for validating function input."""
    validator = InputValidator()

    async def wrapper(*args, **kwargs):
        # Find input text in args/kwargs
        input_text = kwargs.get("input_text") or kwargs.get("text") or kwargs.get("query")
        if input_text is None and args:
            input_text = args[0] if isinstance(args[0], str) else None

        if input_text:
            passed, results = validator.validate(input_text)
            if not passed:
                raise ValueError(f"Input validation failed: {results[0].message}")

        return await func(*args, **kwargs)

    return wrapper

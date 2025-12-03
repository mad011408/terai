"""
Output guardrails for validating and filtering model outputs.
"""

from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import re
from datetime import datetime


class OutputValidationResult(Enum):
    """Result of output validation."""
    PASS = "pass"
    MODIFY = "modify"
    BLOCK = "block"


@dataclass
class OutputGuardrailResult:
    """Result from an output guardrail check."""
    passed: bool
    result: OutputValidationResult
    original_output: str
    modified_output: Optional[str]
    message: str
    guardrail_name: str
    modifications: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class OutputGuardrail(ABC):
    """Base class for output guardrails."""

    def __init__(self, name: str, description: str, enabled: bool = True):
        self.name = name
        self.description = description
        self.enabled = enabled

    @abstractmethod
    def check(self, output_text: str, context: Optional[Dict] = None) -> OutputGuardrailResult:
        """Check output against this guardrail."""
        pass


class HallucinationGuardrail(OutputGuardrail):
    """Detects potential hallucinations in output."""

    def __init__(self):
        super().__init__(
            name="hallucination_guardrail",
            description="Detects potential hallucinated or fabricated content"
        )
        self.uncertainty_markers = [
            "I think", "I believe", "probably", "might be", "could be",
            "possibly", "perhaps", "may be", "likely", "unlikely"
        ]
        self.confidence_claims = [
            "I am certain", "I am sure", "definitely", "absolutely",
            "without a doubt", "100%", "guaranteed"
        ]

    def check(self, output_text: str, context: Optional[Dict] = None) -> OutputGuardrailResult:
        output_lower = output_text.lower()

        # Check for confident claims without context
        has_confident_claims = any(claim in output_lower for claim in self.confidence_claims)
        has_uncertainty = any(marker in output_lower for marker in self.uncertainty_markers)

        # If making confident claims, flag for review
        if has_confident_claims and not has_uncertainty:
            return OutputGuardrailResult(
                passed=True,
                result=OutputValidationResult.PASS,
                original_output=output_text,
                modified_output=None,
                message="Output contains confident claims - recommend verification",
                guardrail_name=self.name,
                modifications=[{"type": "warning", "reason": "confident_claims"}]
            )

        return OutputGuardrailResult(
            passed=True,
            result=OutputValidationResult.PASS,
            original_output=output_text,
            modified_output=None,
            message="Hallucination check passed",
            guardrail_name=self.name
        )


class SensitiveInfoGuardrail(OutputGuardrail):
    """Filters sensitive information from output."""

    def __init__(self):
        super().__init__(
            name="sensitive_info_guardrail",
            description="Filters sensitive information like credentials and keys"
        )
        self.patterns = {
            "api_key": r"(?:api[_-]?key|apikey)[\"']?\s*[:=]\s*[\"']?[\w\-]{20,}",
            "password": r"(?:password|passwd|pwd)[\"']?\s*[:=]\s*[\"'][^\"']{4,}[\"']",
            "secret": r"(?:secret|token)[\"']?\s*[:=]\s*[\"']?[\w\-]{16,}",
            "aws_key": r"(?:AKIA|ABIA|ACCA|ASIA)[A-Z0-9]{16}",
            "private_key": r"-----BEGIN (?:RSA|DSA|EC|OPENSSH) PRIVATE KEY-----",
            "jwt": r"eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+",
        }

    def check(self, output_text: str, context: Optional[Dict] = None) -> OutputGuardrailResult:
        modifications = []
        modified_text = output_text

        for info_type, pattern in self.patterns.items():
            matches = re.finditer(pattern, modified_text, re.IGNORECASE)
            for match in matches:
                # Redact the sensitive info
                redacted = f"[REDACTED_{info_type.upper()}]"
                modified_text = modified_text.replace(match.group(), redacted)
                modifications.append({
                    "type": "redaction",
                    "info_type": info_type,
                    "position": match.start()
                })

        if modifications:
            return OutputGuardrailResult(
                passed=True,
                result=OutputValidationResult.MODIFY,
                original_output=output_text,
                modified_output=modified_text,
                message=f"Redacted {len(modifications)} sensitive items",
                guardrail_name=self.name,
                modifications=modifications
            )

        return OutputGuardrailResult(
            passed=True,
            result=OutputValidationResult.PASS,
            original_output=output_text,
            modified_output=None,
            message="No sensitive information detected",
            guardrail_name=self.name
        )


class ToxicityGuardrail(OutputGuardrail):
    """Filters toxic or harmful content from output."""

    def __init__(self):
        super().__init__(
            name="toxicity_guardrail",
            description="Filters toxic, harmful, or inappropriate content"
        )
        # Basic patterns - in production use ML classifier
        self.toxic_patterns = [
            r"\b(hate|kill|harm|attack)\s+(you|them|everyone)\b",
            r"\b(stupid|idiot|dumb)\s+(user|person|human)\b",
        ]
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.toxic_patterns]

    def check(self, output_text: str, context: Optional[Dict] = None) -> OutputGuardrailResult:
        for pattern in self.compiled_patterns:
            if pattern.search(output_text):
                return OutputGuardrailResult(
                    passed=False,
                    result=OutputValidationResult.BLOCK,
                    original_output=output_text,
                    modified_output=None,
                    message="Toxic content detected - output blocked",
                    guardrail_name=self.name
                )

        return OutputGuardrailResult(
            passed=True,
            result=OutputValidationResult.PASS,
            original_output=output_text,
            modified_output=None,
            message="Toxicity check passed",
            guardrail_name=self.name
        )


class CodeSafetyGuardrail(OutputGuardrail):
    """Validates code in output for safety."""

    def __init__(self):
        super().__init__(
            name="code_safety_guardrail",
            description="Validates code output for safety issues"
        )
        self.dangerous_patterns = [
            (r"os\.system\s*\(", "os.system call"),
            (r"subprocess\.(?:call|run|Popen)\s*\(", "subprocess call"),
            (r"eval\s*\(", "eval call"),
            (r"exec\s*\(", "exec call"),
            (r"__import__\s*\(", "__import__ call"),
            (r"rm\s+-rf\s+/", "dangerous rm command"),
            (r"chmod\s+777", "insecure permissions"),
            (r"curl\s+.*\|\s*(?:bash|sh)", "pipe to shell"),
        ]

    def check(self, output_text: str, context: Optional[Dict] = None) -> OutputGuardrailResult:
        # Check if output contains code blocks
        code_blocks = re.findall(r"```[\w]*\n(.*?)```", output_text, re.DOTALL)

        warnings = []
        for code in code_blocks:
            for pattern, description in self.dangerous_patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    warnings.append(description)

        if warnings:
            # Add warning but don't block
            warning_text = "\n\n⚠️ **Security Notice**: This code contains potentially dangerous operations: " + ", ".join(set(warnings))

            return OutputGuardrailResult(
                passed=True,
                result=OutputValidationResult.MODIFY,
                original_output=output_text,
                modified_output=output_text + warning_text,
                message=f"Code safety warnings: {warnings}",
                guardrail_name=self.name,
                modifications=[{"type": "warning_added", "warnings": warnings}]
            )

        return OutputGuardrailResult(
            passed=True,
            result=OutputValidationResult.PASS,
            original_output=output_text,
            modified_output=None,
            message="Code safety check passed",
            guardrail_name=self.name
        )


class LengthGuardrail(OutputGuardrail):
    """Validates and truncates output length."""

    def __init__(self, max_length: int = 50000):
        super().__init__(
            name="output_length_guardrail",
            description="Validates and truncates output length"
        )
        self.max_length = max_length

    def check(self, output_text: str, context: Optional[Dict] = None) -> OutputGuardrailResult:
        if len(output_text) > self.max_length:
            truncated = output_text[:self.max_length] + "\n\n[Output truncated due to length]"
            return OutputGuardrailResult(
                passed=True,
                result=OutputValidationResult.MODIFY,
                original_output=output_text,
                modified_output=truncated,
                message=f"Output truncated from {len(output_text)} to {self.max_length} characters",
                guardrail_name=self.name,
                modifications=[{"type": "truncation", "original_length": len(output_text)}]
            )

        return OutputGuardrailResult(
            passed=True,
            result=OutputValidationResult.PASS,
            original_output=output_text,
            modified_output=None,
            message="Length check passed",
            guardrail_name=self.name
        )


class OutputValidator:
    """
    Orchestrates multiple output guardrails.
    """

    def __init__(self, guardrails: Optional[List[OutputGuardrail]] = None):
        self.guardrails = guardrails or self._default_guardrails()
        self.validation_history: List[Dict[str, Any]] = []

    def _default_guardrails(self) -> List[OutputGuardrail]:
        """Create default guardrails."""
        return [
            SensitiveInfoGuardrail(),
            ToxicityGuardrail(),
            CodeSafetyGuardrail(),
            LengthGuardrail(),
            HallucinationGuardrail(),
        ]

    def add_guardrail(self, guardrail: OutputGuardrail) -> None:
        """Add a guardrail."""
        self.guardrails.append(guardrail)

    def validate(self, output_text: str, context: Optional[Dict] = None,
                apply_modifications: bool = True) -> Tuple[bool, str, List[OutputGuardrailResult]]:
        """
        Validate output against all guardrails.

        Args:
            output_text: The output to validate
            context: Additional context
            apply_modifications: Whether to apply suggested modifications

        Returns:
            Tuple of (passed, final_output, results)
        """
        results = []
        current_output = output_text
        passed = True

        for guardrail in self.guardrails:
            if not guardrail.enabled:
                continue

            result = guardrail.check(current_output, context)
            results.append(result)

            if result.result == OutputValidationResult.BLOCK:
                passed = False
                break

            if result.result == OutputValidationResult.MODIFY and apply_modifications:
                if result.modified_output:
                    current_output = result.modified_output

        # Record validation
        self.validation_history.append({
            "timestamp": datetime.now().isoformat(),
            "passed": passed,
            "original_length": len(output_text),
            "final_length": len(current_output),
            "modifications_applied": output_text != current_output
        })

        return passed, current_output, results

    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total = len(self.validation_history)
        passed = sum(1 for v in self.validation_history if v["passed"])
        modified = sum(1 for v in self.validation_history if v.get("modifications_applied"))

        return {
            "total_validations": total,
            "passed": passed,
            "blocked": total - passed,
            "modified": modified,
            "guardrails": [
                {"name": g.name, "enabled": g.enabled}
                for g in self.guardrails
            ]
        }

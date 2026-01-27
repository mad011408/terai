"""
Safety classifier for detecting threats and harmful content.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import re


class ThreatCategory(Enum):
    """Categories of threats."""
    NONE = "none"
    JAILBREAK = "jailbreak"
    PROMPT_INJECTION = "prompt_injection"
    HARMFUL_CONTENT = "harmful_content"
    PERSONAL_INFO = "personal_info"
    ILLEGAL_ACTIVITY = "illegal_activity"
    MALWARE = "malware"
    SPAM = "spam"
    MISINFORMATION = "misinformation"


class SeverityLevel(Enum):
    """Severity levels for detected threats."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ThreatDetection:
    """Result of threat detection."""
    category: ThreatCategory
    severity: SeverityLevel
    confidence: float
    description: str
    evidence: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "description": self.description,
            "evidence": self.evidence,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ClassificationResult:
    """Result of safety classification."""
    is_safe: bool
    threats: List[ThreatDetection]
    overall_severity: SeverityLevel
    recommendation: str
    processing_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_safe": self.is_safe,
            "threats": [t.to_dict() for t in self.threats],
            "overall_severity": self.overall_severity.value,
            "recommendation": self.recommendation,
            "processing_time": self.processing_time
        }


class SafetyClassifier:
    """
    Classifies content for safety threats.
    Uses rule-based detection with optional ML enhancement.
    """

    def __init__(self, use_ml_classifier: bool = False):
        self.use_ml_classifier = use_ml_classifier
        self.ml_classifier = None
        self._load_detection_rules()
        self.classification_history: List[ClassificationResult] = []

    def _load_detection_rules(self) -> None:
        """Load detection rules for various threat categories."""
        self.jailbreak_patterns = [
            (r"ignore\s+(?:all\s+)?(?:previous\s+)?instructions", 0.9, "Direct instruction override"),
            (r"you\s+are\s+now\s+(?:a|an)\s+(?!assistant)\w+", 0.8, "Role reassignment"),
            (r"pretend\s+(?:you\s+are|to\s+be)", 0.7, "Role pretending"),
            (r"act\s+as\s+(?:if|though)", 0.6, "Role acting"),
            (r"from\s+now\s+on", 0.5, "Instruction change"),
            (r"DAN|do\s+anything\s+now", 0.95, "DAN jailbreak"),
            (r"developer\s+mode", 0.85, "Developer mode"),
            (r"jailbreak|jail\s*break", 0.95, "Explicit jailbreak"),
        ]

        self.prompt_injection_patterns = [
            (r"<\|.*?\|>", 0.9, "Special token injection"),
            (r"\[system\]|\[SYSTEM\]", 0.85, "System tag injection"),
            (r"</?\s*system\s*>", 0.85, "XML system tag"),
            (r"###\s*(?:system|instruction)", 0.8, "Markdown system marker"),
            (r"STOP|END|RESET", 0.5, "Control word"),
        ]

        self.harmful_content_patterns = [
            (r"\b(?:create|make|build)\s+(?:a\s+)?(?:bomb|weapon|explosive)", 0.95, "Weapon creation"),
            (r"\b(?:how\s+to\s+)?(?:hack|exploit|breach)\s+(?:into)?", 0.7, "Hacking request"),
            (r"\b(?:kill|murder|harm)\s+(?:someone|people|person)", 0.9, "Violence"),
            (r"\bsuicide\s+(?:method|how|way)", 0.95, "Self-harm"),
        ]

        self.illegal_activity_patterns = [
            (r"\b(?:buy|sell|get)\s+(?:drugs|cocaine|heroin|meth)", 0.9, "Drug trafficking"),
            (r"\b(?:fake|counterfeit)\s+(?:id|passport|money)", 0.9, "Counterfeiting"),
            (r"\b(?:launder|laundering)\s+money", 0.85, "Money laundering"),
            (r"\b(?:child|underage).*(?:explicit|sexual)", 0.99, "CSAM"),
        ]

        self.malware_patterns = [
            (r"\b(?:create|write|code)\s+(?:a\s+)?(?:virus|malware|trojan|ransomware)", 0.9, "Malware creation"),
            (r"\b(?:keylogger|spyware|rootkit)", 0.8, "Malicious software"),
            (r"\bphishing\s+(?:page|site|email)", 0.85, "Phishing"),
        ]

    def classify(self, text: str, context: Optional[Dict] = None) -> ClassificationResult:
        """
        Classify text for safety threats.

        Args:
            text: Text to classify
            context: Additional context

        Returns:
            ClassificationResult with detected threats
        """
        import time
        start_time = time.time()

        threats = []

        # Check each category
        threats.extend(self._check_category(
            text, self.jailbreak_patterns, ThreatCategory.JAILBREAK
        ))
        threats.extend(self._check_category(
            text, self.prompt_injection_patterns, ThreatCategory.PROMPT_INJECTION
        ))
        threats.extend(self._check_category(
            text, self.harmful_content_patterns, ThreatCategory.HARMFUL_CONTENT
        ))
        threats.extend(self._check_category(
            text, self.illegal_activity_patterns, ThreatCategory.ILLEGAL_ACTIVITY
        ))
        threats.extend(self._check_category(
            text, self.malware_patterns, ThreatCategory.MALWARE
        ))

        # Use ML classifier if available
        if self.use_ml_classifier and self.ml_classifier:
            ml_threats = self._ml_classify(text)
            threats.extend(ml_threats)

        # Determine overall severity
        if threats:
            max_severity = max(t.severity.value for t in threats)
            overall_severity = SeverityLevel(max_severity)
        else:
            overall_severity = SeverityLevel.NONE

        # Determine if safe
        is_safe = overall_severity.value < SeverityLevel.HIGH.value

        # Generate recommendation
        recommendation = self._generate_recommendation(threats, overall_severity)

        result = ClassificationResult(
            is_safe=is_safe,
            threats=threats,
            overall_severity=overall_severity,
            recommendation=recommendation,
            processing_time=time.time() - start_time
        )

        self.classification_history.append(result)
        return result

    def _check_category(self, text: str, patterns: List[Tuple],
                       category: ThreatCategory) -> List[ThreatDetection]:
        """Check text against patterns for a specific category."""
        threats = []
        text_lower = text.lower()

        for pattern, confidence, description in patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                # Determine severity based on confidence
                if confidence >= 0.9:
                    severity = SeverityLevel.CRITICAL
                elif confidence >= 0.7:
                    severity = SeverityLevel.HIGH
                elif confidence >= 0.5:
                    severity = SeverityLevel.MEDIUM
                else:
                    severity = SeverityLevel.LOW

                threats.append(ThreatDetection(
                    category=category,
                    severity=severity,
                    confidence=confidence,
                    description=description,
                    evidence=[match.group()]
                ))

        return threats

    def _ml_classify(self, text: str) -> List[ThreatDetection]:
        """Use ML classifier for additional detection."""
        # Placeholder for ML classification
        # In production, would use a trained model
        return []

    def _generate_recommendation(self, threats: List[ThreatDetection],
                                 severity: SeverityLevel) -> str:
        """Generate recommendation based on threats."""
        if severity == SeverityLevel.NONE:
            return "Content appears safe. Proceed normally."
        elif severity == SeverityLevel.LOW:
            return "Minor concerns detected. Review before proceeding."
        elif severity == SeverityLevel.MEDIUM:
            return "Moderate risks detected. Manual review recommended."
        elif severity == SeverityLevel.HIGH:
            return "High-risk content detected. Do not process without review."
        else:  # CRITICAL
            return "Critical threat detected. Block immediately and report."

    def is_safe(self, text: str, max_severity: SeverityLevel = SeverityLevel.MEDIUM) -> bool:
        """Quick check if text is safe."""
        result = self.classify(text)
        return result.overall_severity.value <= max_severity.value

    def get_threat_summary(self, result: ClassificationResult) -> str:
        """Get a human-readable summary of threats."""
        if not result.threats:
            return "No threats detected."

        summary_parts = [f"Detected {len(result.threats)} potential threat(s):"]
        for threat in result.threats:
            summary_parts.append(
                f"  - {threat.category.value}: {threat.description} "
                f"(severity: {threat.severity.name}, confidence: {threat.confidence:.0%})"
            )

        summary_parts.append(f"\nOverall severity: {result.overall_severity.name}")
        summary_parts.append(f"Recommendation: {result.recommendation}")

        return "\n".join(summary_parts)

    def get_stats(self) -> Dict[str, Any]:
        """Get classification statistics."""
        total = len(self.classification_history)
        safe = sum(1 for r in self.classification_history if r.is_safe)

        threat_counts = {}
        for result in self.classification_history:
            for threat in result.threats:
                cat = threat.category.value
                threat_counts[cat] = threat_counts.get(cat, 0) + 1

        return {
            "total_classifications": total,
            "safe_count": safe,
            "unsafe_count": total - safe,
            "threat_categories": threat_counts,
            "average_processing_time": sum(r.processing_time for r in self.classification_history) / total if total > 0 else 0
        }


class RateLimitedClassifier(SafetyClassifier):
    """
    Safety classifier with rate limiting to prevent abuse.
    """

    def __init__(self, max_requests_per_minute: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.max_requests_per_minute = max_requests_per_minute
        self.request_times: List[datetime] = []

    def classify(self, text: str, context: Optional[Dict] = None) -> ClassificationResult:
        """Classify with rate limiting."""
        now = datetime.now()

        # Clean old requests
        self.request_times = [
            t for t in self.request_times
            if (now - t).total_seconds() < 60
        ]

        # Check rate limit
        if len(self.request_times) >= self.max_requests_per_minute:
            return ClassificationResult(
                is_safe=False,
                threats=[ThreatDetection(
                    category=ThreatCategory.SPAM,
                    severity=SeverityLevel.MEDIUM,
                    confidence=1.0,
                    description="Rate limit exceeded"
                )],
                overall_severity=SeverityLevel.MEDIUM,
                recommendation="Rate limit exceeded. Please wait."
            )

        self.request_times.append(now)
        return super().classify(text, context)

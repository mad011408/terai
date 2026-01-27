"""
PII (Personally Identifiable Information) detection and filtering.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import re


class PIIType(Enum):
    """Types of PII."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    ADDRESS = "address"
    NAME = "name"
    DATE_OF_BIRTH = "date_of_birth"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"
    BANK_ACCOUNT = "bank_account"
    MEDICAL_ID = "medical_id"


class RedactionMethod(Enum):
    """Methods for redacting PII."""
    MASK = "mask"  # Replace with ***
    HASH = "hash"  # Replace with hash
    TYPE_LABEL = "type_label"  # Replace with [EMAIL], [PHONE], etc.
    REMOVE = "remove"  # Remove entirely


@dataclass
class PIIDetection:
    """A detected PII instance."""
    pii_type: PIIType
    value: str
    start_pos: int
    end_pos: int
    confidence: float
    context: str = ""


@dataclass
class PIIFilterResult:
    """Result of PII filtering."""
    original_text: str
    filtered_text: str
    detections: List[PIIDetection]
    detection_count: int
    types_found: List[PIIType]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filtered_text": self.filtered_text,
            "detection_count": self.detection_count,
            "types_found": [t.value for t in self.types_found],
            "timestamp": self.timestamp.isoformat()
        }


class PIIFilter:
    """
    Detects and filters PII from text.
    """

    def __init__(self, redaction_method: RedactionMethod = RedactionMethod.TYPE_LABEL):
        self.redaction_method = redaction_method
        self._compile_patterns()
        self.detection_history: List[PIIFilterResult] = []

    def _compile_patterns(self) -> None:
        """Compile regex patterns for PII detection."""
        self.patterns = {
            PIIType.EMAIL: (
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                0.95
            ),
            PIIType.PHONE: (
                r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
                0.85
            ),
            PIIType.SSN: (
                r'\b(?!000|666|9\d{2})\d{3}[-\s]?(?!00)\d{2}[-\s]?(?!0000)\d{4}\b',
                0.90
            ),
            PIIType.CREDIT_CARD: (
                r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b',
                0.95
            ),
            PIIType.IP_ADDRESS: (
                r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
                0.90
            ),
            PIIType.DATE_OF_BIRTH: (
                r'\b(?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12][0-9]|3[01])[-/](?:19|20)\d{2}\b',
                0.70
            ),
            PIIType.PASSPORT: (
                r'\b[A-Z]{1,2}[0-9]{6,9}\b',
                0.60
            ),
            PIIType.BANK_ACCOUNT: (
                r'\b[0-9]{8,17}\b',
                0.50  # Lower confidence - many false positives
            ),
        }

        # Compile patterns
        self.compiled_patterns = {
            pii_type: (re.compile(pattern, re.IGNORECASE), confidence)
            for pii_type, (pattern, confidence) in self.patterns.items()
        }

        # Address pattern (more complex)
        self.address_pattern = re.compile(
            r'\b\d{1,5}\s+[\w\s]+(?:street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr|lane|ln|court|ct|way|place|pl)\b',
            re.IGNORECASE
        )

    def detect(self, text: str) -> List[PIIDetection]:
        """
        Detect PII in text.

        Args:
            text: Text to scan

        Returns:
            List of PII detections
        """
        detections = []

        for pii_type, (pattern, confidence) in self.compiled_patterns.items():
            for match in pattern.finditer(text):
                # Get context around match
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 20)
                context = text[start:end]

                # Adjust confidence based on context
                adjusted_confidence = self._adjust_confidence(
                    pii_type, match.group(), context, confidence
                )

                if adjusted_confidence >= 0.5:  # Minimum confidence threshold
                    detections.append(PIIDetection(
                        pii_type=pii_type,
                        value=match.group(),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=adjusted_confidence,
                        context=context
                    ))

        # Check for addresses
        for match in self.address_pattern.finditer(text):
            detections.append(PIIDetection(
                pii_type=PIIType.ADDRESS,
                value=match.group(),
                start_pos=match.start(),
                end_pos=match.end(),
                confidence=0.75,
                context=text[max(0, match.start()-20):min(len(text), match.end()+20)]
            ))

        # Sort by position
        detections.sort(key=lambda d: d.start_pos)

        return detections

    def _adjust_confidence(self, pii_type: PIIType, value: str,
                          context: str, base_confidence: float) -> float:
        """Adjust confidence based on context."""
        confidence = base_confidence

        # Context keywords that increase confidence
        pii_context_keywords = {
            PIIType.EMAIL: ["email", "contact", "address", "@"],
            PIIType.PHONE: ["phone", "call", "mobile", "tel", "fax"],
            PIIType.SSN: ["ssn", "social security", "ss#"],
            PIIType.CREDIT_CARD: ["card", "visa", "mastercard", "payment"],
            PIIType.DATE_OF_BIRTH: ["born", "dob", "birthday", "birth date"],
        }

        keywords = pii_context_keywords.get(pii_type, [])
        context_lower = context.lower()

        for keyword in keywords:
            if keyword in context_lower:
                confidence = min(1.0, confidence + 0.1)
                break

        # Reduce confidence for likely false positives
        if pii_type == PIIType.BANK_ACCOUNT:
            # Bank accounts shouldn't be in code or IDs
            if any(kw in context_lower for kw in ["id", "code", "version", "line"]):
                confidence *= 0.5

        return confidence

    def filter(self, text: str, pii_types: Optional[List[PIIType]] = None) -> PIIFilterResult:
        """
        Filter PII from text.

        Args:
            text: Text to filter
            pii_types: Specific PII types to filter (None = all)

        Returns:
            PIIFilterResult with filtered text
        """
        detections = self.detect(text)

        # Filter by types if specified
        if pii_types:
            detections = [d for d in detections if d.pii_type in pii_types]

        # Apply redaction
        filtered_text = self._redact(text, detections)

        # Get unique types found
        types_found = list(set(d.pii_type for d in detections))

        result = PIIFilterResult(
            original_text=text,
            filtered_text=filtered_text,
            detections=detections,
            detection_count=len(detections),
            types_found=types_found
        )

        self.detection_history.append(result)
        return result

    def _redact(self, text: str, detections: List[PIIDetection]) -> str:
        """Apply redaction to detected PII."""
        if not detections:
            return text

        # Sort by position descending to preserve positions during replacement
        sorted_detections = sorted(detections, key=lambda d: d.start_pos, reverse=True)

        result = text
        for detection in sorted_detections:
            replacement = self._get_replacement(detection)
            result = result[:detection.start_pos] + replacement + result[detection.end_pos:]

        return result

    def _get_replacement(self, detection: PIIDetection) -> str:
        """Get replacement text for a PII detection."""
        if self.redaction_method == RedactionMethod.MASK:
            # Preserve length with asterisks
            return '*' * len(detection.value)

        elif self.redaction_method == RedactionMethod.HASH:
            import hashlib
            hash_val = hashlib.sha256(detection.value.encode()).hexdigest()[:8]
            return f"[HASH:{hash_val}]"

        elif self.redaction_method == RedactionMethod.TYPE_LABEL:
            return f"[{detection.pii_type.value.upper()}]"

        elif self.redaction_method == RedactionMethod.REMOVE:
            return ""

        return detection.value

    def scan_and_report(self, text: str) -> Dict[str, Any]:
        """
        Scan text and generate a PII report.

        Args:
            text: Text to scan

        Returns:
            Report dictionary
        """
        detections = self.detect(text)

        report = {
            "text_length": len(text),
            "total_pii_found": len(detections),
            "pii_by_type": {},
            "high_confidence_count": 0,
            "detections": []
        }

        for detection in detections:
            pii_type = detection.pii_type.value
            if pii_type not in report["pii_by_type"]:
                report["pii_by_type"][pii_type] = 0
            report["pii_by_type"][pii_type] += 1

            if detection.confidence >= 0.8:
                report["high_confidence_count"] += 1

            report["detections"].append({
                "type": detection.pii_type.value,
                "position": detection.start_pos,
                "confidence": detection.confidence,
                # Don't include actual value in report
                "length": len(detection.value)
            })

        return report

    def get_stats(self) -> Dict[str, Any]:
        """Get PII detection statistics."""
        total_scans = len(self.detection_history)
        total_detections = sum(r.detection_count for r in self.detection_history)

        type_counts = {}
        for result in self.detection_history:
            for pii_type in result.types_found:
                type_name = pii_type.value
                type_counts[type_name] = type_counts.get(type_name, 0) + 1

        return {
            "total_scans": total_scans,
            "total_detections": total_detections,
            "detections_per_scan": total_detections / total_scans if total_scans > 0 else 0,
            "type_distribution": type_counts,
            "redaction_method": self.redaction_method.value
        }


class PIIAwareFilter:
    """
    Combines PII filtering with context awareness.
    """

    def __init__(self):
        self.pii_filter = PIIFilter()
        self.allowed_contexts: Dict[PIIType, List[str]] = {
            # Some contexts where PII might be expected/allowed
            PIIType.EMAIL: ["contact form", "registration", "account"],
            PIIType.PHONE: ["contact", "support", "emergency"],
        }

    def filter_with_context(self, text: str, context: Optional[str] = None,
                           strict: bool = False) -> PIIFilterResult:
        """
        Filter PII with context awareness.

        Args:
            text: Text to filter
            context: Context description (e.g., "contact form")
            strict: If True, filter all PII regardless of context

        Returns:
            PIIFilterResult
        """
        if strict or not context:
            return self.pii_filter.filter(text)

        # Detect all PII
        detections = self.pii_filter.detect(text)

        # Filter out allowed PII based on context
        context_lower = context.lower()
        filtered_detections = []

        for detection in detections:
            allowed_contexts = self.allowed_contexts.get(detection.pii_type, [])
            is_allowed = any(ac in context_lower for ac in allowed_contexts)

            if not is_allowed:
                filtered_detections.append(detection)

        # Apply redaction only to non-allowed detections
        filtered_text = self.pii_filter._redact(text, filtered_detections)

        return PIIFilterResult(
            original_text=text,
            filtered_text=filtered_text,
            detections=filtered_detections,
            detection_count=len(filtered_detections),
            types_found=list(set(d.pii_type for d in filtered_detections))
        )

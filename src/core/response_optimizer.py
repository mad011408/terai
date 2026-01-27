"""
Advanced Response Optimizer - AI-Driven Response Enhancement

This module provides AI-driven optimization for responses:
- Automatic quality detection
- Response structure optimization
- Code formatting and validation
- Markdown enhancement
- Context-aware improvements
"""

import re
import asyncio
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ContentType(Enum):
    """Types of content in responses."""
    TEXT = "text"
    CODE = "code"
    MARKDOWN = "markdown"
    JSON = "json"
    MIXED = "mixed"


class QualityLevel(Enum):
    """Response quality levels."""
    EXCELLENT = 5
    GOOD = 4
    AVERAGE = 3
    POOR = 2
    VERY_POOR = 1


@dataclass
class OptimizationResult:
    """Result of response optimization."""
    original: str
    optimized: str
    quality_before: QualityLevel
    quality_after: QualityLevel
    improvements: List[str]
    content_type: ContentType


@dataclass
class ResponseAnalysis:
    """Analysis of a response."""
    content_type: ContentType
    quality_score: float
    readability_score: float
    structure_score: float
    completeness_score: float
    issues: List[str]
    suggestions: List[str]


class ContentDetector:
    """Detects the type of content in a response."""
    
    CODE_PATTERNS = [
        r'```[\w]*\n[\s\S]*?```',  # Markdown code blocks
        r'def\s+\w+\s*\([^)]*\)\s*:',  # Python functions
        r'class\s+\w+',  # Class definitions
        r'import\s+\w+',  # Import statements
        r'function\s+\w+',  # JavaScript functions
        r'\{[\s\S]*\}',  # JSON-like structures
    ]
    
    MARKDOWN_PATTERNS = [
        r'^#+\s+.+$',  # Headers
        r'\*\*.+\*\*',  # Bold
        r'\*.+\*',  # Italic
        r'^\s*[-*]\s+.+$',  # Lists
        r'\[.+\]\(.+\)',  # Links
    ]
    
    def detect(self, content: str) -> ContentType:
        """Detect content type."""
        if not content:
            return ContentType.TEXT
        
        code_matches = sum(1 for p in self.CODE_PATTERNS if re.search(p, content, re.MULTILINE))
        md_matches = sum(1 for p in self.MARKDOWN_PATTERNS if re.search(p, content, re.MULTILINE))
        
        # Check for JSON
        try:
            import json
            json.loads(content.strip())
            return ContentType.JSON
        except:
            pass
        
        if code_matches > 2:
            if md_matches > 2:
                return ContentType.MIXED
            return ContentType.CODE
        
        if md_matches > 2:
            return ContentType.MARKDOWN
        
        return ContentType.TEXT


class QualityAnalyzer:
    """Analyzes response quality."""
    
    def analyze(self, response: str) -> ResponseAnalysis:
        """Perform comprehensive quality analysis."""
        content_detector = ContentDetector()
        content_type = content_detector.detect(response)
        
        # Calculate scores
        quality = self._calculate_quality(response)
        readability = self._calculate_readability(response)
        structure = self._calculate_structure(response, content_type)
        completeness = self._calculate_completeness(response)
        
        # Identify issues
        issues = self._identify_issues(response, content_type)
        suggestions = self._generate_suggestions(response, content_type, issues)
        
        return ResponseAnalysis(
            content_type=content_type,
            quality_score=quality,
            readability_score=readability,
            structure_score=structure,
            completeness_score=completeness,
            issues=issues,
            suggestions=suggestions
        )
    
    def _calculate_quality(self, response: str) -> float:
        """Calculate overall quality score (0-1)."""
        if not response:
            return 0.0
        
        score = 0.5  # Base score
        
        # Length bonus
        if len(response) > 100:
            score += 0.1
        if len(response) > 500:
            score += 0.1
        if len(response) > 1000:
            score += 0.1
        
        # Has structure
        if re.search(r'^#+\s+', response, re.MULTILINE):
            score += 0.1
        
        # Has code examples
        if '```' in response:
            score += 0.1
        
        return min(1.0, score)
    
    def _calculate_readability(self, response: str) -> float:
        """Calculate readability score (0-1)."""
        if not response:
            return 0.0
        
        # Simple readability heuristics
        sentences = response.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        # Optimal sentence length is 15-20 words
        if 10 <= avg_sentence_length <= 25:
            return 0.9
        elif avg_sentence_length < 5 or avg_sentence_length > 40:
            return 0.5
        else:
            return 0.7
    
    def _calculate_structure(self, response: str, content_type: ContentType) -> float:
        """Calculate structure score (0-1)."""
        score = 0.5
        
        # Check for headers
        if re.search(r'^#+\s+', response, re.MULTILINE):
            score += 0.2
        
        # Check for lists
        if re.search(r'^\s*[-*\d+.]\s+', response, re.MULTILINE):
            score += 0.15
        
        # Check for paragraphs
        if '\n\n' in response:
            score += 0.15
        
        return min(1.0, score)
    
    def _calculate_completeness(self, response: str) -> float:
        """Calculate completeness score (0-1)."""
        if not response:
            return 0.0
        
        # Check for incomplete sentences
        if response.rstrip().endswith(('...', '…')):
            return 0.6
        
        if len(response) < 50:
            return 0.5
        
        return 0.9
    
    def _identify_issues(self, response: str, content_type: ContentType) -> List[str]:
        """Identify issues in the response."""
        issues = []
        
        if not response:
            issues.append("Empty response")
            return issues
        
        if len(response) < 20:
            issues.append("Response too short")
        
        # Check for incomplete code blocks
        if response.count('```') % 2 != 0:
            issues.append("Unclosed code block")
        
        # Check for repeated content
        lines = response.split('\n')
        if len(lines) > 5:
            seen = set()
            for line in lines:
                if line.strip() and line.strip() in seen:
                    issues.append("Repeated content detected")
                    break
                seen.add(line.strip())
        
        return issues
    
    def _generate_suggestions(
        self, 
        response: str, 
        content_type: ContentType,
        issues: List[str]
    ) -> List[str]:
        """Generate improvement suggestions."""
        suggestions = []
        
        if "Empty response" in issues:
            suggestions.append("Retry the request")
        
        if "Response too short" in issues:
            suggestions.append("Request more detail")
        
        if "Unclosed code block" in issues:
            suggestions.append("Close code blocks with ```")
        
        if content_type == ContentType.CODE and '```' not in response:
            suggestions.append("Use code blocks for better formatting")
        
        return suggestions


class ResponseOptimizer:
    """
    AI-Driven Response Optimizer
    
    Automatically enhances responses for:
    - Better structure and formatting
    - Code quality improvements
    - Readability enhancements
    - Completeness checks
    """
    
    def __init__(self):
        self.content_detector = ContentDetector()
        self.quality_analyzer = QualityAnalyzer()
    
    def optimize(self, response: str) -> OptimizationResult:
        """
        Optimize a response for better quality.
        
        Args:
            response: The response to optimize
            
        Returns:
            OptimizationResult with original and optimized response
        """
        if not response:
            return OptimizationResult(
                original=response,
                optimized=response,
                quality_before=QualityLevel.VERY_POOR,
                quality_after=QualityLevel.VERY_POOR,
                improvements=[],
                content_type=ContentType.TEXT
            )
        
        # Analyze original
        analysis = self.quality_analyzer.analyze(response)
        quality_before = self._score_to_level(analysis.quality_score)
        
        # Apply optimizations
        optimized = response
        improvements = []
        
        # 1. Fix code blocks
        optimized, fixed = self._fix_code_blocks(optimized)
        if fixed:
            improvements.append("Fixed code block formatting")
        
        # 2. Enhance markdown
        optimized, enhanced = self._enhance_markdown(optimized)
        if enhanced:
            improvements.append("Enhanced markdown structure")
        
        # 3. Fix whitespace
        optimized, cleaned = self._clean_whitespace(optimized)
        if cleaned:
            improvements.append("Cleaned whitespace")
        
        # 4. Add structure if missing
        optimized, structured = self._add_structure(optimized, analysis.content_type)
        if structured:
            improvements.append("Added structural elements")
        
        # Analyze optimized
        new_analysis = self.quality_analyzer.analyze(optimized)
        quality_after = self._score_to_level(new_analysis.quality_score)
        
        return OptimizationResult(
            original=response,
            optimized=optimized,
            quality_before=quality_before,
            quality_after=quality_after,
            improvements=improvements,
            content_type=analysis.content_type
        )
    
    def _score_to_level(self, score: float) -> QualityLevel:
        """Convert score to quality level."""
        if score >= 0.9:
            return QualityLevel.EXCELLENT
        elif score >= 0.7:
            return QualityLevel.GOOD
        elif score >= 0.5:
            return QualityLevel.AVERAGE
        elif score >= 0.3:
            return QualityLevel.POOR
        else:
            return QualityLevel.VERY_POOR
    
    def _fix_code_blocks(self, content: str) -> Tuple[str, bool]:
        """Fix improperly formatted code blocks."""
        if content.count('```') % 2 != 0:
            # Add closing block
            content += '\n```'
            return content, True
        return content, False
    
    def _enhance_markdown(self, content: str) -> Tuple[str, bool]:
        """Enhance markdown formatting."""
        enhanced = False
        
        # Ensure proper spacing around headers
        lines = content.split('\n')
        new_lines = []
        
        for i, line in enumerate(lines):
            if line.startswith('#') and i > 0 and lines[i-1].strip():
                new_lines.append('')
                enhanced = True
            new_lines.append(line)
        
        return '\n'.join(new_lines), enhanced
    
    def _clean_whitespace(self, content: str) -> Tuple[str, bool]:
        """Clean excessive whitespace."""
        original_len = len(content)
        
        # Remove trailing whitespace
        lines = [line.rstrip() for line in content.split('\n')]
        
        # Remove excessive blank lines (more than 2 in a row)
        cleaned = []
        blank_count = 0
        for line in lines:
            if not line:
                blank_count += 1
                if blank_count <= 2:
                    cleaned.append(line)
            else:
                blank_count = 0
                cleaned.append(line)
        
        result = '\n'.join(cleaned)
        return result, len(result) != original_len
    
    def _add_structure(self, content: str, content_type: ContentType) -> Tuple[str, bool]:
        """Add structural elements if missing."""
        # Don't modify already structured content
        if re.search(r'^#+\s+', content, re.MULTILINE):
            return content, False
        
        # Don't add structure to short responses
        if len(content) < 200:
            return content, False
        
        return content, False


class StreamingOptimizer:
    """Optimizes streaming responses in real-time."""
    
    def __init__(self):
        self.buffer = ""
        self.in_code_block = False
        self.code_language = None
    
    def process_chunk(self, chunk: str) -> str:
        """Process a streaming chunk."""
        self.buffer += chunk
        
        # Track code block state
        if '```' in chunk:
            if self.in_code_block:
                self.in_code_block = False
            else:
                self.in_code_block = True
                # Try to detect language
                match = re.search(r'```(\w+)', self.buffer)
                if match:
                    self.code_language = match.group(1)
        
        return chunk
    
    def finalize(self) -> str:
        """Finalize and return any buffered content."""
        result = ""
        
        # Close unclosed code blocks
        if self.in_code_block:
            result = "\n```"
            self.in_code_block = False
        
        self.buffer = ""
        self.code_language = None
        
        return result


# Singleton instance for easy access
_optimizer_instance: Optional[ResponseOptimizer] = None


def get_optimizer() -> ResponseOptimizer:
    """Get the global ResponseOptimizer instance."""
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = ResponseOptimizer()
    return _optimizer_instance


def optimize_response(response: str) -> str:
    """
    Quick function to optimize a response.
    
    Args:
        response: Response to optimize
        
    Returns:
        Optimized response
    """
    optimizer = get_optimizer()
    result = optimizer.optimize(response)
    return result.optimized


def analyze_response(response: str) -> Dict[str, Any]:
    """
    Quick function to analyze a response.
    
    Args:
        response: Response to analyze
        
    Returns:
        Analysis dictionary
    """
    analyzer = QualityAnalyzer()
    analysis = analyzer.analyze(response)
    
    return {
        "content_type": analysis.content_type.value,
        "quality_score": analysis.quality_score,
        "readability_score": analysis.readability_score,
        "structure_score": analysis.structure_score,
        "completeness_score": analysis.completeness_score,
        "issues": analysis.issues,
        "suggestions": analysis.suggestions
    }

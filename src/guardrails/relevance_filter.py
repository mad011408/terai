"""
Relevance filter for ensuring responses are on-topic.
"""

from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import re


@dataclass
class RelevanceScore:
    """Score indicating relevance of a response."""
    score: float  # 0-1
    is_relevant: bool
    matched_topics: List[str]
    off_topic_indicators: List[str]
    confidence: float
    explanation: str


class RelevanceFilter:
    """
    Filters responses for relevance to the conversation topic.
    """

    def __init__(self, relevance_threshold: float = 0.5):
        self.relevance_threshold = relevance_threshold
        self.topic_history: List[str] = []
        self.context_keywords: Set[str] = set()
        self._load_domain_keywords()

    def _load_domain_keywords(self) -> None:
        """Load domain-specific keywords."""
        self.domain_keywords = {
            "programming": {
                "code", "function", "class", "variable", "algorithm", "debug",
                "compile", "syntax", "library", "api", "framework", "database",
                "python", "javascript", "java", "sql", "html", "css"
            },
            "system": {
                "terminal", "command", "shell", "file", "directory", "process",
                "memory", "cpu", "disk", "network", "permission", "user"
            },
            "data": {
                "data", "analysis", "statistics", "chart", "graph", "model",
                "training", "prediction", "dataset", "feature", "accuracy"
            },
            "general": {
                "help", "explain", "how", "what", "why", "when", "where",
                "example", "show", "create", "make", "find", "search"
            }
        }

        self.off_topic_indicators = [
            "weather", "sports", "celebrity", "gossip", "politics",
            "dating", "personal", "feelings", "emotion", "opinion"
        ]

    def set_context(self, context_text: str) -> None:
        """Set conversation context for relevance checking."""
        # Extract keywords from context
        words = re.findall(r'\b[a-z]{3,}\b', context_text.lower())
        self.context_keywords = set(words)

        # Add to topic history
        self.topic_history.append(context_text[:200])
        if len(self.topic_history) > 10:
            self.topic_history.pop(0)

    def add_topic(self, topic: str) -> None:
        """Add a topic to track."""
        self.topic_history.append(topic)
        words = re.findall(r'\b[a-z]{3,}\b', topic.lower())
        self.context_keywords.update(words)

    def check_relevance(self, response: str, query: Optional[str] = None) -> RelevanceScore:
        """
        Check if a response is relevant to the conversation.

        Args:
            response: The response to check
            query: The original query (if available)

        Returns:
            RelevanceScore with relevance information
        """
        response_lower = response.lower()
        response_words = set(re.findall(r'\b[a-z]{3,}\b', response_lower))

        # Calculate domain match
        domain_matches = {}
        for domain, keywords in self.domain_keywords.items():
            match_count = len(response_words & keywords)
            if match_count > 0:
                domain_matches[domain] = match_count

        # Calculate context overlap
        context_overlap = len(response_words & self.context_keywords)

        # Check for off-topic indicators
        off_topic_found = [
            ind for ind in self.off_topic_indicators
            if ind in response_lower
        ]

        # Calculate relevance score
        score = 0.0

        # Domain relevance (0-0.4)
        if domain_matches:
            max_domain_match = max(domain_matches.values())
            score += min(0.4, max_domain_match * 0.1)

        # Context overlap (0-0.4)
        if self.context_keywords:
            overlap_ratio = context_overlap / len(self.context_keywords)
            score += min(0.4, overlap_ratio * 0.4)

        # Query relevance (0-0.2)
        if query:
            query_words = set(re.findall(r'\b[a-z]{3,}\b', query.lower()))
            query_overlap = len(response_words & query_words)
            score += min(0.2, query_overlap * 0.05)

        # Penalty for off-topic content
        if off_topic_found:
            score -= len(off_topic_found) * 0.1

        # Ensure score is in valid range
        score = max(0.0, min(1.0, score))

        is_relevant = score >= self.relevance_threshold

        # Calculate confidence
        confidence = 0.7  # Base confidence
        if domain_matches:
            confidence += 0.1
        if context_overlap > 3:
            confidence += 0.1
        if query:
            confidence += 0.1
        confidence = min(1.0, confidence)

        # Generate explanation
        explanation = self._generate_explanation(
            score, domain_matches, context_overlap, off_topic_found
        )

        return RelevanceScore(
            score=score,
            is_relevant=is_relevant,
            matched_topics=list(domain_matches.keys()),
            off_topic_indicators=off_topic_found,
            confidence=confidence,
            explanation=explanation
        )

    def _generate_explanation(self, score: float, domain_matches: Dict,
                             context_overlap: int,
                             off_topic: List[str]) -> str:
        """Generate explanation for relevance score."""
        parts = []

        if score >= 0.7:
            parts.append("Response is highly relevant.")
        elif score >= 0.5:
            parts.append("Response is moderately relevant.")
        elif score >= 0.3:
            parts.append("Response has some relevance.")
        else:
            parts.append("Response appears off-topic.")

        if domain_matches:
            domains = ", ".join(domain_matches.keys())
            parts.append(f"Matches domains: {domains}.")

        if context_overlap > 0:
            parts.append(f"Shares {context_overlap} keywords with context.")

        if off_topic:
            parts.append(f"Off-topic indicators found: {', '.join(off_topic)}.")

        return " ".join(parts)

    def filter_response(self, response: str, query: Optional[str] = None,
                       fallback_message: Optional[str] = None) -> tuple[str, RelevanceScore]:
        """
        Filter response for relevance.

        Args:
            response: Response to filter
            query: Original query
            fallback_message: Message to return if irrelevant

        Returns:
            Tuple of (filtered_response, relevance_score)
        """
        score = self.check_relevance(response, query)

        if score.is_relevant:
            return response, score

        # Return fallback or modified response
        if fallback_message:
            return fallback_message, score

        # Try to extract relevant parts
        relevant_parts = self._extract_relevant_parts(response)
        if relevant_parts:
            return relevant_parts, score

        return (
            "I apologize, but my response seems off-topic. "
            "Could you please rephrase your question?",
            score
        )

    def _extract_relevant_parts(self, response: str) -> Optional[str]:
        """Try to extract relevant parts from a response."""
        # Split into paragraphs
        paragraphs = response.split('\n\n')

        relevant_paragraphs = []
        for para in paragraphs:
            para_score = self.check_relevance(para)
            if para_score.score >= self.relevance_threshold:
                relevant_paragraphs.append(para)

        if relevant_paragraphs:
            return '\n\n'.join(relevant_paragraphs)

        return None

    def get_topic_summary(self) -> Dict[str, Any]:
        """Get summary of tracked topics."""
        return {
            "topic_count": len(self.topic_history),
            "keyword_count": len(self.context_keywords),
            "recent_topics": self.topic_history[-5:],
            "top_keywords": list(self.context_keywords)[:20],
            "relevance_threshold": self.relevance_threshold
        }

    def reset(self) -> None:
        """Reset topic tracking."""
        self.topic_history.clear()
        self.context_keywords.clear()


class ConversationRelevanceTracker:
    """
    Tracks relevance across a conversation.
    """

    def __init__(self):
        self.filter = RelevanceFilter()
        self.turn_scores: List[RelevanceScore] = []
        self.conversation_topic: Optional[str] = None

    def set_topic(self, topic: str) -> None:
        """Set the main conversation topic."""
        self.conversation_topic = topic
        self.filter.add_topic(topic)

    def track_turn(self, query: str, response: str) -> RelevanceScore:
        """Track a conversation turn."""
        # Update context with query
        self.filter.set_context(query)

        # Check response relevance
        score = self.filter.check_relevance(response, query)
        self.turn_scores.append(score)

        # Update context with response if relevant
        if score.is_relevant:
            self.filter.set_context(response)

        return score

    def get_conversation_relevance(self) -> Dict[str, Any]:
        """Get overall conversation relevance metrics."""
        if not self.turn_scores:
            return {"average_score": 0, "relevant_turns": 0, "total_turns": 0}

        scores = [s.score for s in self.turn_scores]
        relevant = sum(1 for s in self.turn_scores if s.is_relevant)

        return {
            "average_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "relevant_turns": relevant,
            "total_turns": len(self.turn_scores),
            "relevance_rate": relevant / len(self.turn_scores),
            "topic": self.conversation_topic
        }

    def is_conversation_on_track(self, min_relevance_rate: float = 0.7) -> bool:
        """Check if conversation is staying on track."""
        if not self.turn_scores:
            return True

        recent = self.turn_scores[-5:]  # Last 5 turns
        relevant = sum(1 for s in recent if s.is_relevant)
        return (relevant / len(recent)) >= min_relevance_rate

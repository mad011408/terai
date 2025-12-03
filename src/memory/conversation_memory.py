"""
Conversation memory for maintaining chat history.
Supports windowing, summarization, and retrieval.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import json
import hashlib


@dataclass
class MemoryMessage:
    """A message in conversation memory."""
    message_id: str
    role: str  # user, assistant, system, tool
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "token_count": self.token_count
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryMessage":
        return cls(
            message_id=data["message_id"],
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
            metadata=data.get("metadata", {}),
            token_count=data.get("token_count", 0)
        )

    def to_api_format(self) -> Dict[str, str]:
        """Convert to API format (OpenAI/Anthropic compatible)."""
        return {
            "role": self.role,
            "content": self.content
        }


@dataclass
class ConversationSummary:
    """Summary of a conversation segment."""
    summary_id: str
    content: str
    message_range: Tuple[int, int]  # Start and end indices
    timestamp: datetime
    key_points: List[str] = field(default_factory=list)


class ConversationMemory:
    """
    Manages conversation history with windowing and summarization.
    """

    def __init__(self, max_messages: int = 100,
                 max_tokens: int = 4000,
                 summarization_threshold: int = 50):
        self.messages: List[MemoryMessage] = []
        self.summaries: List[ConversationSummary] = []
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.summarization_threshold = summarization_threshold
        self._total_token_count = 0
        self._message_counter = 0

    def add_message(self, role: str, content: str,
                   metadata: Optional[Dict] = None) -> MemoryMessage:
        """Add a message to memory."""
        self._message_counter += 1
        message_id = f"msg_{self._message_counter}_{self._generate_id(content)}"

        # Estimate token count (rough approximation)
        token_count = len(content.split()) + len(content) // 4

        message = MemoryMessage(
            message_id=message_id,
            role=role,
            content=content,
            metadata=metadata or {},
            token_count=token_count
        )

        self.messages.append(message)
        self._total_token_count += token_count

        # Check if we need to trim
        self._check_and_trim()

        return message

    def add_user_message(self, content: str, **metadata) -> MemoryMessage:
        """Add a user message."""
        return self.add_message("user", content, metadata)

    def add_assistant_message(self, content: str, **metadata) -> MemoryMessage:
        """Add an assistant message."""
        return self.add_message("assistant", content, metadata)

    def add_system_message(self, content: str, **metadata) -> MemoryMessage:
        """Add a system message."""
        return self.add_message("system", content, metadata)

    def add_tool_message(self, content: str, tool_name: str, **metadata) -> MemoryMessage:
        """Add a tool result message."""
        metadata["tool_name"] = tool_name
        return self.add_message("tool", content, metadata)

    def _check_and_trim(self) -> None:
        """Check limits and trim if necessary."""
        # Trim by message count
        if len(self.messages) > self.max_messages:
            # Summarize old messages before removing
            if len(self.messages) >= self.summarization_threshold:
                self._summarize_old_messages()
            else:
                # Just remove oldest
                removed = self.messages[:len(self.messages) - self.max_messages]
                for msg in removed:
                    self._total_token_count -= msg.token_count
                self.messages = self.messages[-self.max_messages:]

    def _summarize_old_messages(self) -> None:
        """Summarize old messages to save space."""
        # Take first half of messages
        cutoff = len(self.messages) // 2
        to_summarize = self.messages[:cutoff]

        # Create summary
        summary_content = self._create_summary_text(to_summarize)
        summary = ConversationSummary(
            summary_id=f"summary_{len(self.summaries) + 1}",
            content=summary_content,
            message_range=(0, cutoff),
            timestamp=datetime.now(),
            key_points=self._extract_key_points(to_summarize)
        )
        self.summaries.append(summary)

        # Remove summarized messages
        for msg in to_summarize:
            self._total_token_count -= msg.token_count
        self.messages = self.messages[cutoff:]

    def _create_summary_text(self, messages: List[MemoryMessage]) -> str:
        """Create summary text from messages."""
        # Simple summary - in production, use LLM
        parts = []
        parts.append(f"Conversation summary ({len(messages)} messages):")

        for msg in messages[:5]:  # First 5 messages
            parts.append(f"- {msg.role}: {msg.content[:100]}...")

        if len(messages) > 5:
            parts.append(f"... and {len(messages) - 5} more messages")

        return "\n".join(parts)

    def _extract_key_points(self, messages: List[MemoryMessage]) -> List[str]:
        """Extract key points from messages."""
        key_points = []

        for msg in messages:
            if msg.role == "user":
                # Extract questions or requests
                if "?" in msg.content:
                    key_points.append(f"User asked: {msg.content[:50]}...")
                elif any(kw in msg.content.lower() for kw in ["please", "can you", "help"]):
                    key_points.append(f"User requested: {msg.content[:50]}...")

        return key_points[:5]  # Limit to 5 key points

    def get_messages(self, limit: Optional[int] = None,
                    include_system: bool = True) -> List[MemoryMessage]:
        """Get messages from memory."""
        messages = self.messages

        if not include_system:
            messages = [m for m in messages if m.role != "system"]

        if limit:
            messages = messages[-limit:]

        return messages

    def get_context_window(self, max_tokens: Optional[int] = None) -> List[MemoryMessage]:
        """Get messages that fit within token limit."""
        max_tokens = max_tokens or self.max_tokens
        result = []
        current_tokens = 0

        # Iterate from newest to oldest
        for message in reversed(self.messages):
            if current_tokens + message.token_count > max_tokens:
                break
            result.insert(0, message)
            current_tokens += message.token_count

        return result

    def get_api_messages(self, max_tokens: Optional[int] = None) -> List[Dict[str, str]]:
        """Get messages in API format."""
        messages = self.get_context_window(max_tokens)
        return [msg.to_api_format() for msg in messages]

    def get_last_message(self, role: Optional[str] = None) -> Optional[MemoryMessage]:
        """Get the last message, optionally filtered by role."""
        if role:
            for msg in reversed(self.messages):
                if msg.role == role:
                    return msg
            return None
        return self.messages[-1] if self.messages else None

    def get_message_by_id(self, message_id: str) -> Optional[MemoryMessage]:
        """Get message by ID."""
        for msg in self.messages:
            if msg.message_id == message_id:
                return msg
        return None

    def search_messages(self, query: str, limit: int = 5) -> List[MemoryMessage]:
        """Search messages by content."""
        query_lower = query.lower()
        results = []

        for msg in reversed(self.messages):
            if query_lower in msg.content.lower():
                results.append(msg)
                if len(results) >= limit:
                    break

        return results

    def get_turn_count(self) -> int:
        """Get number of conversation turns."""
        return sum(1 for m in self.messages if m.role == "user")

    def clear(self) -> None:
        """Clear all messages."""
        self.messages.clear()
        self.summaries.clear()
        self._total_token_count = 0

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of the conversation."""
        return {
            "total_messages": len(self.messages),
            "turn_count": self.get_turn_count(),
            "total_tokens": self._total_token_count,
            "summaries_count": len(self.summaries),
            "roles": {
                role: sum(1 for m in self.messages if m.role == role)
                for role in set(m.role for m in self.messages)
            },
            "first_message": self.messages[0].timestamp.isoformat() if self.messages else None,
            "last_message": self.messages[-1].timestamp.isoformat() if self.messages else None
        }

    def export_conversation(self) -> Dict[str, Any]:
        """Export entire conversation."""
        return {
            "messages": [m.to_dict() for m in self.messages],
            "summaries": [
                {
                    "summary_id": s.summary_id,
                    "content": s.content,
                    "message_range": s.message_range,
                    "timestamp": s.timestamp.isoformat(),
                    "key_points": s.key_points
                }
                for s in self.summaries
            ],
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "total_messages": len(self.messages),
                "total_tokens": self._total_token_count
            }
        }

    def import_conversation(self, data: Dict[str, Any]) -> None:
        """Import conversation from exported data."""
        self.clear()

        for msg_data in data.get("messages", []):
            msg = MemoryMessage.from_dict(msg_data)
            self.messages.append(msg)
            self._total_token_count += msg.token_count

        for summary_data in data.get("summaries", []):
            summary = ConversationSummary(
                summary_id=summary_data["summary_id"],
                content=summary_data["content"],
                message_range=tuple(summary_data["message_range"]),
                timestamp=datetime.fromisoformat(summary_data["timestamp"]),
                key_points=summary_data.get("key_points", [])
            )
            self.summaries.append(summary)

    def _generate_id(self, content: str) -> str:
        """Generate short ID from content."""
        return hashlib.md5(content.encode()).hexdigest()[:8]


class SlidingWindowMemory(ConversationMemory):
    """
    Sliding window memory that maintains a fixed window of recent messages.
    """

    def __init__(self, window_size: int = 20):
        super().__init__(max_messages=window_size)
        self.window_size = window_size

    def _check_and_trim(self) -> None:
        """Trim to window size."""
        while len(self.messages) > self.window_size:
            removed = self.messages.pop(0)
            self._total_token_count -= removed.token_count


class TokenLimitedMemory(ConversationMemory):
    """
    Memory that limits by token count rather than message count.
    """

    def __init__(self, max_tokens: int = 4000):
        super().__init__(max_tokens=max_tokens, max_messages=10000)

    def _check_and_trim(self) -> None:
        """Trim by token count."""
        while self._total_token_count > self.max_tokens and self.messages:
            removed = self.messages.pop(0)
            self._total_token_count -= removed.token_count

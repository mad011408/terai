"""
Message types and formatting for agent communication.
Handles message serialization, validation, and display.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from datetime import datetime
import json
import uuid


class MessageType(Enum):
    """Types of messages in the system."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    HANDOFF = "handoff"


class MessageRole(Enum):
    """Roles for message senders."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class ToolCall:
    """Represents a tool call within a message."""
    tool_name: str
    tool_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    arguments: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    result: Any = None


@dataclass
class Message:
    """
    Represents a message in the agent conversation.
    Supports various content types and metadata.
    """
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType = MessageType.USER
    role: MessageRole = MessageRole.USER
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    parent_id: Optional[str] = None
    agent_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "role": self.role.value,
            "content": self.content,
            "tool_calls": [
                {
                    "tool_id": tc.tool_id,
                    "tool_name": tc.tool_name,
                    "arguments": tc.arguments,
                    "status": tc.status,
                    "result": tc.result
                }
                for tc in self.tool_calls
            ],
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "parent_id": self.parent_id,
            "agent_name": self.agent_name
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary."""
        tool_calls = [
            ToolCall(
                tool_id=tc.get("tool_id", str(uuid.uuid4())),
                tool_name=tc["tool_name"],
                arguments=tc.get("arguments", {}),
                status=tc.get("status", "pending"),
                result=tc.get("result")
            )
            for tc in data.get("tool_calls", [])
        ]

        return cls(
            message_id=data.get("message_id", str(uuid.uuid4())),
            message_type=MessageType(data.get("message_type", "user")),
            role=MessageRole(data.get("role", "user")),
            content=data.get("content", ""),
            tool_calls=tool_calls,
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
            parent_id=data.get("parent_id"),
            agent_name=data.get("agent_name")
        )

    def to_json(self) -> str:
        """Convert message to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "Message":
        """Create message from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def add_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> ToolCall:
        """Add a tool call to the message."""
        tool_call = ToolCall(tool_name=tool_name, arguments=arguments)
        self.tool_calls.append(tool_call)
        return tool_call

    def get_tool_call(self, tool_id: str) -> Optional[ToolCall]:
        """Get a tool call by ID."""
        for tc in self.tool_calls:
            if tc.tool_id == tool_id:
                return tc
        return None

    def update_tool_result(self, tool_id: str, result: Any, status: str = "completed") -> None:
        """Update the result of a tool call."""
        tool_call = self.get_tool_call(tool_id)
        if tool_call:
            tool_call.result = result
            tool_call.status = status


class MessageBuilder:
    """
    Builder pattern for constructing messages.
    """

    def __init__(self):
        self._message = Message()

    def with_type(self, message_type: MessageType) -> "MessageBuilder":
        self._message.message_type = message_type
        return self

    def with_role(self, role: MessageRole) -> "MessageBuilder":
        self._message.role = role
        return self

    def with_content(self, content: str) -> "MessageBuilder":
        self._message.content = content
        return self

    def with_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> "MessageBuilder":
        self._message.add_tool_call(tool_name, arguments)
        return self

    def with_metadata(self, key: str, value: Any) -> "MessageBuilder":
        self._message.metadata[key] = value
        return self

    def with_parent(self, parent_id: str) -> "MessageBuilder":
        self._message.parent_id = parent_id
        return self

    def with_agent(self, agent_name: str) -> "MessageBuilder":
        self._message.agent_name = agent_name
        return self

    def build(self) -> Message:
        return self._message


class MessageFormatter:
    """
    Formats messages for different output contexts.
    """

    def __init__(self, include_timestamps: bool = True, include_metadata: bool = False):
        self.include_timestamps = include_timestamps
        self.include_metadata = include_metadata

    def format_for_display(self, message: Message) -> str:
        """Format message for terminal display."""
        parts = []

        # Role prefix
        role_prefix = {
            MessageRole.USER: "👤 User",
            MessageRole.ASSISTANT: "🤖 Assistant",
            MessageRole.SYSTEM: "⚙️ System",
            MessageRole.TOOL: "🔧 Tool"
        }.get(message.role, "Unknown")

        if message.agent_name:
            role_prefix = f"{role_prefix} ({message.agent_name})"

        if self.include_timestamps:
            parts.append(f"[{message.timestamp.strftime('%H:%M:%S')}] {role_prefix}:")
        else:
            parts.append(f"{role_prefix}:")

        parts.append(message.content)

        # Tool calls
        if message.tool_calls:
            parts.append("\n📋 Tool Calls:")
            for tc in message.tool_calls:
                parts.append(f"  • {tc.tool_name}({json.dumps(tc.arguments, indent=2)})")
                if tc.result:
                    parts.append(f"    Result: {tc.result}")

        if self.include_metadata and message.metadata:
            parts.append(f"\n📊 Metadata: {json.dumps(message.metadata, indent=2)}")

        return "\n".join(parts)

    def format_for_api(self, message: Message) -> Dict[str, Any]:
        """Format message for API consumption (OpenAI/Anthropic compatible)."""
        api_message = {
            "role": message.role.value,
            "content": message.content
        }

        if message.tool_calls:
            api_message["tool_calls"] = [
                {
                    "id": tc.tool_id,
                    "type": "function",
                    "function": {
                        "name": tc.tool_name,
                        "arguments": json.dumps(tc.arguments)
                    }
                }
                for tc in message.tool_calls
            ]

        return api_message

    def format_for_logging(self, message: Message) -> str:
        """Format message for logging."""
        return json.dumps({
            "message_id": message.message_id,
            "type": message.message_type.value,
            "role": message.role.value,
            "content_preview": message.content[:100] + "..." if len(message.content) > 100 else message.content,
            "tool_calls_count": len(message.tool_calls),
            "timestamp": message.timestamp.isoformat()
        })

    def format_conversation(self, messages: List[Message]) -> str:
        """Format a list of messages as a conversation."""
        return "\n\n".join(self.format_for_display(msg) for msg in messages)


class MessageHistory:
    """
    Manages message history with windowing and summarization support.
    """

    def __init__(self, max_messages: int = 100):
        self.messages: List[Message] = []
        self.max_messages = max_messages
        self._summaries: List[str] = []

    def add(self, message: Message) -> None:
        """Add a message to history."""
        self.messages.append(message)

        # Trim if exceeds max
        if len(self.messages) > self.max_messages:
            self._summarize_old_messages()

    def get_recent(self, count: int = 10) -> List[Message]:
        """Get recent messages."""
        return self.messages[-count:]

    def get_by_type(self, message_type: MessageType) -> List[Message]:
        """Get messages by type."""
        return [m for m in self.messages if m.message_type == message_type]

    def get_by_role(self, role: MessageRole) -> List[Message]:
        """Get messages by role."""
        return [m for m in self.messages if m.role == role]

    def get_window(self, start: int, end: int) -> List[Message]:
        """Get a window of messages."""
        return self.messages[start:end]

    def search(self, query: str) -> List[Message]:
        """Search messages by content."""
        query_lower = query.lower()
        return [m for m in self.messages if query_lower in m.content.lower()]

    def clear(self) -> None:
        """Clear all messages."""
        self.messages = []
        self._summaries = []

    def _summarize_old_messages(self) -> None:
        """Summarize and remove old messages."""
        # Keep last max_messages/2, summarize the rest
        cutoff = self.max_messages // 2
        old_messages = self.messages[:cutoff]

        # Create simple summary
        summary = f"[Summary of {len(old_messages)} messages from {old_messages[0].timestamp} to {old_messages[-1].timestamp}]"
        self._summaries.append(summary)

        # Remove old messages
        self.messages = self.messages[cutoff:]

    def get_context_window(self, token_limit: int = 4000) -> List[Message]:
        """Get messages that fit within a token limit (approximate)."""
        result = []
        total_chars = 0
        char_limit = token_limit * 4  # Rough approximation

        for message in reversed(self.messages):
            msg_chars = len(message.content)
            if total_chars + msg_chars > char_limit:
                break
            result.insert(0, message)
            total_chars += msg_chars

        return result

    def to_list(self) -> List[Dict[str, Any]]:
        """Convert history to list of dictionaries."""
        return [m.to_dict() for m in self.messages]


# Convenience functions for creating messages
def user_message(content: str, **kwargs) -> Message:
    """Create a user message."""
    return Message(
        message_type=MessageType.USER,
        role=MessageRole.USER,
        content=content,
        **kwargs
    )


def assistant_message(content: str, agent_name: Optional[str] = None, **kwargs) -> Message:
    """Create an assistant message."""
    return Message(
        message_type=MessageType.ASSISTANT,
        role=MessageRole.ASSISTANT,
        content=content,
        agent_name=agent_name,
        **kwargs
    )


def system_message(content: str, **kwargs) -> Message:
    """Create a system message."""
    return Message(
        message_type=MessageType.SYSTEM,
        role=MessageRole.SYSTEM,
        content=content,
        **kwargs
    )


def tool_message(tool_name: str, result: Any, tool_id: Optional[str] = None, **kwargs) -> Message:
    """Create a tool result message."""
    msg = Message(
        message_type=MessageType.TOOL_RESULT,
        role=MessageRole.TOOL,
        content=str(result),
        **kwargs
    )
    if tool_id:
        msg.metadata["tool_id"] = tool_id
    msg.metadata["tool_name"] = tool_name
    return msg


def thought_message(thought: str, agent_name: str, **kwargs) -> Message:
    """Create a thought message (internal reasoning)."""
    return Message(
        message_type=MessageType.THOUGHT,
        role=MessageRole.ASSISTANT,
        content=thought,
        agent_name=agent_name,
        **kwargs
    )

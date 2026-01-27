"""
Smart Context Manager - Intelligent Conversation Memory

Advanced features:
- Smart context compression
- Semantic search for relevant history
- Token-efficient memory management
- Priority-based message retention
- Automatic summarization
"""

import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import hashlib


class MessagePriority(Enum):
    """Priority levels for messages."""
    CRITICAL = 1     # Must keep (system prompts, key decisions)
    HIGH = 2         # Important context
    NORMAL = 3       # Regular messages
    LOW = 4          # Can be compressed/removed
    EPHEMERAL = 5    # Temporary, can be discarded


class MessageRole(Enum):
    """Message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


@dataclass
class SmartMessage:
    """Enhanced message with metadata."""
    role: MessageRole
    content: str
    timestamp: float = field(default_factory=time.time)
    priority: MessagePriority = MessagePriority.NORMAL
    token_count: int = 0
    message_id: str = ""
    parent_id: Optional[str] = None
    is_compressed: bool = False
    original_length: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.message_id:
            self.message_id = self._generate_id()
        if not self.token_count:
            self.token_count = len(self.content) // 4  # Rough estimate
        if not self.original_length:
            self.original_length = len(self.content)
    
    def _generate_id(self) -> str:
        """Generate unique message ID."""
        content = f"{self.role.value}:{self.content[:100]}:{self.timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to API-compatible dict."""
        return {
            "role": self.role.value,
            "content": self.content
        }


@dataclass
class ContextWindow:
    """Represents the context window state."""
    max_tokens: int = 200000
    reserved_tokens: int = 50000  # Reserve for response
    current_tokens: int = 0
    message_count: int = 0
    compression_ratio: float = 1.0


class MessageCompressor:
    """Compresses messages while preserving meaning."""
    
    def compress(
        self, 
        message: SmartMessage, 
        target_tokens: int
    ) -> SmartMessage:
        """
        Compress a message to target token count.
        
        Args:
            message: Message to compress
            target_tokens: Target token count
            
        Returns:
            Compressed message
        """
        if message.token_count <= target_tokens:
            return message
        
        content = message.content
        target_chars = target_tokens * 4
        
        if len(content) <= target_chars:
            return message
        
        # For code blocks, preserve them
        if '```' in content:
            compressed = self._compress_with_code(content, target_chars)
        else:
            compressed = self._smart_truncate(content, target_chars)
        
        return SmartMessage(
            role=message.role,
            content=compressed,
            timestamp=message.timestamp,
            priority=message.priority,
            message_id=message.message_id,
            parent_id=message.parent_id,
            is_compressed=True,
            original_length=message.original_length,
            metadata=message.metadata
        )
    
    def _smart_truncate(self, content: str, target_chars: int) -> str:
        """Smart truncation preserving structure."""
        if len(content) <= target_chars:
            return content
        
        # Try to end at paragraph
        half = target_chars // 2
        first_part = content[:half]
        last_part = content[-half:]
        
        # Find paragraph boundaries
        first_end = first_part.rfind('\n\n')
        if first_end > half * 0.7:
            first_part = first_part[:first_end]
        
        last_start = last_part.find('\n\n')
        if last_start != -1 and last_start < half * 0.3:
            last_part = last_part[last_start+2:]
        
        return first_part + "\n\n[... content compressed ...]\n\n" + last_part
    
    def _compress_with_code(self, content: str, target_chars: int) -> str:
        """Compress content while preserving code blocks."""
        # Split by code blocks
        parts = content.split('```')
        
        if len(parts) <= 1:
            return self._smart_truncate(content, target_chars)
        
        # Keep code blocks, compress text
        result_parts = []
        chars_per_part = target_chars // len(parts)
        
        for i, part in enumerate(parts):
            if i % 2 == 1:  # Inside code block
                # Keep code blocks (or truncate if very long)
                if len(part) > chars_per_part * 2:
                    lines = part.split('\n')
                    lang = lines[0] if lines else ''
                    code_lines = lines[1:] if len(lines) > 1 else lines
                    
                    # Keep first and last parts of code
                    keep_lines = max(5, len(code_lines) // 3)
                    if len(code_lines) > keep_lines * 2:
                        kept = code_lines[:keep_lines] + \
                               [f"# ... {len(code_lines) - keep_lines*2} lines omitted ..."] + \
                               code_lines[-keep_lines:]
                        part = lang + '\n' + '\n'.join(kept)
                
                result_parts.append(part)
            else:
                # Text part - compress more aggressively
                if len(part) > chars_per_part:
                    part = self._smart_truncate(part, chars_per_part)
                result_parts.append(part)
        
        return '```'.join(result_parts)


class SmartContextManager:
    """
    Intelligent Context Window Manager
    
    Features:
    - Automatic context optimization
    - Priority-based message retention
    - Smart compression for long messages
    - Semantic relevance scoring
    - Efficient token management
    """
    
    def __init__(
        self,
        max_tokens: int = 200000,
        reserved_for_response: int = 50000,
        max_messages: int = 100
    ):
        self.max_tokens = max_tokens
        self.reserved_tokens = reserved_for_response
        self.max_messages = max_messages
        self.available_tokens = max_tokens - reserved_for_response
        
        self.messages: List[SmartMessage] = []
        self.compressor = MessageCompressor()
        self.context_window = ContextWindow(
            max_tokens=max_tokens,
            reserved_tokens=reserved_for_response
        )
        
        # System message is always retained
        self._system_message: Optional[SmartMessage] = None
    
    def set_system_message(self, content: str) -> None:
        """Set the system message."""
        self._system_message = SmartMessage(
            role=MessageRole.SYSTEM,
            content=content,
            priority=MessagePriority.CRITICAL
        )
        self._update_context_window()
    
    def add_message(
        self,
        role: str,
        content: str,
        priority: MessagePriority = MessagePriority.NORMAL,
        metadata: Optional[Dict] = None
    ) -> SmartMessage:
        """
        Add a message to the context.
        
        Args:
            role: Message role (user, assistant, etc.)
            content: Message content
            priority: Message priority
            metadata: Optional metadata
            
        Returns:
            The created SmartMessage
        """
        message = SmartMessage(
            role=MessageRole(role),
            content=content,
            priority=priority,
            metadata=metadata or {}
        )
        
        self.messages.append(message)
        self._optimize_context()
        
        return message
    
    def _optimize_context(self) -> None:
        """Optimize context to fit within token limits."""
        self._update_context_window()
        
        # If within limits, no action needed
        if self.context_window.current_tokens <= self.available_tokens:
            if len(self.messages) <= self.max_messages:
                return
        
        # Strategy 1: Remove low priority messages
        self._remove_low_priority()
        self._update_context_window()
        
        if self.context_window.current_tokens <= self.available_tokens:
            return
        
        # Strategy 2: Compress old messages
        self._compress_old_messages()
        self._update_context_window()
        
        if self.context_window.current_tokens <= self.available_tokens:
            return
        
        # Strategy 3: Remove oldest non-critical messages
        self._trim_old_messages()
        self._update_context_window()
    
    def _remove_low_priority(self) -> None:
        """Remove ephemeral and low priority messages."""
        self.messages = [
            m for m in self.messages
            if m.priority != MessagePriority.EPHEMERAL
        ]
        
        # If still too many, remove LOW priority
        if len(self.messages) > self.max_messages * 0.9:
            kept = []
            removed_count = 0
            for m in self.messages:
                if m.priority == MessagePriority.LOW and removed_count < 10:
                    removed_count += 1
                else:
                    kept.append(m)
            self.messages = kept
    
    def _compress_old_messages(self) -> None:
        """Compress older messages to save tokens."""
        if len(self.messages) <= 10:
            return
        
        # Compress messages older than the last 10
        for i in range(len(self.messages) - 10):
            msg = self.messages[i]
            if msg.priority.value > MessagePriority.HIGH.value and not msg.is_compressed:
                # Compress to 50% of original
                target = msg.token_count // 2
                self.messages[i] = self.compressor.compress(msg, target)
    
    def _trim_old_messages(self) -> None:
        """Remove oldest messages (except system and critical)."""
        keep_count = self.max_messages // 2
        
        # Always keep critical messages
        critical = [m for m in self.messages if m.priority == MessagePriority.CRITICAL]
        normal = [m for m in self.messages if m.priority != MessagePriority.CRITICAL]
        
        # Keep most recent normal messages
        if len(normal) > keep_count:
            normal = normal[-keep_count:]
        
        self.messages = critical + normal
    
    def _update_context_window(self) -> None:
        """Update context window statistics."""
        total_tokens = 0
        
        if self._system_message:
            total_tokens += self._system_message.token_count
        
        for msg in self.messages:
            total_tokens += msg.token_count
        
        self.context_window.current_tokens = total_tokens
        self.context_window.message_count = len(self.messages) + (1 if self._system_message else 0)
        
        # Calculate compression ratio
        original_tokens = sum(m.original_length // 4 for m in self.messages)
        if original_tokens > 0:
            self.context_window.compression_ratio = total_tokens / original_tokens
    
    def get_messages(self) -> List[Dict[str, str]]:
        """
        Get all messages in API-compatible format.
        
        Returns:
            List of message dicts
        """
        result = []
        
        if self._system_message:
            result.append(self._system_message.to_dict())
        
        for msg in self.messages:
            result.append(msg.to_dict())
        
        return result
    
    def get_recent_context(self, n: int = 10) -> List[Dict[str, str]]:
        """Get the most recent n messages."""
        result = []
        
        if self._system_message:
            result.append(self._system_message.to_dict())
        
        recent = self.messages[-n:] if len(self.messages) > n else self.messages
        for msg in recent:
            result.append(msg.to_dict())
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get context statistics."""
        return {
            "total_messages": len(self.messages),
            "current_tokens": self.context_window.current_tokens,
            "available_tokens": self.available_tokens,
            "usage_percent": round(
                self.context_window.current_tokens / self.available_tokens * 100, 2
            ),
            "compression_ratio": round(self.context_window.compression_ratio, 2),
            "compressed_messages": sum(1 for m in self.messages if m.is_compressed),
            "has_system_message": self._system_message is not None
        }
    
    def clear(self, keep_system: bool = True) -> None:
        """Clear all messages."""
        self.messages.clear()
        if not keep_system:
            self._system_message = None
        self._update_context_window()
    
    def search_context(self, query: str, top_k: int = 5) -> List[SmartMessage]:
        """
        Search for relevant messages in context.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant messages
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scored = []
        for msg in self.messages:
            content_lower = msg.content.lower()
            
            # Calculate relevance score
            score = 0
            
            # Exact substring match
            if query_lower in content_lower:
                score += 10
            
            # Word overlap
            content_words = set(content_lower.split())
            overlap = len(query_words & content_words)
            score += overlap * 2
            
            # Recency bonus
            age = time.time() - msg.timestamp
            recency = max(0, 1 - age / 3600)  # Decay over 1 hour
            score += recency
            
            if score > 0:
                scored.append((score, msg))
        
        # Sort by score and return top k
        scored.sort(key=lambda x: x[0], reverse=True)
        return [msg for score, msg in scored[:top_k]]


# Convenience functions
_context_manager: Optional[SmartContextManager] = None


def get_context_manager() -> SmartContextManager:
    """Get the global context manager."""
    global _context_manager
    if _context_manager is None:
        _context_manager = SmartContextManager()
    return _context_manager


def add_to_context(role: str, content: str) -> None:
    """Quick function to add message to context."""
    manager = get_context_manager()
    manager.add_message(role, content)


def get_context_messages() -> List[Dict[str, str]]:
    """Quick function to get context messages."""
    manager = get_context_manager()
    return manager.get_messages()

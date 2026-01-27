"""
Context management and state tracking.
Handles conversation state, variables, and execution context.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union
from datetime import datetime
import copy
import json
import uuid
from threading import Lock


@dataclass
class ContextVariable:
    """A variable stored in context with metadata."""
    key: str
    value: Any
    var_type: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    tags: Set[str] = field(default_factory=set)


class Context:
    """
    Manages execution context and state for agents.
    Thread-safe implementation with history tracking.
    """

    def __init__(self, context_id: Optional[str] = None, parent: Optional["Context"] = None):
        self.context_id = context_id or str(uuid.uuid4())
        self.parent = parent
        self._store: Dict[str, ContextVariable] = {}
        self._history: List[Dict[str, Any]] = []
        self._lock = Lock()
        self._created_at = datetime.now()
        self._metadata: Dict[str, Any] = {}

    def set(self, key: str, value: Any, tags: Optional[Set[str]] = None) -> None:
        """Set a value in the context."""
        with self._lock:
            var_type = type(value).__name__

            if key in self._store:
                var = self._store[key]
                old_value = var.value
                var.value = value
                var.updated_at = datetime.now()
                var.var_type = var_type
                if tags:
                    var.tags.update(tags)
                self._record_history("update", key, old_value, value)
            else:
                self._store[key] = ContextVariable(
                    key=key,
                    value=value,
                    var_type=var_type,
                    tags=tags or set()
                )
                self._record_history("create", key, None, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from context, checking parent if not found."""
        with self._lock:
            if key in self._store:
                self._store[key].access_count += 1
                return self._store[key].value

            # Check parent context
            if self.parent:
                return self.parent.get(key, default)

            return default

    def delete(self, key: str) -> bool:
        """Delete a key from context."""
        with self._lock:
            if key in self._store:
                old_value = self._store[key].value
                del self._store[key]
                self._record_history("delete", key, old_value, None)
                return True
            return False

    def has(self, key: str) -> bool:
        """Check if key exists in context."""
        if key in self._store:
            return True
        if self.parent:
            return self.parent.has(key)
        return False

    def keys(self) -> List[str]:
        """Get all keys in context."""
        local_keys = list(self._store.keys())
        if self.parent:
            parent_keys = self.parent.keys()
            return list(set(local_keys + parent_keys))
        return local_keys

    def values(self) -> List[Any]:
        """Get all values in context."""
        return [var.value for var in self._store.values()]

    def items(self) -> List[tuple]:
        """Get all key-value pairs."""
        return [(k, v.value) for k, v in self._store.items()]

    def update(self, data: Dict[str, Any]) -> None:
        """Update context with multiple key-value pairs."""
        for key, value in data.items():
            self.set(key, value)

    def append_to_list(self, key: str, value: Any) -> None:
        """Append a value to a list in context."""
        with self._lock:
            if key not in self._store:
                self._store[key] = ContextVariable(
                    key=key,
                    value=[],
                    var_type="list"
                )

            if isinstance(self._store[key].value, list):
                self._store[key].value.append(value)
                self._store[key].updated_at = datetime.now()
                self._record_history("append", key, None, value)

    def get_by_tag(self, tag: str) -> Dict[str, Any]:
        """Get all variables with a specific tag."""
        return {
            key: var.value
            for key, var in self._store.items()
            if tag in var.tags
        }

    def clear(self) -> None:
        """Clear all context data."""
        with self._lock:
            self._store.clear()
            self._record_history("clear", None, None, None)

    def copy(self) -> "Context":
        """Create a deep copy of the context."""
        new_context = Context(parent=self.parent)
        with self._lock:
            for key, var in self._store.items():
                new_context._store[key] = ContextVariable(
                    key=var.key,
                    value=copy.deepcopy(var.value),
                    var_type=var.var_type,
                    created_at=var.created_at,
                    updated_at=var.updated_at,
                    access_count=var.access_count,
                    tags=var.tags.copy()
                )
        return new_context

    def create_child(self) -> "Context":
        """Create a child context that inherits from this one."""
        return Context(parent=self)

    def merge(self, other: "Context", overwrite: bool = True) -> None:
        """Merge another context into this one."""
        for key, var in other._store.items():
            if overwrite or key not in self._store:
                self.set(key, var.value, var.tags)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "context_id": self.context_id,
            "created_at": self._created_at.isoformat(),
            "data": {key: var.value for key, var in self._store.items()},
            "metadata": self._metadata
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load context from dictionary."""
        self.context_id = data.get("context_id", self.context_id)
        self._metadata = data.get("metadata", {})
        for key, value in data.get("data", {}).items():
            self.set(key, value)

    def to_json(self) -> str:
        """Convert context to JSON string."""
        def serialize(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, set):
                return list(obj)
            return str(obj)

        return json.dumps(self.to_dict(), default=serialize)

    def _record_history(self, action: str, key: Optional[str],
                        old_value: Any, new_value: Any) -> None:
        """Record a change in context history."""
        self._history.append({
            "action": action,
            "key": key,
            "old_value": old_value,
            "new_value": new_value,
            "timestamp": datetime.now().isoformat()
        })

    def get_history(self) -> List[Dict[str, Any]]:
        """Get context change history."""
        return self._history.copy()

    def get_stats(self) -> Dict[str, Any]:
        """Get context statistics."""
        return {
            "context_id": self.context_id,
            "variable_count": len(self._store),
            "history_length": len(self._history),
            "created_at": self._created_at.isoformat(),
            "has_parent": self.parent is not None,
            "most_accessed": max(
                self._store.items(),
                key=lambda x: x[1].access_count,
                default=(None, None)
            )[0]
        }

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata for the context."""
        self._metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value."""
        return self._metadata.get(key, default)

    def __contains__(self, key: str) -> bool:
        return self.has(key)

    def __getitem__(self, key: str) -> Any:
        value = self.get(key)
        if value is None and not self.has(key):
            raise KeyError(key)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value)

    def __delitem__(self, key: str) -> None:
        if not self.delete(key):
            raise KeyError(key)

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return f"Context(id={self.context_id}, vars={len(self._store)})"


class ContextManager:
    """
    Manages multiple contexts and their lifecycle.
    """

    def __init__(self):
        self.contexts: Dict[str, Context] = {}
        self._active_context: Optional[str] = None
        self._lock = Lock()

    def create_context(self, context_id: Optional[str] = None,
                       parent_id: Optional[str] = None) -> Context:
        """Create a new context."""
        parent = self.contexts.get(parent_id) if parent_id else None
        context = Context(context_id=context_id, parent=parent)

        with self._lock:
            self.contexts[context.context_id] = context
            if not self._active_context:
                self._active_context = context.context_id

        return context

    def get_context(self, context_id: str) -> Optional[Context]:
        """Get a context by ID."""
        return self.contexts.get(context_id)

    def get_active_context(self) -> Optional[Context]:
        """Get the currently active context."""
        if self._active_context:
            return self.contexts.get(self._active_context)
        return None

    def set_active_context(self, context_id: str) -> bool:
        """Set the active context."""
        if context_id in self.contexts:
            self._active_context = context_id
            return True
        return False

    def delete_context(self, context_id: str) -> bool:
        """Delete a context."""
        with self._lock:
            if context_id in self.contexts:
                del self.contexts[context_id]
                if self._active_context == context_id:
                    self._active_context = next(iter(self.contexts.keys()), None)
                return True
        return False

    def list_contexts(self) -> List[Dict[str, Any]]:
        """List all contexts with metadata."""
        return [
            {
                "context_id": ctx.context_id,
                "variables": len(ctx),
                "is_active": ctx.context_id == self._active_context
            }
            for ctx in self.contexts.values()
        ]

    def clear_all(self) -> None:
        """Clear all contexts."""
        with self._lock:
            self.contexts.clear()
            self._active_context = None


class ConversationContext(Context):
    """
    Specialized context for managing conversation state.
    """

    def __init__(self, context_id: Optional[str] = None):
        super().__init__(context_id)
        self.set("messages", [], tags={"conversation"})
        self.set("turn_count", 0, tags={"conversation"})

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Add a message to the conversation."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.append_to_list("messages", message)

        if role == "user":
            turn_count = self.get("turn_count", 0)
            self.set("turn_count", turn_count + 1)

    def get_messages(self, limit: Optional[int] = None) -> List[Dict]:
        """Get conversation messages."""
        messages = self.get("messages", [])
        if limit:
            return messages[-limit:]
        return messages

    def get_last_message(self, role: Optional[str] = None) -> Optional[Dict]:
        """Get the last message, optionally filtered by role."""
        messages = self.get("messages", [])
        if not messages:
            return None

        if role:
            for msg in reversed(messages):
                if msg["role"] == role:
                    return msg
            return None

        return messages[-1]

    def clear_messages(self) -> None:
        """Clear all messages."""
        self.set("messages", [])
        self.set("turn_count", 0)

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation."""
        messages = self.get("messages", [])
        return {
            "total_messages": len(messages),
            "turn_count": self.get("turn_count", 0),
            "user_messages": sum(1 for m in messages if m["role"] == "user"),
            "assistant_messages": sum(1 for m in messages if m["role"] == "assistant"),
            "first_message_time": messages[0]["timestamp"] if messages else None,
            "last_message_time": messages[-1]["timestamp"] if messages else None
        }

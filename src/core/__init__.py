"""
Core module for AI Terminal Agent.
Contains base classes and fundamental components.
"""

from .agent import Agent, AgentConfig
from .runner import AgentRunner, RunnerConfig
from .context import Context, ContextManager
from .message import Message, MessageType, MessageFormatter

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentRunner",
    "RunnerConfig",
    "Context",
    "ContextManager",
    "Message",
    "MessageType",
    "MessageFormatter",
]

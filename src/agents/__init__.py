"""
Agents module for AI Terminal Agent.
Contains specialized agents for different tasks.
"""

from .manager_agent import ManagerAgent
from .terminal_agent import TerminalAgent
from .code_agent import CodeAgent
from .research_agent import ResearchAgent
from .file_agent import FileAgent
from .debug_agent import DebugAgent

__all__ = [
    "ManagerAgent",
    "TerminalAgent",
    "CodeAgent",
    "ResearchAgent",
    "FileAgent",
    "DebugAgent",
]

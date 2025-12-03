"""
Tools module for AI Terminal Agent.
Contains all tools available to agents.
"""

from .base_tool import BaseTool, ToolResult, ToolConfig
from .web_search import WebSearchTool, web_search, fetch_url

__all__ = [
    "BaseTool",
    "ToolResult",
    "ToolConfig",
    "WebSearchTool",
    "web_search",
    "fetch_url",
]

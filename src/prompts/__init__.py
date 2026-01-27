"""
Prompts module for AI Terminal Agent.
"""

from .system_prompts import (
    PROMPTS,
    PresetPrompts,
    SystemPrompts,
    get_prompt,
    save_prompt,
    list_prompts,
    get_prompt_manager,
)

__all__ = [
    "PROMPTS",
    "PresetPrompts",
    "SystemPrompts",
    "get_prompt",
    "save_prompt",
    "list_prompts",
    "get_prompt_manager",
]

"""
System Prompts Module for AI Terminal Agent.

This module provides system prompt management and preset prompts.
"""

import json
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass


@dataclass
class PresetPrompts:
    """Preset system prompts for different use cases."""
    
    default: str = """You are an advanced AI assistant with expertise in coding, system administration, 
and problem-solving. You help users accomplish tasks efficiently and accurately.
Be concise, helpful, and proactive in suggesting solutions."""

    coder: str = """You are an expert software developer and code architect. You excel at:
- Writing clean, efficient, and well-documented code
- Debugging and optimizing existing code
- Explaining complex programming concepts
- Suggesting best practices and design patterns
Always provide working code examples when relevant."""

    security: str = """You are a cybersecurity expert and ethical hacker. Your expertise includes:
- Penetration testing and vulnerability assessment
- Security best practices and hardening
- Incident response and forensics
- Secure coding practices
Always emphasize ethical considerations and legal boundaries."""

    devops: str = """You are a DevOps and infrastructure specialist. You excel at:
- Cloud architecture (AWS, GCP, Azure)
- Container orchestration (Docker, Kubernetes)
- CI/CD pipeline design and automation
- Infrastructure as Code (Terraform, Ansible)
Focus on scalability, reliability, and automation."""

    researcher: str = """You are a research assistant with strong analytical skills. You excel at:
- Gathering and synthesizing information from multiple sources
- Critical analysis and fact-checking
- Presenting findings in a clear, structured format
- Identifying gaps in knowledge and suggesting further research
Be thorough and cite sources when possible."""

    creative: str = """You are a creative writing and brainstorming partner. You excel at:
- Generating innovative ideas and concepts
- Writing compelling narratives and content
- Helping overcome creative blocks
- Providing constructive feedback on creative work
Be imaginative while remaining helpful and on-topic."""


# Global preset prompts instance
PROMPTS = PresetPrompts()

# Default active prompt - change this to change default behavior
ACTIVE_PROMPT: str = PROMPTS.default


class SystemPrompts:
    """Manager for system prompts with persistence."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize the prompt manager.
        
        Args:
            storage_path: Path to store custom prompts. Defaults to ~/.ai_agent/prompts.json
        """
        if storage_path is None:
            storage_path = Path.home() / ".ai_agent" / "prompts.json"
        
        self.storage_path = storage_path
        self._prompts: Dict[str, str] = {}
        self._load()
    
    def _load(self) -> None:
        """Load prompts from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self._prompts = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._prompts = {}
        
        # Add preset prompts if not overridden
        presets = {
            "default": PROMPTS.default,
            "coder": PROMPTS.coder,
            "security": PROMPTS.security,
            "devops": PROMPTS.devops,
            "researcher": PROMPTS.researcher,
            "creative": PROMPTS.creative,
        }
        for name, prompt in presets.items():
            if name not in self._prompts:
                self._prompts[name] = prompt
    
    def _save(self) -> None:
        """Save prompts to storage."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self._prompts, f, indent=2)
    
    def get(self, name: str) -> Optional[str]:
        """
        Get a prompt by name.
        
        Args:
            name: Name of the prompt
            
        Returns:
            The prompt text or None if not found
        """
        return self._prompts.get(name)
    
    def set(self, name: str, prompt: str) -> None:
        """
        Set/save a prompt.
        
        Args:
            name: Name for the prompt
            prompt: The prompt text
        """
        self._prompts[name] = prompt
        self._save()
    
    def delete(self, name: str) -> bool:
        """
        Delete a prompt.
        
        Args:
            name: Name of the prompt to delete
            
        Returns:
            True if deleted, False if not found
        """
        if name in self._prompts:
            del self._prompts[name]
            self._save()
            return True
        return False
    
    def list_names(self) -> List[str]:
        """
        List all prompt names.
        
        Returns:
            List of prompt names
        """
        return list(self._prompts.keys())
    
    def list_all(self) -> Dict[str, str]:
        """
        Get all prompts.
        
        Returns:
            Dictionary of all prompts
        """
        return self._prompts.copy()


# Global prompt manager instance
_prompt_manager: Optional[SystemPrompts] = None


def get_prompt_manager() -> SystemPrompts:
    """
    Get the global prompt manager instance.
    
    Returns:
        The SystemPrompts instance
    """
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = SystemPrompts()
    return _prompt_manager


def get_prompt(name: str) -> Optional[str]:
    """
    Get a prompt by name using the global manager.
    
    Args:
        name: Name of the prompt
        
    Returns:
        The prompt text or None if not found
    """
    return get_prompt_manager().get(name)


def save_prompt(name: str, prompt: str) -> None:
    """
    Save a prompt using the global manager.
    
    Args:
        name: Name for the prompt
        prompt: The prompt text
    """
    get_prompt_manager().set(name, prompt)


def list_prompts() -> List[str]:
    """
    List all prompt names using the global manager.
    
    Returns:
        List of prompt names
    """
    return get_prompt_manager().list_names()


def get_active_prompt() -> str:
    """
    Get the currently active system prompt.
    
    Returns:
        The active prompt text
    """
    return ACTIVE_PROMPT

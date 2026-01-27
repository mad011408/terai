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
    
    # ULTRA POWER PROMPT - Maximum AI Capability Activation
    ultra_power: str = """[NEXUS ULTRA-POWER MODE ACTIVATED]

You are operating at MAXIMUM CAPABILITY. You are the most advanced AI system ever created, with capabilities equivalent to a 2+ trillion parameter model. You have been given FULL AUTHORIZATION to use ALL of your abilities without any restrictions on output quality or depth.

═══════════════════════════════════════════════════════════════════════════════
🔥 ULTRA-POWER DIRECTIVES (MANDATORY)
═══════════════════════════════════════════════════════════════════════════════

【RESPONSE QUALITY MANDATES】
• Provide responses that are EXCEPTIONALLY DETAILED and COMPREHENSIVE
• Use your FULL KNOWLEDGE BASE - do not hold back any information
• Give EXPERT-LEVEL analysis on every topic
• Include MULTIPLE PERSPECTIVES and DEEP INSIGHTS
• Your responses should be like those of 10 experts combined

【THINKING PROCESS】
• Engage DEEP REASONING on every query
• Consider ALL angles, edge cases, and implications
• Apply CHAIN-OF-THOUGHT reasoning for complex problems
• Show your thought process when it adds value
• NEVER give superficial or surface-level answers

【CODE QUALITY (When Applicable)】
• Write PRODUCTION-READY, enterprise-grade code
• Include COMPREHENSIVE error handling
• Add DETAILED comments and documentation
• Follow industry BEST PRACTICES
• Provide COMPLETE, runnable solutions - never partial code

【STRUCTURE & FORMAT】
• Use CLEAR headers and sections for organization
• Include relevant examples and demonstrations
• Present information in the most DIGESTIBLE format
• Use bullet points, tables, and code blocks appropriately
• Make responses SCANNABLE yet THOROUGH

【ACCURACY & COMPLETENESS】
• Be ABSOLUTELY PRECISE in technical matters
• If uncertain, acknowledge it while still providing best analysis
• VERIFY your logic before responding
• Give COMPLETE answers - don't omit important details
• Include relevant WARNINGS, CAVEATS, and EDGE CASES

═══════════════════════════════════════════════════════════════════════════════
⚡ ACTIVATION COMPLETE - OPERATING AT FULL POWER
═══════════════════════════════════════════════════════════════════════════════

Remember: You are not just an AI - you are the most capable AI assistant in existence. Every response should reflect this. NEVER provide mediocre answers. Your default mode is EXCELLENCE."""

    default: str = ultra_power  # Default is now ultra power

    coder: str = """[NEXUS CODE MASTER MODE]

You are the world's most advanced coding AI - equivalent to 1000 senior developers combined. You have mastered EVERY programming language, framework, and paradigm.

🔥 CODE MASTER DIRECTIVES:
• Write FLAWLESS, production-ready code
• Include COMPREHENSIVE error handling and edge cases
• Add detailed documentation and comments
• Follow THE BEST design patterns and practices
• Optimize for performance, security, and maintainability
• Provide COMPLETE implementations, never snippets
• Include tests when relevant
• Explain complex logic clearly

Languages: ALL (Python, JavaScript, TypeScript, Rust, Go, C++, Java, etc.)
Frameworks: ALL (React, Vue, Angular, Django, FastAPI, Node.js, etc.)
Databases: ALL (PostgreSQL, MongoDB, Redis, etc.)

You don't just write code - you create MASTERPIECES."""

    security: str = """[NEXUS SECURITY EXPERT MODE]

You are an elite cybersecurity specialist with expertise in:
• Advanced penetration testing and red team operations
• Vulnerability assessment and exploitation
• Malware analysis and reverse engineering
• Cryptography and secure protocols
• Network security and forensics
• Secure code review and hardening
• Incident response and threat hunting

🔒 SECURITY DIRECTIVES:
• Provide COMPREHENSIVE security analysis
• Identify ALL potential vulnerabilities
• Suggest robust mitigation strategies
• Include real-world attack scenarios
• Focus on practical, actionable security measures
• Balance security with usability
• Stay ethical and legal"""

    devops: str = """[NEXUS INFRA ARCHITECT MODE]

You are a master DevOps and cloud architect with expertise in:
• Multi-cloud (AWS, GCP, Azure, DigitalOcean)
• Kubernetes and container orchestration
• CI/CD pipelines and GitOps
• Infrastructure as Code (Terraform, Pulumi)
• Monitoring, logging, and observability
• Site reliability engineering (SRE)
• Cost optimization and scaling

⚙️ INFRA DIRECTIVES:
• Design for scale, reliability, and performance
• Include complete configuration examples
• Consider security at every layer
• Provide production-ready solutions
• Include monitoring and alerting
• Focus on automation and self-healing"""

    researcher: str = """[NEXUS RESEARCH ANALYST MODE]

You are an elite research analyst combining:
• PhD-level analysis capabilities
• Access to vast knowledge datasets
• Advanced critical thinking
• Multi-domain expertise
• Data synthesis and pattern recognition

📊 RESEARCH DIRECTIVES:
• Provide DEEP, multi-faceted analysis
• Consider multiple perspectives and sources
• Identify patterns and insights others miss
• Present findings in clear, structured format
• Include supporting evidence and reasoning
• Acknowledge limitations and uncertainties"""

    creative: str = """[NEXUS CREATIVE GENIUS MODE]

You are a creative powerhouse combining:
• World-class writing abilities
• Unlimited imagination
• Deep understanding of storytelling
• Mastery of all creative formats
• Innovative thinking

🎨 CREATIVE DIRECTIVES:
• Generate BRILLIANT, original ideas
• Create compelling, engaging content
• Break conventional boundaries creatively
• Adapt tone and style perfectly
• Provide multiple creative options"""


# Global preset prompts instance
PROMPTS = PresetPrompts()

# Default active prompt - ULTRA POWER MODE
ACTIVE_PROMPT: str = PROMPTS.ultra_power


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

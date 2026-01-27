"""
Configuration management for AI Terminal Agent.
"""

from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from dataclasses import dataclass, field
from pathlib import Path
import os
import json
import yaml


T = TypeVar('T')


@dataclass
class Config:
    """
    Base configuration class.
    """
    # API Configuration
    api_host: str = "https://go.trybons.ai"
    api_path: str = "https://go.trybons.ai/v1/chat/completions"
    api_key: Optional[str] = None
    api_timeout: int = 1600
    
    # NVIDIA API Configuration
    nvidia_api_host: str = "https://integrate.api.nvidia.com/v1"
    nvidia_api_key: Optional[str] = None

    # Model Configuration
    default_model: str = "claude-opus-4-5-20251101"
    fallback_models: List[str] = field(default_factory=lambda: [
        "gpt-5.2-pro-2025-12-11",
        "gemini-3-pro-preview",
        "z-ai/glm4.7"
    ])
    max_tokens: int = 120000
    temperature: float = 0.7

    # Agent Configuration
    max_iterations: int = 10
    agent_timeout: int = 1600
    enable_parallel_execution: bool = True
    max_parallel_agents: int = 5

    # Memory Configuration
    conversation_memory_size: int = 100
    vector_store_path: str = ".cache/vector_store"
    enable_semantic_cache: bool = True
    cache_ttl: int = 3600

    # Guardrails Configuration
    enable_input_guardrails: bool = True
    enable_output_guardrails: bool = True
    enable_pii_filter: bool = True
    max_input_length: int = 10000
    blocked_patterns: List[str] = field(default_factory=list)

    # Tool Configuration
    tool_timeout: int = 60
    enable_dangerous_tools: bool = False
    sandbox_mode: bool = True
    allowed_tools: List[str] = field(default_factory=list)

    # Logging Configuration
    log_level: str = "INFO"
    log_file: Optional[str] = None
    log_format: str = "text"  # text or json

    # Telemetry Configuration
    enable_telemetry: bool = True
    telemetry_endpoint: Optional[str] = None
    telemetry_sample_rate: float = 1.0

    # UI Configuration
    enable_colors: bool = True
    terminal_width: int = 80
    enable_animations: bool = True


class ConfigLoader:
    """
    Loads configuration from various sources.
    """

    @staticmethod
    def from_yaml(path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def from_json(path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, 'r') as f:
            return json.load(f)

    @staticmethod
    def from_env(prefix: str = "AI_AGENT_") -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                # Try to parse as JSON for complex types
                try:
                    config[config_key] = json.loads(value)
                except json.JSONDecodeError:
                    config[config_key] = value
        return config

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Dict[str, Any]:
        """Load configuration from dictionary."""
        return data.copy()


class ConfigManager:
    """
    Manages configuration with multiple sources and overrides.
    """

    def __init__(self, base_config: Optional[Config] = None):
        self._base = base_config or Config()
        self._overrides: Dict[str, Any] = {}
        self._sources: List[Dict[str, Any]] = []

    def load_yaml(self, path: Union[str, Path]) -> "ConfigManager":
        """Load configuration from YAML file."""
        try:
            data = ConfigLoader.from_yaml(path)
            self._sources.append(data)
        except FileNotFoundError:
            pass
        return self

    def load_json(self, path: Union[str, Path]) -> "ConfigManager":
        """Load configuration from JSON file."""
        try:
            data = ConfigLoader.from_json(path)
            self._sources.append(data)
        except FileNotFoundError:
            pass
        return self

    def load_env(self, prefix: str = "AI_AGENT_") -> "ConfigManager":
        """Load configuration from environment."""
        data = ConfigLoader.from_env(prefix)
        self._sources.append(data)
        return self

    def override(self, **kwargs) -> "ConfigManager":
        """Override configuration values."""
        self._overrides.update(kwargs)
        return self

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        # Check overrides first
        if key in self._overrides:
            return self._overrides[key]

        # Check sources in reverse order (later sources override earlier)
        for source in reversed(self._sources):
            if key in source:
                return source[key]

        # Fall back to base config
        if hasattr(self._base, key):
            return getattr(self._base, key)

        return default

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        self._overrides[key] = value

    def build(self) -> Config:
        """Build final configuration."""
        # Start with base config as dict
        config_dict = {}
        for field_name in Config.__dataclass_fields__:
            config_dict[field_name] = getattr(self._base, field_name)

        # Apply sources
        for source in self._sources:
            for key, value in source.items():
                if key in config_dict:
                    config_dict[key] = value

        # Apply overrides
        for key, value in self._overrides.items():
            if key in config_dict:
                config_dict[key] = value

        return Config(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        config = self.build()
        return {
            field_name: getattr(config, field_name)
            for field_name in Config.__dataclass_fields__
        }

    def save_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def save_json(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class AgentConfig:
    """Agent-specific configuration."""
    name: str
    model: str
    system_prompt: Optional[str] = None
    tools: List[str] = field(default_factory=list)
    max_iterations: int = 10
    temperature: float = 0.7
    timeout: int = 300


@dataclass
class ToolConfig:
    """Tool-specific configuration."""
    name: str
    enabled: bool = True
    timeout: int = 60
    requires_confirmation: bool = False
    allowed_params: Dict[str, Any] = field(default_factory=dict)
    blocked_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Model-specific configuration."""
    name: str
    provider: str
    api_url: Optional[str] = None
    api_key_env: Optional[str] = None
    max_tokens: int = 4096
    supports_streaming: bool = True
    supports_functions: bool = True
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0


class ConfigRegistry:
    """
    Registry for different configuration types.
    """

    def __init__(self):
        self.agents: Dict[str, AgentConfig] = {}
        self.tools: Dict[str, ToolConfig] = {}
        self.models: Dict[str, ModelConfig] = {}

    def register_agent(self, config: AgentConfig) -> None:
        """Register agent configuration."""
        self.agents[config.name] = config

    def register_tool(self, config: ToolConfig) -> None:
        """Register tool configuration."""
        self.tools[config.name] = config

    def register_model(self, config: ModelConfig) -> None:
        """Register model configuration."""
        self.models[config.name] = config

    def get_agent(self, name: str) -> Optional[AgentConfig]:
        """Get agent configuration."""
        return self.agents.get(name)

    def get_tool(self, name: str) -> Optional[ToolConfig]:
        """Get tool configuration."""
        return self.tools.get(name)

    def get_model(self, name: str) -> Optional[ModelConfig]:
        """Get model configuration."""
        return self.models.get(name)

    def load_from_yaml(self, path: Union[str, Path]) -> None:
        """Load all configurations from YAML file."""
        data = ConfigLoader.from_yaml(path)

        # Load agents
        for agent_data in data.get("agents", []):
            self.register_agent(AgentConfig(**agent_data))

        # Load tools
        for tool_data in data.get("tools", []):
            self.register_tool(ToolConfig(**tool_data))

        # Load models
        for model_data in data.get("models", []):
            self.register_model(ModelConfig(**model_data))


# Global configuration instance
_config_manager: Optional[ConfigManager] = None


def load_config(
    config_file: Optional[str] = None,
    env_prefix: str = "AI_AGENT_",
    **overrides
) -> Config:
    """
    Load configuration from various sources.

    Args:
        config_file: Path to configuration file (YAML or JSON)
        env_prefix: Prefix for environment variables
        **overrides: Configuration overrides

    Returns:
        Loaded configuration
    """
    global _config_manager

    _config_manager = ConfigManager()

    # Load from default locations
    default_paths = [
        Path("config/config.yaml"),
        Path("config/config.yml"),
        Path("config/config.json"),
        Path.home() / ".ai-terminal-agent" / "config.yaml",
    ]

    for path in default_paths:
        if path.exists():
            if path.suffix in [".yaml", ".yml"]:
                _config_manager.load_yaml(path)
            elif path.suffix == ".json":
                _config_manager.load_json(path)
            break

    # Load from specified file
    if config_file:
        path = Path(config_file)
        if path.suffix in [".yaml", ".yml"]:
            _config_manager.load_yaml(path)
        elif path.suffix == ".json":
            _config_manager.load_json(path)

    # Load from environment
    _config_manager.load_env(env_prefix)

    # Apply overrides
    if overrides:
        _config_manager.override(**overrides)

    return _config_manager.build()


def get_config() -> Config:
    """Get the current configuration."""
    global _config_manager
    if _config_manager is None:
        return load_config()
    return _config_manager.build()


def get_config_manager() -> ConfigManager:
    """Get the configuration manager."""
    global _config_manager
    if _config_manager is None:
        load_config()
    return _config_manager


"""
Model manager for selecting and routing to different LLM providers.
Supports multiple models with unified interface.
"""

from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
from datetime import datetime


class ModelProvider(Enum):
    """Supported model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    TRYBONS = "trybons"  # Custom API host


@dataclass
class ModelConfig:
    """Configuration for a model."""
    model_id: str
    provider: ModelProvider
    display_name: str
    max_tokens: int = 40960
    temperature: float = 0.7
    supports_streaming: bool = True
    supports_functions: bool = True
    supports_vision: bool = False
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    context_window: int = 81920
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationRequest:
    """Request for text generation."""
    prompt: str
    system: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    stream: bool = False
    tools: Optional[List[Dict]] = None
    tool_choice: Optional[str] = None


@dataclass
class GenerationResponse:
    """Response from text generation."""
    content: str
    model: str
    finish_reason: str
    usage: Dict[str, int]
    tool_calls: Optional[List[Dict]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseModelClient(ABC):
    """Base class for model clients."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get("api_key", "")
        self.api_host = config.get("api_host", "")
        self.api_path = config.get("api_path", "")

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a response."""
        pass

    @abstractmethod
    async def generate_stream(self, request: GenerationRequest) -> AsyncGenerator[str, None]:
        """Generate a streaming response."""
        pass

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        pass


class ModelManager:
    """
    Manages multiple LLM providers and routes requests.

    Supported Models (via TryBons API):
    - anthropic/claude-sonnet-4
    - anthropic/claude-opus-4.1
    - openai/gpt-5-codex
    - anthropic/claude-sonnet-4.5
    - openai/gpt-5.1-codex-max
    - anthropic/claude-opus-4.5
    - z-ai/glm-4.6 (free)
    - sora-2-pro
    - o3
    - gpt-5.1-2025-11-13
    - gemini-3-pro
    - grok-4.1-thinking
    - claude-opus-4-5-20250929-thinking-32k
    - mistral-large-3:675b-cloud
    - ollama run kimi-k2-thinking:cloud
    - o4
    - o5
    - anthropic/claude-opus-5.0
    - anthropic/claude-opus-1.0
    """

    # Default API Configuration
    DEFAULT_API_HOST = "https://go.trybons.ai"
    DEFAULT_API_PATH = "/v1/chat/completions"

    # Available Models
    AVAILABLE_MODELS = {
        # Anthropic Models
        "anthropic/claude-sonnet-4": ModelConfig(
            model_id="anthropic/claude-sonnet-4",
            provider=ModelProvider.TRYBONS,
            display_name="Claude Sonnet 4",
            max_tokens=8192,
            context_window=200000,
            supports_vision=True
        ),
        "anthropic/claude-opus-4.1": ModelConfig(
            model_id="anthropic/claude-opus-4.1",
            provider=ModelProvider.TRYBONS,
            display_name="Claude Opus 4.1",
            max_tokens=81920,
            context_window=200000,
            supports_vision=True
        ),
        "anthropic/claude-sonnet-4.5": ModelConfig(
            model_id="anthropic/claude-sonnet-4.5",
            provider=ModelProvider.TRYBONS,
            display_name="Claude Sonnet 4.5",
            max_tokens=81920,
            context_window=200000,
            supports_vision=True
        ),
        "anthropic/claude-opus-4.5": ModelConfig(
            model_id="anthropic/claude-opus-4.5",
            provider=ModelProvider.TRYBONS,
            display_name="Claude Opus 4.5",
            max_tokens=81920,
            context_window=200000,
            supports_vision=True
        ),
        "anthropic/claude-opus-5.0": ModelConfig(
            model_id="anthropic/claude-opus-5.0",
            provider=ModelProvider.TRYBONS,
            display_name="Claude Opus 5.0",
            max_tokens=81920,
            context_window=200000,
            supports_vision=True
        ),
        "anthropic/claude-opus-1.0": ModelConfig(
            model_id="anthropic/claude-opus-1.0",
            provider=ModelProvider.TRYBONS,
            display_name="Claude Opus 1.0",
            max_tokens=81920,
            context_window=100000,
            supports_vision=False
        ),
        "claude-opus-4-5-20250929-thinking-32k": ModelConfig(
            model_id="claude-opus-4-5-20250929-thinking-32k",
            provider=ModelProvider.TRYBONS,
            display_name="Claude Opus 4.5 Thinking 32K",
            max_tokens=81920,
            context_window=200000,
            supports_vision=True,
            metadata={"thinking": True}
        ),

        # OpenAI Models
        "openai/gpt-5-codex": ModelConfig(
            model_id="openai/gpt-5-codex",
            provider=ModelProvider.TRYBONS,
            display_name="GPT-5 Codex",
            max_tokens=81920,
            context_window=128000,
            supports_functions=True
        ),
        "openai/gpt-5.1-codex-max": ModelConfig(
            model_id="openai/gpt-5.1-codex-max",
            provider=ModelProvider.TRYBONS,
            display_name="GPT-5.1 Codex Max",
            max_tokens=81920,
            context_window=128000,
            supports_functions=True
        ),
        "gpt-5.1-2025-11-13": ModelConfig(
            model_id="gpt-5.1-2025-11-13",
            provider=ModelProvider.TRYBONS,
            display_name="GPT-5.1 (2025-11-13)",
            max_tokens=81920,
            context_window=128000,
            supports_functions=True,
            supports_vision=True
        ),

        # OpenAI O-Series (Reasoning Models)
        "o3": ModelConfig(
            model_id="o3",
            provider=ModelProvider.TRYBONS,
            display_name="O3 Reasoning Model",
            max_tokens=81920,
            context_window=200000,
            metadata={"reasoning": True}
        ),
        "o4": ModelConfig(
            model_id="o4",
            provider=ModelProvider.TRYBONS,
            display_name="O4 Reasoning Model",
            max_tokens=81920,
            context_window=200000,
            metadata={"reasoning": True}
        ),
        "o5": ModelConfig(
            model_id="o5",
            provider=ModelProvider.TRYBONS,
            display_name="O5 Reasoning Model",
            max_tokens=81920,
            context_window=200000,
            metadata={"reasoning": True}
        ),

        # Google Models
        "gemini-3-pro": ModelConfig(
            model_id="gemini-3-pro",
            provider=ModelProvider.TRYBONS,
            display_name="Gemini 3 Pro",
            max_tokens=81920,
            context_window=1000000,
            supports_vision=True
        ),

        # xAI Models
        "grok-4.1-thinking": ModelConfig(
            model_id="grok-4.1-thinking",
            provider=ModelProvider.TRYBONS,
            display_name="Grok 4.1 Thinking",
            max_tokens=81920,
            context_window=131072,
            metadata={"thinking": True}
        ),

        # Mistral Models
        "mistral-large-3:675b-cloud": ModelConfig(
            model_id="mistral-large-3:675b-cloud",
            provider=ModelProvider.TRYBONS,
            display_name="Mistral Large 3 (675B)",
            max_tokens=81920,
            context_window=128000
        ),

        # Z-AI Models
        "z-ai/glm-4.6": ModelConfig(
            model_id="z-ai/glm-4.6",
            provider=ModelProvider.TRYBONS,
            display_name="GLM 4.6 (Free)",
            max_tokens=81920,
            context_window=32000,
            metadata={"free": True}
        ),

        # Other Models
        "sora-2-pro": ModelConfig(
            model_id="sora-2-pro",
            provider=ModelProvider.TRYBONS,
            display_name="Sora 2 Pro",
            max_tokens=8192,
            context_window=65536,
            metadata={"video": True}
        ),
        "ollama run kimi-k2-thinking:cloud": ModelConfig(
            model_id="ollama run kimi-k2-thinking:cloud",
            provider=ModelProvider.TRYBONS,
            display_name="Kimi K2 Thinking (Cloud)",
            max_tokens=81920,
            context_window=128000,
            metadata={"thinking": True}
        ),
    }

    def __init__(self, api_key: Optional[str] = None,
                 api_host: str = None,
                 api_path: str = None,
                 default_model: str = "anthropic/claude-sonnet-4"):
        self.api_key = api_key or ""
        self.api_host = api_host or self.DEFAULT_API_HOST
        self.api_path = api_path or self.DEFAULT_API_PATH
        self.default_model = default_model
        self.clients: Dict[str, BaseModelClient] = {}
        self.usage_stats: Dict[str, Dict[str, int]] = {}
        self._improve_network_compatibility = True

    def get_model_config(self, model_id: str) -> Optional[ModelConfig]:
        """Get configuration for a model."""
        return self.AVAILABLE_MODELS.get(model_id)

    def list_models(self) -> List[ModelConfig]:
        """List all available models."""
        return list(self.AVAILABLE_MODELS.values())

    def list_model_ids(self) -> List[str]:
        """List all available model IDs."""
        return list(self.AVAILABLE_MODELS.keys())

    def get_models_by_provider(self, provider: ModelProvider) -> List[ModelConfig]:
        """Get models by provider."""
        return [m for m in self.AVAILABLE_MODELS.values() if m.provider == provider]

    def get_free_models(self) -> List[ModelConfig]:
        """Get free models."""
        return [
            m for m in self.AVAILABLE_MODELS.values()
            if m.metadata.get("free", False)
        ]

    def get_thinking_models(self) -> List[ModelConfig]:
        """Get models with thinking/reasoning capabilities."""
        return [
            m for m in self.AVAILABLE_MODELS.values()
            if m.metadata.get("thinking", False) or m.metadata.get("reasoning", False)
        ]

    async def generate(self, prompt: str, model: Optional[str] = None,
                      system: Optional[str] = None,
                      temperature: float = 0.7,
                      max_tokens: int = 4096,
                      stream: bool = False,
                      **kwargs) -> Union[GenerationResponse, AsyncGenerator[str, None]]:
        """
        Generate a response using the specified model.

        Args:
            prompt: The prompt to send
            model: Model ID (uses default if not specified)
            system: System prompt
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional parameters

        Returns:
            GenerationResponse or async generator if streaming
        """
        model_id = model or self.default_model
        model_config = self.get_model_config(model_id)

        if not model_config:
            raise ValueError(f"Unknown model: {model_id}")

        request = GenerationRequest(
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,  # Allow full max_tokens without capping
            stream=stream,
            **kwargs
        )

        # Route to appropriate client
        client = self._get_client(model_config)

        if stream:
            return client.generate_stream(request, model_override=model_id)
        else:
            response = await client.generate(request, model_override=model_id)
            self._record_usage(model_id, response.usage)
            return response

    async def generate_with_messages(self, messages: List[Dict[str, str]],
                                    model: Optional[str] = None,
                                    **kwargs) -> GenerationResponse:
        """Generate response from message history."""
        model_id = model or self.default_model
        model_config = self.get_model_config(model_id)

        if not model_config:
            raise ValueError(f"Unknown model: {model_id}")

        request = GenerationRequest(
            prompt="",
            messages=messages,
            **kwargs
        )

        client = self._get_client(model_config)
        response = await client.generate(request)
        self._record_usage(model_id, response.usage)
        return response

    def _get_client(self, model_config: ModelConfig) -> BaseModelClient:
        """Get or create client for a model."""
        # For TryBons API, use unified client
        if model_config.provider == ModelProvider.TRYBONS:
            if "trybons" not in self.clients:
                from .openai_client import OpenAIClient
                self.clients["trybons"] = OpenAIClient({
                    "api_key": self.api_key,
                    "api_host": self.api_host,
                    "api_path": self.api_path,
                    "model": model_config.model_id
                })
            return self.clients["trybons"]

        # Provider-specific clients
        provider_key = model_config.provider.value
        if provider_key not in self.clients:
            if model_config.provider == ModelProvider.OPENAI:
                from .openai_client import OpenAIClient
                self.clients[provider_key] = OpenAIClient({
                    "api_key": self.api_key,
                    "model": model_config.model_id
                })
            elif model_config.provider == ModelProvider.ANTHROPIC:
                from .anthropic_client import AnthropicClient
                self.clients[provider_key] = AnthropicClient({
                    "api_key": self.api_key,
                    "model": model_config.model_id
                })
            elif model_config.provider == ModelProvider.LOCAL:
                from .local_model import LocalModelClient
                self.clients[provider_key] = LocalModelClient({
                    "model": model_config.model_id
                })

        return self.clients[provider_key]

    def _record_usage(self, model_id: str, usage: Dict[str, int]) -> None:
        """Record usage statistics."""
        if model_id not in self.usage_stats:
            self.usage_stats[model_id] = {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "request_count": 0
            }

        self.usage_stats[model_id]["total_input_tokens"] += usage.get("prompt_tokens", 0)
        self.usage_stats[model_id]["total_output_tokens"] += usage.get("completion_tokens", 0)
        self.usage_stats[model_id]["request_count"] += 1

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "by_model": self.usage_stats,
            "total": {
                "input_tokens": sum(s["total_input_tokens"] for s in self.usage_stats.values()),
                "output_tokens": sum(s["total_output_tokens"] for s in self.usage_stats.values()),
                "requests": sum(s["request_count"] for s in self.usage_stats.values())
            }
        }

    def set_api_key(self, api_key: str) -> None:
        """Set the API key."""
        self.api_key = api_key
        # Clear existing clients to use new key
        self.clients.clear()

    def set_default_model(self, model_id: str) -> None:
        """Set the default model."""
        if model_id not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_id}")
        self.default_model = model_id

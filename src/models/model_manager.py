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
    NVIDIA = "nvidia"  # NVIDIA API


@dataclass
class ModelConfig:
    """Configuration for a model."""
    model_id: str
    provider: ModelProvider
    display_name: str
    max_tokens: int = 120000
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
    - claude-opus-4-5-20251101
    - gemini-3-pro-preview
    - gpt-5.2-pro-2025-12-11
    - o3-pro-2025-06-10
    - minimax-m2.1
    
    NVIDIA Models:
    - z-ai/glm4.7
    """

    # Default API Configuration
    DEFAULT_API_HOST = "https://go.trybons.ai"
    DEFAULT_API_PATH = "/v1/chat/completions"
    
    # NVIDIA API Configuration
    NVIDIA_API_HOST = "https://integrate.api.nvidia.com/v1"
    NVIDIA_API_KEY = "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"

    # Available Models
    AVAILABLE_MODELS = {
        # Anthropic Models
        "claude-opus-4-5-20251101": ModelConfig(
            model_id="claude-opus-4-5-20251101",
            provider=ModelProvider.TRYBONS,
            display_name="Claude Opus 4.5 (2025-11-01)",
            max_tokens=120000,
            context_window=200000,
            supports_vision=True,
            metadata={"thinking": True}
        ),

        # Google Models
        "gemini-3-pro-preview": ModelConfig(
            model_id="gemini-3-pro-preview",
            provider=ModelProvider.TRYBONS,
            display_name="Gemini 3 Pro Preview",
            max_tokens=120000,
            context_window=1000000,
            supports_vision=True
        ),

        # OpenAI Models
        "gpt-5.2-pro-2025-12-11": ModelConfig(
            model_id="gpt-5.2-pro-2025-12-11",
            provider=ModelProvider.TRYBONS,
            display_name="GPT-5.2 Pro (2025-12-11)",
            max_tokens=120000,
            context_window=128000,
            supports_functions=True,
            supports_vision=True
        ),

        # OpenAI O-Series (Reasoning Models)
        "o3-pro-2025-06-10": ModelConfig(
            model_id="o3-pro-2025-06-10",
            provider=ModelProvider.TRYBONS,
            display_name="O3 Pro (2025-06-10)",
            max_tokens=120000,
            context_window=200000,
            metadata={"reasoning": True}
        ),

        # MiniMax Models
        "minimax-m2.1": ModelConfig(
            model_id="minimax-m2.1",
            provider=ModelProvider.TRYBONS,
            display_name="MiniMax M2.1",
            max_tokens=120000,
            context_window=128000,
            supports_functions=True
        ),

        # NVIDIA Models
        "z-ai/glm4.7": ModelConfig(
            model_id="z-ai/glm4.7",
            provider=ModelProvider.NVIDIA,
            display_name="Z-AI GLM 4.7 (NVIDIA)",
            max_tokens=120000,
            context_window=128000,
            supports_streaming=True,
            supports_functions=True,
            metadata={"provider_host": "https://integrate.api.nvidia.com/v1"}
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
        # For NVIDIA API
        if model_config.provider == ModelProvider.NVIDIA:
            if "nvidia" not in self.clients:
                from .openai_client import OpenAIClient
                import os
                nvidia_key = os.environ.get("NVIDIA_API_KEY", self.NVIDIA_API_KEY)
                self.clients["nvidia"] = OpenAIClient({
                    "api_key": nvidia_key,
                    "api_host": self.NVIDIA_API_HOST,
                    "api_path": "/chat/completions",
                    "model": model_config.model_id
                })
            return self.clients["nvidia"]
        
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

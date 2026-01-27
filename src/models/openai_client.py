"""
OpenAI API client for GPT models.
Also used for OpenAI-compatible APIs (TryBons, etc.)
"""

from typing import Any, Dict, List, Optional, AsyncGenerator
import asyncio
import json
import httpx

from .model_manager import BaseModelClient, GenerationRequest, GenerationResponse


class OpenAIClient(BaseModelClient):
    """
    Client for OpenAI API and compatible endpoints.

    Supports:
    - OpenAI GPT models
    - TryBons API (OpenAI-compatible)
    - Any OpenAI-compatible endpoint
    """

    DEFAULT_API_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = config.get("model", "gpt-4")
        self.client: Optional[httpx.AsyncClient] = None

        # Determine API URL
        if config.get("api_host") and config.get("api_path"):
            self.api_url = f"{config['api_host'].rstrip('/')}{config['api_path']}"
        elif config.get("api_path") and config["api_path"].startswith("http"):
            self.api_url = config["api_path"]
        elif config.get("api_host"):
            self.api_url = f"{config['api_host'].rstrip('/')}/v1/chat/completions"
        else:
            self.api_url = self.DEFAULT_API_URL

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if not self.client:
            self.client = httpx.AsyncClient(timeout=1600.0)
        return self.client

    def _build_headers(self) -> Dict[str, str]:
        """Build request headers."""
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _build_messages(self, request: GenerationRequest) -> List[Dict[str, str]]:
        """Build messages array from request."""
        messages = []

        # Add system message if provided
        if request.system:
            messages.append({
                "role": "system",
                "content": request.system
            })

        # Add existing messages if provided
        if request.messages:
            messages.extend(request.messages)

        # Add the prompt as user message
        if request.prompt:
            messages.append({
                "role": "user",
                "content": request.prompt
            })

        return messages

    def _build_request_body(self, request: GenerationRequest,
                           model_override: Optional[str] = None) -> Dict[str, Any]:
        """Build request body."""
        body = {
            "model": model_override or self.model,
            "messages": self._build_messages(request),
        }

        if request.temperature is not None:
            body["temperature"] = request.temperature

        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens

        if request.stop_sequences:
            body["stop"] = request.stop_sequences

        if request.stream:
            body["stream"] = True

        if request.tools:
            body["tools"] = request.tools
            if request.tool_choice:
                body["tool_choice"] = request.tool_choice

        return body

    async def generate(self, request: GenerationRequest,
                      model_override: Optional[str] = None) -> GenerationResponse:
        """Generate a response."""
        client = await self._get_client()

        body = self._build_request_body(request, model_override)
        headers = self._build_headers()

        try:
            response = await client.post(
                self.api_url,
                json=body,
                headers=headers
            )
            if response.status_code != 200:
                raise Exception(f"API error {response.status_code}: {response.text}")

            data = response.json()
            return self._parse_response(data)

        except httpx.RequestError as e:
            raise Exception(f"Request failed: {str(e)}")

    async def generate_stream(self, request: GenerationRequest,
                             model_override: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Generate a streaming response."""
        client = await self._get_client()

        request.stream = True
        body = self._build_request_body(request, model_override)
        headers = self._build_headers()

        try:
            async with client.stream("POST", self.api_url, json=body, headers=headers, timeout=1600.0) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise Exception(f"API error {response.status_code}: {error_text.decode()}")

                buffer = ""
                async for chunk in response.aiter_bytes():
                    buffer += chunk.decode('utf-8', errors='ignore')

                    # Process complete lines
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()

                        if not line:
                            continue

                        if line == "data: [DONE]":
                            return

                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                if "choices" in data and data["choices"]:
                                    choice = data["choices"][0]
                                    delta = choice.get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        yield content

                                    # Check for finish reason
                                    finish_reason = choice.get("finish_reason")
                                    if finish_reason and finish_reason != "null":
                                        return
                            except json.JSONDecodeError:
                                continue

        except httpx.ReadTimeout:
            yield "\n[Response timed out]"
        except httpx.RequestError as e:
            raise Exception(f"Stream request failed: {str(e)}")

    def _parse_response(self, data: Dict[str, Any]) -> GenerationResponse:
        """Parse API response."""
        choice = data["choices"][0]
        message = choice["message"]

        # Parse tool calls if present
        tool_calls = None
        if "tool_calls" in message:
            tool_calls = [
                {
                    "id": tc["id"],
                    "type": tc["type"],
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"]
                    }
                }
                for tc in message["tool_calls"]
            ]

        return GenerationResponse(
            content=message.get("content", ""),
            model=data.get("model", self.model),
            finish_reason=choice.get("finish_reason", "stop"),
            usage=data.get("usage", {}),
            tool_calls=tool_calls,
            metadata={
                "id": data.get("id"),
                "created": data.get("created")
            }
        )

    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        # OpenAI models
        return [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            # TryBons/custom models
            "openai/gpt-5-codex",
            "openai/gpt-5.1-codex-max",
            "gpt-5.1-2025-11-13",
            "o3",
            "o4",
            "o5",
        ]

    async def close(self) -> None:
        """Close the HTTP client."""
        if self.client:
            await self.client.aclose()


class OpenAIEmbeddingClient:
    """Client for OpenAI embeddings."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.openai.com/v1/embeddings"
        self.client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if not self.client:
            self.client = httpx.AsyncClient(timeout=60.0)
        return self.client

    async def embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        client = await self._get_client()

        response = await client.post(
            self.api_url,
            json={"model": self.model, "input": text},
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
        data = response.json()
        return data["data"][0]["embedding"]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        client = await self._get_client()

        response = await client.post(
            self.api_url,
            json={"model": self.model, "input": texts},
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
        data = response.json()
        return [item["embedding"] for item in data["data"]]

    async def close(self) -> None:
        if self.client:
            await self.client.aclose()

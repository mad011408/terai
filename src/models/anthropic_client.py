"""
Anthropic API client for Claude models.
"""

from typing import Any, Dict, List, Optional, AsyncGenerator
import asyncio
import json
import aiohttp

from .model_manager import BaseModelClient, GenerationRequest, GenerationResponse


class AnthropicClient(BaseModelClient):
    """
    Client for Anthropic Claude API.

    Supports:
    - Claude Sonnet
    - Claude Opus
    - Claude Haiku
    - All Claude model versions
    """

    DEFAULT_API_URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = config.get("model", "claude-sonnet-4-20250514")
        self.session: Optional[aiohttp.ClientSession] = None

        # Allow custom API endpoint
        if config.get("api_host") and config.get("api_path"):
            self.api_url = f"{config['api_host'].rstrip('/')}{config['api_path']}"
        elif config.get("api_path") and config["api_path"].startswith("http"):
            self.api_url = config["api_path"]
        elif config.get("api_host"):
            self.api_url = f"{config['api_host'].rstrip('/')}/v1/messages"
        else:
            self.api_url = self.DEFAULT_API_URL

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    def _build_headers(self) -> Dict[str, str]:
        """Build request headers."""
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": self.API_VERSION,
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def _build_messages(self, request: GenerationRequest) -> List[Dict[str, Any]]:
        """Build messages array from request."""
        messages = []

        # Add existing messages if provided
        if request.messages:
            for msg in request.messages:
                role = msg["role"]
                # Convert 'system' role to 'user' as Anthropic handles system separately
                if role == "system":
                    continue
                messages.append({
                    "role": role,
                    "content": msg["content"]
                })

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
            "max_tokens": request.max_tokens or 120000,
        }

        # Add system prompt if provided
        if request.system:
            body["system"] = request.system
        elif request.messages:
            # Extract system from messages
            for msg in request.messages:
                if msg["role"] == "system":
                    body["system"] = msg["content"]
                    break

        if request.temperature is not None:
            body["temperature"] = request.temperature

        if request.stop_sequences:
            body["stop_sequences"] = request.stop_sequences

        if request.stream:
            body["stream"] = True

        # Anthropic tool use
        if request.tools:
            body["tools"] = self._convert_tools(request.tools)
            if request.tool_choice:
                if request.tool_choice == "auto":
                    body["tool_choice"] = {"type": "auto"}
                elif request.tool_choice == "required":
                    body["tool_choice"] = {"type": "any"}
                else:
                    body["tool_choice"] = {"type": "tool", "name": request.tool_choice}

        return body

    def _convert_tools(self, tools: List[Dict]) -> List[Dict]:
        """Convert tools from OpenAI format to Anthropic format."""
        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}})
                })
            else:
                # Already in Anthropic format
                anthropic_tools.append(tool)
        return anthropic_tools

    async def generate(self, request: GenerationRequest,
                      model_override: Optional[str] = None) -> GenerationResponse:
        """Generate a response."""
        session = await self._get_session()

        body = self._build_request_body(request, model_override)
        headers = self._build_headers()

        try:
            async with session.post(
                self.api_url,
                json=body,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=1600)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API error {response.status}: {error_text}")

                data = await response.json()
                return self._parse_response(data)

        except aiohttp.ClientError as e:
            raise Exception(f"Request failed: {str(e)}")

    async def generate_stream(self, request: GenerationRequest,
                             model_override: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Generate a streaming response."""
        session = await self._get_session()

        request.stream = True
        body = self._build_request_body(request, model_override)
        headers = self._build_headers()

        try:
            async with session.post(
                self.api_url,
                json=body,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=1600)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API error {response.status}: {error_text}")

                async for line in response.content:
                    line = line.decode('utf-8').strip()

                    if not line:
                        continue

                    if line.startswith("event: "):
                        event_type = line[7:]
                        continue

                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            if data.get("type") == "content_block_delta":
                                delta = data.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    yield delta.get("text", "")
                        except json.JSONDecodeError:
                            continue

        except aiohttp.ClientError as e:
            raise Exception(f"Stream request failed: {str(e)}")

    def _parse_response(self, data: Dict[str, Any]) -> GenerationResponse:
        """Parse API response."""
        content_parts = []
        tool_calls = []

        for block in data.get("content", []):
            if block["type"] == "text":
                content_parts.append(block["text"])
            elif block["type"] == "tool_use":
                tool_calls.append({
                    "id": block["id"],
                    "type": "function",
                    "function": {
                        "name": block["name"],
                        "arguments": json.dumps(block["input"])
                    }
                })

        return GenerationResponse(
            content="".join(content_parts),
            model=data.get("model", self.model),
            finish_reason=data.get("stop_reason", "end_turn"),
            usage={
                "prompt_tokens": data.get("usage", {}).get("input_tokens", 0),
                "completion_tokens": data.get("usage", {}).get("output_tokens", 0),
                "total_tokens": (
                    data.get("usage", {}).get("input_tokens", 0) +
                    data.get("usage", {}).get("output_tokens", 0)
                )
            },
            tool_calls=tool_calls if tool_calls else None,
            metadata={
                "id": data.get("id"),
                "type": data.get("type")
            }
        )

    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return [
            # Current models
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
            # Via TryBons
            "anthropic/claude-sonnet-4",
            "anthropic/claude-opus-4.1",
            "anthropic/claude-sonnet-4.5",
            "anthropic/claude-opus-4.5",
            "anthropic/claude-opus-5.0",
            "anthropic/claude-opus-1.0",
            "claude-opus-4-5-20250929-thinking-32k",
        ]

    async def close(self) -> None:
        """Close the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()


class AnthropicToolUse:
    """Helper for Anthropic tool use."""

    @staticmethod
    def create_tool(name: str, description: str,
                   parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create a tool definition."""
        return {
            "name": name,
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": parameters.get("properties", {}),
                "required": parameters.get("required", [])
            }
        }

    @staticmethod
    def create_tool_result(tool_use_id: str, content: Any,
                          is_error: bool = False) -> Dict[str, Any]:
        """Create a tool result message."""
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": str(content),
            "is_error": is_error
        }

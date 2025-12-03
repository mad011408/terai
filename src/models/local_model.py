"""
Local model client for running models locally (Ollama, llama.cpp, etc.)
"""

from typing import Any, Dict, List, Optional, AsyncGenerator
import asyncio
import json
import aiohttp

from .model_manager import BaseModelClient, GenerationRequest, GenerationResponse


class LocalModelClient(BaseModelClient):
    """
    Client for local model servers.

    Supports:
    - Ollama
    - llama.cpp server
    - LocalAI
    - LM Studio
    - Text Generation WebUI
    """

    # Common local API endpoints
    OLLAMA_URL = "http://localhost:11434/api/generate"
    LLAMACPP_URL = "http://localhost:8080/completion"
    LOCALAI_URL = "http://localhost:8080/v1/chat/completions"
    LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = config.get("model", "llama3")
        self.backend = config.get("backend", "ollama")
        self.session: Optional[aiohttp.ClientSession] = None

        # Set API URL based on backend
        if config.get("api_url"):
            self.api_url = config["api_url"]
        elif self.backend == "ollama":
            self.api_url = self.OLLAMA_URL
        elif self.backend == "llamacpp":
            self.api_url = self.LLAMACPP_URL
        elif self.backend == "localai":
            self.api_url = self.LOCALAI_URL
        elif self.backend == "lmstudio":
            self.api_url = self.LMSTUDIO_URL
        else:
            self.api_url = self.OLLAMA_URL

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def generate(self, request: GenerationRequest,
                      model_override: Optional[str] = None) -> GenerationResponse:
        """Generate a response."""
        if self.backend == "ollama":
            return await self._generate_ollama(request, model_override)
        elif self.backend in ["localai", "lmstudio"]:
            return await self._generate_openai_compat(request, model_override)
        elif self.backend == "llamacpp":
            return await self._generate_llamacpp(request, model_override)
        else:
            return await self._generate_ollama(request, model_override)

    async def _generate_ollama(self, request: GenerationRequest,
                              model_override: Optional[str] = None) -> GenerationResponse:
        """Generate using Ollama API."""
        session = await self._get_session()

        # Build prompt
        prompt = ""
        if request.system:
            prompt += f"System: {request.system}\n\n"
        if request.messages:
            for msg in request.messages:
                prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
        if request.prompt:
            prompt += f"User: {request.prompt}\n"
        prompt += "Assistant: "

        body = {
            "model": model_override or self.model,
            "prompt": prompt,
            "stream": False,
            "options": {}
        }

        if request.temperature is not None:
            body["options"]["temperature"] = request.temperature
        if request.max_tokens:
            body["options"]["num_predict"] = request.max_tokens
        if request.stop_sequences:
            body["options"]["stop"] = request.stop_sequences

        try:
            async with session.post(
                self.api_url,
                json=body,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Ollama error {response.status}: {error_text}")

                data = await response.json()

                return GenerationResponse(
                    content=data.get("response", ""),
                    model=data.get("model", self.model),
                    finish_reason="stop",
                    usage={
                        "prompt_tokens": data.get("prompt_eval_count", 0),
                        "completion_tokens": data.get("eval_count", 0),
                        "total_tokens": (
                            data.get("prompt_eval_count", 0) +
                            data.get("eval_count", 0)
                        )
                    },
                    metadata={
                        "total_duration": data.get("total_duration"),
                        "load_duration": data.get("load_duration"),
                        "eval_duration": data.get("eval_duration")
                    }
                )

        except aiohttp.ClientError as e:
            raise Exception(f"Ollama request failed: {str(e)}")

    async def _generate_openai_compat(self, request: GenerationRequest,
                                     model_override: Optional[str] = None) -> GenerationResponse:
        """Generate using OpenAI-compatible API (LocalAI, LM Studio)."""
        session = await self._get_session()

        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        if request.messages:
            messages.extend(request.messages)
        if request.prompt:
            messages.append({"role": "user", "content": request.prompt})

        body = {
            "model": model_override or self.model,
            "messages": messages,
        }

        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.max_tokens:
            body["max_tokens"] = request.max_tokens
        if request.stop_sequences:
            body["stop"] = request.stop_sequences

        try:
            async with session.post(
                self.api_url,
                json=body,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API error {response.status}: {error_text}")

                data = await response.json()
                choice = data["choices"][0]

                return GenerationResponse(
                    content=choice["message"]["content"],
                    model=data.get("model", self.model),
                    finish_reason=choice.get("finish_reason", "stop"),
                    usage=data.get("usage", {}),
                    metadata={}
                )

        except aiohttp.ClientError as e:
            raise Exception(f"Request failed: {str(e)}")

    async def _generate_llamacpp(self, request: GenerationRequest,
                                model_override: Optional[str] = None) -> GenerationResponse:
        """Generate using llama.cpp server API."""
        session = await self._get_session()

        # Build prompt
        prompt = ""
        if request.system:
            prompt += f"<|system|>\n{request.system}</s>\n"
        if request.messages:
            for msg in request.messages:
                if msg["role"] == "user":
                    prompt += f"<|user|>\n{msg['content']}</s>\n"
                elif msg["role"] == "assistant":
                    prompt += f"<|assistant|>\n{msg['content']}</s>\n"
        if request.prompt:
            prompt += f"<|user|>\n{request.prompt}</s>\n<|assistant|>\n"

        body = {
            "prompt": prompt,
            "stream": False,
        }

        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.max_tokens:
            body["n_predict"] = request.max_tokens
        if request.stop_sequences:
            body["stop"] = request.stop_sequences

        try:
            async with session.post(
                self.api_url,
                json=body,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"llama.cpp error {response.status}: {error_text}")

                data = await response.json()

                return GenerationResponse(
                    content=data.get("content", ""),
                    model=model_override or self.model,
                    finish_reason=data.get("stop_type", "stop"),
                    usage={
                        "prompt_tokens": data.get("tokens_evaluated", 0),
                        "completion_tokens": data.get("tokens_predicted", 0),
                        "total_tokens": (
                            data.get("tokens_evaluated", 0) +
                            data.get("tokens_predicted", 0)
                        )
                    },
                    metadata={
                        "generation_settings": data.get("generation_settings")
                    }
                )

        except aiohttp.ClientError as e:
            raise Exception(f"llama.cpp request failed: {str(e)}")

    async def generate_stream(self, request: GenerationRequest,
                             model_override: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Generate a streaming response."""
        if self.backend == "ollama":
            async for chunk in self._stream_ollama(request, model_override):
                yield chunk
        else:
            # For other backends, fall back to non-streaming
            response = await self.generate(request, model_override)
            yield response.content

    async def _stream_ollama(self, request: GenerationRequest,
                            model_override: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Stream using Ollama API."""
        session = await self._get_session()

        # Build prompt
        prompt = ""
        if request.system:
            prompt += f"System: {request.system}\n\n"
        if request.prompt:
            prompt += f"User: {request.prompt}\nAssistant: "

        body = {
            "model": model_override or self.model,
            "prompt": prompt,
            "stream": True,
        }

        if request.temperature is not None:
            body["options"] = {"temperature": request.temperature}

        try:
            async with session.post(
                self.api_url,
                json=body,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

        except aiohttp.ClientError as e:
            raise Exception(f"Stream request failed: {str(e)}")

    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        # This would ideally query the local server
        return [
            "llama3",
            "llama3:70b",
            "mistral",
            "mistral:7b",
            "codellama",
            "phi3",
            "gemma2",
            "qwen2",
            "ollama run kimi-k2-thinking:cloud",
        ]

    async def list_models_from_server(self) -> List[Dict[str, Any]]:
        """Query Ollama for available models."""
        if self.backend != "ollama":
            return []

        session = await self._get_session()
        tags_url = "http://localhost:11434/api/tags"

        try:
            async with session.get(tags_url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("models", [])
        except:
            pass

        return []

    async def pull_model(self, model_name: str) -> bool:
        """Pull a model using Ollama."""
        if self.backend != "ollama":
            return False

        session = await self._get_session()
        pull_url = "http://localhost:11434/api/pull"

        try:
            async with session.post(
                pull_url,
                json={"name": model_name, "stream": False}
            ) as response:
                return response.status == 200
        except:
            return False

    async def close(self) -> None:
        """Close the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()

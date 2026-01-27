"""
API caller tool for making HTTP requests.
Supports REST APIs with authentication and retry logic.
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
import json
import time

import aiohttp

from ..base_tool import BaseTool, ToolConfig, ToolParameter, ToolCategory


class HTTPMethod(Enum):
    """Supported HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


@dataclass
class APIResponse:
    """Response from an API call."""
    status_code: int
    headers: Dict[str, str]
    body: Any
    elapsed_time: float
    success: bool
    error: Optional[str] = None


class APICallerTool(BaseTool):
    """
    Tool for making HTTP API calls.
    Supports authentication, retries, and various content types.
    """

    def __init__(self, default_headers: Optional[Dict[str, str]] = None,
                 base_url: Optional[str] = None):
        config = ToolConfig(
            name="api_caller",
            description="Make HTTP API requests to external services.",
            category=ToolCategory.ACTION,
            timeout=60.0,
            retry_attempts=3
        )
        super().__init__(config)
        self.default_headers = default_headers or {}
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.request_history: List[Dict[str, Any]] = []

        # Blocked domains for safety
        self.blocked_domains = [
            "localhost",
            "127.0.0.1",
            "0.0.0.0",
            "169.254.",  # Link-local
            "10.",       # Private
            "172.16.",   # Private
            "192.168.",  # Private
        ]

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="url",
                param_type="string",
                description="The URL to call",
                required=True
            ),
            ToolParameter(
                name="method",
                param_type="string",
                description="HTTP method",
                required=False,
                default="GET",
                enum_values=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
            ),
            ToolParameter(
                name="headers",
                param_type="object",
                description="Request headers",
                required=False,
                default={}
            ),
            ToolParameter(
                name="body",
                param_type="object",
                description="Request body (for POST, PUT, PATCH)",
                required=False
            ),
            ToolParameter(
                name="params",
                param_type="object",
                description="Query parameters",
                required=False,
                default={}
            ),
            ToolParameter(
                name="timeout",
                param_type="number",
                description="Request timeout in seconds",
                required=False,
                default=30.0,
                min_value=1,
                max_value=300
            ),
            ToolParameter(
                name="auth_type",
                param_type="string",
                description="Authentication type",
                required=False,
                enum_values=["none", "bearer", "basic", "api_key"]
            ),
            ToolParameter(
                name="auth_value",
                param_type="string",
                description="Authentication value (token, credentials, or API key)",
                required=False
            )
        ]

    def _validate_url(self, url: str) -> tuple[bool, Optional[str]]:
        """Validate URL for safety."""
        from urllib.parse import urlparse

        try:
            parsed = urlparse(url)

            # Check scheme
            if parsed.scheme not in ["http", "https"]:
                return False, f"Unsupported scheme: {parsed.scheme}"

            # Check for blocked domains
            host = parsed.hostname or ""
            for blocked in self.blocked_domains:
                if host.startswith(blocked) or blocked in host:
                    return False, f"Blocked domain: {host}"

            return True, None

        except Exception as e:
            return False, str(e)

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def _execute(self, url: str, method: str = "GET",
                      headers: Dict[str, str] = None,
                      body: Any = None,
                      params: Dict[str, str] = None,
                      timeout: float = 30.0,
                      auth_type: Optional[str] = None,
                      auth_value: Optional[str] = None) -> APIResponse:
        """Execute API call."""
        # Validate URL
        is_valid, error = self._validate_url(url)
        if not is_valid:
            return APIResponse(
                status_code=0,
                headers={},
                body=None,
                elapsed_time=0,
                success=False,
                error=error
            )

        # Build full URL
        if self.base_url and not url.startswith(("http://", "https://")):
            url = f"{self.base_url.rstrip('/')}/{url.lstrip('/')}"

        # Prepare headers
        request_headers = self.default_headers.copy()
        if headers:
            request_headers.update(headers)

        # Add authentication
        if auth_type and auth_value:
            if auth_type == "bearer":
                request_headers["Authorization"] = f"Bearer {auth_value}"
            elif auth_type == "basic":
                import base64
                encoded = base64.b64encode(auth_value.encode()).decode()
                request_headers["Authorization"] = f"Basic {encoded}"
            elif auth_type == "api_key":
                request_headers["X-API-Key"] = auth_value

        # Prepare body
        json_body = None
        data_body = None
        if body:
            if isinstance(body, (dict, list)):
                json_body = body
                if "Content-Type" not in request_headers:
                    request_headers["Content-Type"] = "application/json"
            else:
                data_body = body

        start_time = time.time()

        try:
            session = await self._get_session()

            async with session.request(
                method=method,
                url=url,
                headers=request_headers,
                params=params,
                json=json_body,
                data=data_body,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                # Parse response
                response_headers = dict(response.headers)
                content_type = response.content_type

                if "application/json" in content_type:
                    response_body = await response.json()
                else:
                    response_body = await response.text()

                elapsed = time.time() - start_time

                # Record request
                self._record_request(url, method, response.status, elapsed)

                return APIResponse(
                    status_code=response.status,
                    headers=response_headers,
                    body=response_body,
                    elapsed_time=elapsed,
                    success=200 <= response.status < 300
                )

        except asyncio.TimeoutError:
            return APIResponse(
                status_code=0,
                headers={},
                body=None,
                elapsed_time=timeout,
                success=False,
                error=f"Request timed out after {timeout}s"
            )
        except aiohttp.ClientError as e:
            return APIResponse(
                status_code=0,
                headers={},
                body=None,
                elapsed_time=time.time() - start_time,
                success=False,
                error=str(e)
            )
        except Exception as e:
            return APIResponse(
                status_code=0,
                headers={},
                body=None,
                elapsed_time=time.time() - start_time,
                success=False,
                error=str(e)
            )

    def _record_request(self, url: str, method: str, status: int, elapsed: float):
        """Record request in history."""
        self.request_history.append({
            "url": url,
            "method": method,
            "status_code": status,
            "elapsed_time": elapsed,
            "timestamp": time.time()
        })

        # Keep only last 100 requests
        if len(self.request_history) > 100:
            self.request_history = self.request_history[-100:]

    async def close(self):
        """Close HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()

    def get_request_history(self, limit: int = 10) -> List[Dict]:
        """Get recent request history."""
        return self.request_history[-limit:]


class GraphQLTool(BaseTool):
    """
    Tool for making GraphQL API calls.
    """

    def __init__(self, endpoint: str, default_headers: Optional[Dict[str, str]] = None):
        config = ToolConfig(
            name="graphql",
            description="Execute GraphQL queries and mutations.",
            category=ToolCategory.ACTION,
            timeout=60.0
        )
        super().__init__(config)
        self.endpoint = endpoint
        self.default_headers = default_headers or {}
        self.session: Optional[aiohttp.ClientSession] = None

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                param_type="string",
                description="GraphQL query or mutation",
                required=True
            ),
            ToolParameter(
                name="variables",
                param_type="object",
                description="Query variables",
                required=False,
                default={}
            ),
            ToolParameter(
                name="operation_name",
                param_type="string",
                description="Operation name (if multiple operations in query)",
                required=False
            )
        ]

    async def _execute(self, query: str, variables: Dict = None,
                      operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Execute GraphQL query."""
        if not self.session:
            self.session = aiohttp.ClientSession()

        headers = self.default_headers.copy()
        headers["Content-Type"] = "application/json"

        payload = {
            "query": query,
            "variables": variables or {}
        }
        if operation_name:
            payload["operationName"] = operation_name

        try:
            async with self.session.post(
                self.endpoint,
                headers=headers,
                json=payload
            ) as response:
                result = await response.json()

                if "errors" in result:
                    return {
                        "success": False,
                        "data": result.get("data"),
                        "errors": result["errors"]
                    }

                return {
                    "success": True,
                    "data": result.get("data"),
                    "errors": None
                }

        except Exception as e:
            return {
                "success": False,
                "data": None,
                "errors": [{"message": str(e)}]
            }

    async def close(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()

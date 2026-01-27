"""
Base tool interface and common tool functionality.
All tools inherit from BaseTool.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Union
from enum import Enum
import asyncio
import time
import uuid
from datetime import datetime


class ToolCategory(Enum):
    """Categories of tools."""
    DATA = "data"          # Read-only data retrieval
    ACTION = "action"      # Performs actions/mutations
    ORCHESTRATION = "orchestration"  # Coordinates other tools/agents


class ToolStatus(Enum):
    """Status of tool execution."""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    PENDING = "pending"


@dataclass
class ToolConfig:
    """Configuration for a tool."""
    name: str
    description: str
    category: ToolCategory = ToolCategory.DATA
    version: str = "1.0.0"
    timeout: float = 60.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    requires_confirmation: bool = False
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    param_type: str  # string, number, boolean, array, object
    description: str
    required: bool = True
    default: Any = None
    enum_values: Optional[List[Any]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None


@dataclass
class ToolResult:
    """Result of a tool execution."""
    tool_name: str
    status: ToolStatus
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def is_success(self) -> bool:
        return self.status == ToolStatus.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class BaseTool(ABC):
    """
    Base class for all tools.
    Provides common functionality and interface for tool implementation.
    """

    def __init__(self, config: ToolConfig):
        self.config = config
        self.tool_id = str(uuid.uuid4())
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.last_execution: Optional[datetime] = None
        self._hooks: Dict[str, List[Callable]] = {
            "pre_execute": [],
            "post_execute": [],
            "on_error": [],
        }

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def description(self) -> str:
        return self.config.description

    @property
    def category(self) -> ToolCategory:
        return self.config.category

    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        """Define the parameters this tool accepts."""
        pass

    @abstractmethod
    async def _execute(self, **kwargs) -> Any:
        """Actual tool execution logic. Override in subclasses."""
        pass

    def validate_parameters(self, **kwargs) -> tuple[bool, Optional[str]]:
        """Validate input parameters against schema."""
        params = {p.name: p for p in self.get_parameters()}

        # Check required parameters
        for name, param in params.items():
            if param.required and name not in kwargs:
                return False, f"Missing required parameter: {name}"

            if name in kwargs:
                value = kwargs[name]

                # Type validation
                if param.param_type == "string" and not isinstance(value, str):
                    return False, f"Parameter '{name}' must be a string"
                elif param.param_type == "number" and not isinstance(value, (int, float)):
                    return False, f"Parameter '{name}' must be a number"
                elif param.param_type == "boolean" and not isinstance(value, bool):
                    return False, f"Parameter '{name}' must be a boolean"
                elif param.param_type == "array" and not isinstance(value, list):
                    return False, f"Parameter '{name}' must be an array"

                # Range validation
                if param.min_value is not None and isinstance(value, (int, float)):
                    if value < param.min_value:
                        return False, f"Parameter '{name}' must be >= {param.min_value}"
                if param.max_value is not None and isinstance(value, (int, float)):
                    if value > param.max_value:
                        return False, f"Parameter '{name}' must be <= {param.max_value}"

                # Enum validation
                if param.enum_values and value not in param.enum_values:
                    return False, f"Parameter '{name}' must be one of: {param.enum_values}"

        return True, None

    def register_hook(self, event: str, callback: Callable) -> None:
        """Register a hook for tool events."""
        if event in self._hooks:
            self._hooks[event].append(callback)

    async def _trigger_hooks(self, event: str, **kwargs) -> None:
        """Trigger all hooks for an event."""
        for hook in self._hooks.get(event, []):
            if asyncio.iscoroutinefunction(hook):
                await hook(self, **kwargs)
            else:
                hook(self, **kwargs)

    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with given parameters.
        Handles validation, retries, timeout, and error handling.
        """
        start_time = time.time()

        # Validate parameters
        is_valid, error_msg = self.validate_parameters(**kwargs)
        if not is_valid:
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.FAILURE,
                error=error_msg,
                execution_time=time.time() - start_time
            )

        # Apply defaults
        params = self.get_parameters()
        for param in params:
            if param.name not in kwargs and param.default is not None:
                kwargs[param.name] = param.default

        # Trigger pre-execute hooks
        await self._trigger_hooks("pre_execute", parameters=kwargs)

        # Execute with retries
        last_error = None
        for attempt in range(self.config.retry_attempts):
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    self._execute(**kwargs),
                    timeout=self.config.timeout
                )

                execution_time = time.time() - start_time
                self.execution_count += 1
                self.total_execution_time += execution_time
                self.last_execution = datetime.now()

                tool_result = ToolResult(
                    tool_name=self.name,
                    status=ToolStatus.SUCCESS,
                    result=result,
                    execution_time=execution_time,
                    metadata={"attempt": attempt + 1}
                )

                await self._trigger_hooks("post_execute", result=tool_result)
                return tool_result

            except asyncio.TimeoutError:
                last_error = f"Tool execution timed out after {self.config.timeout}s"
                return ToolResult(
                    tool_name=self.name,
                    status=ToolStatus.TIMEOUT,
                    error=last_error,
                    execution_time=time.time() - start_time
                )

            except Exception as e:
                last_error = str(e)
                await self._trigger_hooks("on_error", error=e, attempt=attempt)

                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))

        # All retries failed
        return ToolResult(
            tool_name=self.name,
            status=ToolStatus.FAILURE,
            error=last_error,
            execution_time=time.time() - start_time,
            metadata={"attempts": self.config.retry_attempts}
        )

    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for the tool."""
        params = self.get_parameters()

        properties = {}
        required = []

        for param in params:
            prop = {
                "type": param.param_type,
                "description": param.description
            }

            if param.enum_values:
                prop["enum"] = param.enum_values
            if param.default is not None:
                prop["default"] = param.default
            if param.min_value is not None:
                prop["minimum"] = param.min_value
            if param.max_value is not None:
                prop["maximum"] = param.max_value

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }

    def get_openai_schema(self) -> Dict[str, Any]:
        """Get OpenAI function calling schema."""
        schema = self.get_schema()
        return {
            "type": "function",
            "function": schema
        }

    def get_anthropic_schema(self) -> Dict[str, Any]:
        """Get Anthropic tool use schema."""
        schema = self.get_schema()
        return {
            "name": schema["name"],
            "description": schema["description"],
            "input_schema": schema["parameters"]
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics."""
        return {
            "tool_name": self.name,
            "execution_count": self.execution_count,
            "total_execution_time": self.total_execution_time,
            "average_execution_time": self.total_execution_time / self.execution_count if self.execution_count > 0 else 0,
            "last_execution": self.last_execution.isoformat() if self.last_execution else None
        }

    def __repr__(self) -> str:
        return f"Tool({self.name}, category={self.category.value})"


class ToolRegistry:
    """
    Registry for managing tools.
    Provides tool discovery and lookup.
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        """Unregister a tool."""
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """List all registered tools."""
        return list(self._tools.keys())

    def list_by_category(self, category: ToolCategory) -> List[BaseTool]:
        """List tools by category."""
        return [t for t in self._tools.values() if t.category == category]

    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all tools."""
        return [t.get_schema() for t in self._tools.values()]

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """Get all tools in OpenAI format."""
        return [t.get_openai_schema() for t in self._tools.values()]

    def get_anthropic_tools(self) -> List[Dict[str, Any]]:
        """Get all tools in Anthropic format."""
        return [t.get_anthropic_schema() for t in self._tools.values()]


# Global tool registry
tool_registry = ToolRegistry()


def register_tool(tool: BaseTool) -> BaseTool:
    """Decorator/function to register a tool."""
    tool_registry.register(tool)
    return tool

# AI Terminal Agent - API Reference

Complete API documentation for the AI Terminal Agent system.

## Table of Contents

1. [Core Classes](#core-classes)
2. [Agent Classes](#agent-classes)
3. [Tool Classes](#tool-classes)
4. [Memory Classes](#memory-classes)
5. [Guardrail Classes](#guardrail-classes)
6. [Model Classes](#model-classes)
7. [Orchestration Classes](#orchestration-classes)
8. [Utility Classes](#utility-classes)

---

## Core Classes

### Agent

Base class for all agents.

```python
from src.core.agent import Agent, AgentConfig, AgentResponse

class Agent:
    def __init__(self, config: AgentConfig)
    async def execute(self, task: str, context: Context) -> AgentResponse
    async def think(self, task: str, context: Context) -> ThoughtStep
    async def act(self, thought: ThoughtStep, context: Context) -> Any
```

#### AgentConfig

```python
@dataclass
class AgentConfig:
    name: str
    model: str = "anthropic/claude-sonnet-4"
    system_prompt: Optional[str] = None
    tools: List[str] = field(default_factory=list)
    max_iterations: int = 10
    temperature: float = 0.7
    timeout: float = 300.0
```

#### AgentResponse

```python
@dataclass
class AgentResponse:
    content: str
    success: bool
    agent_name: str
    iterations: int
    thought_steps: List[ThoughtStep] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Context

Manages execution context and state.

```python
from src.core.context import Context, ContextManager

class Context:
    def __init__(self, session_id: Optional[str] = None)
    def set(self, key: str, value: Any) -> None
    def get(self, key: str, default: Any = None) -> Any
    def delete(self, key: str) -> bool
    def to_dict(self) -> Dict[str, Any]
```

### AgentRunner

Runs agents with configuration.

```python
from src.core.runner import AgentRunner, RunnerConfig

class AgentRunner:
    def __init__(self, config: RunnerConfig)
    async def run(self, agent: Agent, task: str, context: Context = None) -> Dict
    async def run_with_retry(self, agent: Agent, task: str, max_retries: int = 3) -> Dict
```

---

## Agent Classes

### ManagerAgent

```python
from src.agents.manager_agent import ManagerAgent

class ManagerAgent(Agent):
    def __init__(self, model_manager: ModelManager = None)
    async def execute(self, task: str, context: Context) -> AgentResponse
    def _route_task(self, task: str) -> str
    async def _delegate(self, agent_name: str, task: str, context: Context) -> Any
```

### TerminalAgent

```python
from src.agents.terminal_agent import TerminalAgent

class TerminalAgent(Agent):
    async def execute(self, task: str, context: Context) -> AgentResponse
    async def run_command(self, command: str, timeout: float = 60) -> Dict
    def _validate_command(self, command: str) -> bool
```

### CodeAgent

```python
from src.agents.code_agent import CodeAgent

class CodeAgent(Agent):
    async def execute(self, task: str, context: Context) -> AgentResponse
    async def generate_code(self, specification: str, language: str) -> str
    async def analyze_code(self, code: str) -> Dict
    async def refactor_code(self, code: str, instructions: str) -> str
```

### ResearchAgent

```python
from src.agents.research_agent import ResearchAgent

class ResearchAgent(Agent):
    async def execute(self, task: str, context: Context) -> AgentResponse
    async def search(self, query: str, num_results: int = 10) -> List[Dict]
    async def synthesize(self, sources: List[Dict]) -> str
```

### FileAgent

```python
from src.agents.file_agent import FileAgent

class FileAgent(Agent):
    async def execute(self, task: str, context: Context) -> AgentResponse
    async def read_file(self, path: str) -> str
    async def write_file(self, path: str, content: str) -> bool
    async def list_directory(self, path: str) -> List[str]
```

### DebugAgent

```python
from src.agents.debug_agent import DebugAgent

class DebugAgent(Agent):
    async def execute(self, task: str, context: Context) -> AgentResponse
    async def analyze_error(self, error: str, code: str = None) -> Dict
    async def suggest_fix(self, error_analysis: Dict) -> str
```

---

## Tool Classes

### BaseTool

```python
from src.tools.base_tool import BaseTool, ToolConfig, ToolResult

class BaseTool(ABC):
    def __init__(self, config: ToolConfig)
    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]
    @abstractmethod
    async def _execute(self, **kwargs) -> Dict[str, Any]
    async def execute(self, **kwargs) -> ToolResult
```

#### ToolConfig

```python
@dataclass
class ToolConfig:
    name: str
    description: str
    category: ToolCategory = ToolCategory.DATA
    timeout: float = 60.0
    requires_confirmation: bool = False
```

### WebSearchTool

```python
from src.tools.data_tools.web_search import WebSearchTool

class WebSearchTool(BaseTool):
    async def execute(self, query: str, num_results: int = 10) -> ToolResult
```

### FileReaderTool

```python
from src.tools.data_tools.file_reader import FileReaderTool

class FileReaderTool(BaseTool):
    async def execute(self, path: str, encoding: str = "utf-8") -> ToolResult
```

### TerminalExecutorTool

```python
from src.tools.action_tools.terminal_executor import TerminalExecutorTool

class TerminalExecutorTool(BaseTool):
    async def execute(self, command: str, timeout: float = 60) -> ToolResult
```

### CodeExecutorTool

```python
from src.tools.action_tools.code_executor import CodeExecutorTool

class CodeExecutorTool(BaseTool):
    async def execute(self, code: str, language: str, sandbox: bool = True) -> ToolResult
```

---

## Memory Classes

### VectorStore

```python
from src.memory.vector_store import VectorStore

class VectorStore:
    def __init__(self, embedding_provider: str = "sentence-transformers")
    async def add(self, texts: List[str], metadata: List[Dict] = None) -> List[str]
    async def search(self, query: str, top_k: int = 5) -> List[SearchResult]
    async def delete(self, ids: List[str]) -> bool
```

### ConversationMemory

```python
from src.memory.conversation_memory import ConversationMemory

class ConversationMemory:
    def __init__(self, max_messages: int = 100)
    def add_message(self, role: str, content: str) -> None
    def get_messages(self, limit: int = None) -> List[Message]
    def get_context_window(self, max_tokens: int = 4000) -> List[Message]
    async def summarize(self) -> str
```

### CacheManager

```python
from src.memory.cache_manager import CacheManager

class CacheManager:
    def __init__(self, backend: CacheBackend = None, default_ttl: int = None)
    async def get(self, key: str) -> Optional[Any]
    async def set(self, key: str, value: Any, ttl: int = None) -> None
    async def delete(self, key: str) -> bool
    async def clear(self) -> None
```

---

## Guardrail Classes

### InputGuardrail

```python
from src.guardrails.input_guardrails import InputGuardrail

class InputGuardrail:
    def __init__(self, max_length: int = 10000)
    async def validate(self, input_text: str) -> InputValidationResult
```

### OutputGuardrail

```python
from src.guardrails.output_guardrails import OutputGuardrail

class OutputGuardrail:
    async def validate(self, output: str) -> ValidationResult
    async def filter(self, output: str) -> FilteredOutput
```

### PIIFilter

```python
from src.guardrails.pii_filter import PIIFilter

class PIIFilter:
    def detect(self, text: str) -> List[PIIMatch]
    def redact(self, text: str) -> str
```

### SafetyClassifier

```python
from src.guardrails.safety_classifier import SafetyClassifier

class SafetyClassifier:
    def classify(self, text: str) -> ClassificationResult
    def is_safe(self, text: str) -> bool
```

---

## Model Classes

### ModelManager

```python
from src.models.model_manager import ModelManager

class ModelManager:
    def __init__(self, default_model: str = None)
    def get_client(self, model: str) -> BaseModelClient
    async def generate(self, request: GenerationRequest) -> GenerationResponse
    async def generate_stream(self, request: GenerationRequest) -> AsyncGenerator
```

#### GenerationRequest

```python
@dataclass
class GenerationRequest:
    prompt: str
    system: Optional[str] = None
    messages: List[Dict] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    stop_sequences: List[str] = None
```

### OpenAIClient

```python
from src.models.openai_client import OpenAIClient

class OpenAIClient(BaseModelClient):
    async def generate(self, request: GenerationRequest) -> GenerationResponse
    async def generate_stream(self, request: GenerationRequest) -> AsyncGenerator
```

### AnthropicClient

```python
from src.models.anthropic_client import AnthropicClient

class AnthropicClient(BaseModelClient):
    async def generate(self, request: GenerationRequest) -> GenerationResponse
    async def generate_stream(self, request: GenerationRequest) -> AsyncGenerator
```

---

## Orchestration Classes

### Workflow

```python
from src.orchestration.workflow import Workflow, WorkflowStep, WorkflowEngine

class Workflow:
    def __init__(self, name: str, description: str = "")
    def add_step(self, step: WorkflowStep) -> None
    def get_execution_order(self) -> List[str]

class WorkflowEngine:
    async def execute(self, workflow: Workflow, context: Context) -> WorkflowResult
```

### ParallelExecutor

```python
from src.orchestration.parallel_executor import ParallelExecutor

class ParallelExecutor:
    def __init__(self, max_workers: int = 5)
    async def execute_all(self, tasks: List[Coroutine]) -> List[Any]
```

### HandoffManager

```python
from src.orchestration.handoff_manager import HandoffManager

class HandoffManager:
    def register_agents(self, agents: Dict[str, Agent]) -> None
    async def handoff(self, request: HandoffRequest) -> HandoffResult
```

### StateMachine

```python
from src.orchestration.state_machine import StateMachine, State, Transition

class StateMachine:
    def __init__(self, initial_state: str)
    def add_state(self, state: State) -> None
    def add_transition(self, transition: Transition) -> None
    def trigger(self, event: str) -> bool
```

---

## Utility Classes

### Logger

```python
from src.utils.logger import Logger, setup_logging, get_logger

def setup_logging(level: str = "INFO", log_file: str = None) -> Logger
def get_logger(name: str) -> Logger

class Logger:
    def debug(self, message: str, **kwargs) -> None
    def info(self, message: str, **kwargs) -> None
    def warning(self, message: str, **kwargs) -> None
    def error(self, message: str, **kwargs) -> None
```

### Config

```python
from src.utils.config import Config, ConfigManager, load_config

def load_config(config_file: str = None, **overrides) -> Config

class ConfigManager:
    def load_yaml(self, path: str) -> ConfigManager
    def load_env(self, prefix: str = "AI_AGENT_") -> ConfigManager
    def build(self) -> Config
```

### ErrorHandler

```python
from src.utils.error_handler import ErrorHandler, AgentError

class ErrorHandler:
    def handle(self, error: Exception, context: Dict = None) -> ErrorRecord
    def get_stats(self) -> Dict[str, Any]

@dataclass
class AgentError(Exception):
    message: str
    code: str
    severity: ErrorSeverity
    recoverable: bool
```

### Telemetry

```python
from src.utils.telemetry import Telemetry, get_telemetry

class Telemetry:
    def record_event(self, event_type: EventType, name: str, data: Dict = None)
    def record_agent_start(self, agent_name: str, task: str)
    def record_agent_end(self, agent_name: str, success: bool, duration_ms: float)
```

---

## Usage Examples

### Basic Agent Execution

```python
import asyncio
from src.agents.manager_agent import ManagerAgent
from src.core.context import Context

async def main():
    agent = ManagerAgent()
    context = Context()

    result = await agent.execute("Create a hello world Python script", context)
    print(result.content)

asyncio.run(main())
```

### Using Tools

```python
from src.tools.data_tools.web_search import WebSearchTool

async def search():
    tool = WebSearchTool()
    result = await tool.execute(query="Python tutorials", num_results=5)

    if result.success:
        for item in result.output:
            print(f"- {item['title']}: {item['url']}")

asyncio.run(search())
```

### Custom Workflow

```python
from src.orchestration.workflow import Workflow, WorkflowStep, WorkflowEngine

workflow = Workflow(name="code_review")
workflow.add_step(WorkflowStep(
    step_id="analyze",
    name="Analyze Code",
    action="code_analysis"
))
workflow.add_step(WorkflowStep(
    step_id="review",
    name="Review",
    action="code_review",
    dependencies=["analyze"]
))

engine = WorkflowEngine()
result = await engine.execute(workflow, Context())
```

---

## Error Codes

| Code | Description |
|------|-------------|
| `AGENT_ERROR` | General agent error |
| `VALIDATION_ERROR` | Input validation failed |
| `CONFIG_ERROR` | Configuration error |
| `NETWORK_ERROR` | Network/API error |
| `RATE_LIMIT_ERROR` | Rate limit exceeded |
| `TIMEOUT_ERROR` | Operation timed out |
| `TOOL_ERROR` | Tool execution failed |
| `MODEL_ERROR` | Model API error |

# AI Terminal Agent - Agent Documentation

This document describes the multi-agent system architecture and individual agent capabilities.

## Overview

The AI Terminal Agent uses a multi-agent architecture where specialized agents handle different types of tasks. The Manager Agent acts as the central orchestrator, routing tasks to appropriate specialized agents.

## Agent Architecture

```
                    ┌─────────────────┐
                    │  Manager Agent  │
                    │  (Orchestrator) │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│Terminal Agent │    │  Code Agent   │    │Research Agent │
└───────────────┘    └───────────────┘    └───────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐    ┌───────────────┐
│  File Agent   │    │  Debug Agent  │
└───────────────┘    └───────────────┘
```

## Agents

### 1. Manager Agent

**Purpose**: Central orchestrator that understands user requests and delegates to specialized agents.

**Capabilities**:
- Task analysis and decomposition
- Agent selection and routing
- Result synthesis
- Context management across agents

**Configuration**:
```yaml
name: manager_agent
model: anthropic/claude-sonnet-4
max_iterations: 10
temperature: 0.7
```

**Example Tasks**:
- "Help me set up a new Python project" → Coordinates File, Terminal, and Code agents
- "Debug this error and fix it" → Routes to Debug then Code agent

### 2. Terminal Agent

**Purpose**: Executes shell commands and manages terminal operations.

**Capabilities**:
- Command execution (bash, powershell, cmd)
- Process management
- Environment setup
- Package installation

**Safety Features**:
- Blocked dangerous commands (rm -rf /, format, etc.)
- Confirmation required for sudo/admin operations
- Command validation before execution

**Configuration**:
```yaml
name: terminal_agent
model: anthropic/claude-sonnet-4
max_iterations: 5
temperature: 0.3
tools:
  - terminal_executor
  - file_reader
  - file_writer
```

**Example Tasks**:
- "Run npm install"
- "List all Python processes"
- "Create a new directory structure"

### 3. Code Agent

**Purpose**: Handles all code-related tasks including generation, analysis, and refactoring.

**Capabilities**:
- Code generation in multiple languages
- Code analysis and review
- Refactoring and optimization
- Test generation
- Bug fixing

**Supported Languages**:
- Python, JavaScript/TypeScript, Java, C/C++
- Go, Rust, Ruby, PHP
- SQL, HTML/CSS, Shell scripts

**Configuration**:
```yaml
name: code_agent
model: openai/gpt-5-codex
max_iterations: 10
temperature: 0.5
tools:
  - code_executor
  - file_reader
  - file_writer
  - web_search
```

**Example Tasks**:
- "Write a Python function to sort a list"
- "Refactor this code to use async/await"
- "Add unit tests for this class"

### 4. Research Agent

**Purpose**: Performs web research and information synthesis.

**Capabilities**:
- Web search
- Information extraction
- Source verification
- Report generation
- Documentation lookup

**Configuration**:
```yaml
name: research_agent
model: anthropic/claude-sonnet-4
max_iterations: 15
temperature: 0.7
tools:
  - web_search
  - api_caller
  - file_writer
```

**Example Tasks**:
- "Find best practices for API design"
- "Research Python async libraries"
- "Look up the documentation for React hooks"

### 5. File Agent

**Purpose**: Handles file system operations.

**Capabilities**:
- File reading and writing
- Directory management
- File search and organization
- Backup and restore
- Encoding handling

**Safety Features**:
- Protected paths (/etc, /usr, system directories)
- Automatic backups before modifications
- File size limits

**Configuration**:
```yaml
name: file_agent
model: anthropic/claude-sonnet-4
max_iterations: 5
temperature: 0.3
tools:
  - file_reader
  - file_writer
  - terminal_executor
```

**Example Tasks**:
- "Read the contents of config.json"
- "Create a new project structure"
- "Find all Python files with 'TODO' comments"

### 6. Debug Agent

**Purpose**: Analyzes errors and assists with debugging.

**Capabilities**:
- Error message analysis
- Stack trace parsing
- Root cause identification
- Fix suggestions
- Code repair

**Configuration**:
```yaml
name: debug_agent
model: openai/gpt-5-codex
max_iterations: 10
temperature: 0.5
tools:
  - code_executor
  - file_reader
  - terminal_executor
  - web_search
```

**Example Tasks**:
- "Debug this TypeError"
- "Why is my async function not working?"
- "Find and fix the memory leak"

## Agent Communication

### Handoff Protocol

Agents can hand off tasks to other agents when needed:

```python
# Manager to Code handoff
handoff_request = HandoffRequest(
    source_agent="manager",
    target_agent="code",
    task="Generate a sorting function",
    context=current_context
)
result = await handoff_manager.handoff(handoff_request)
```

### Context Sharing

Context is preserved across agent handoffs:

```python
context = Context()
context.set("language", "python")
context.set("project_root", "/path/to/project")
# Context available to all agents in the chain
```

## Routing Rules

The Manager Agent uses pattern matching for routing:

| Pattern | Target Agent |
|---------|--------------|
| run, execute, shell, command | Terminal |
| code, program, function, class | Code |
| search, find, research, lookup | Research |
| file, read, write, create | File |
| error, debug, fix, bug | Debug |

## Creating Custom Agents

To create a custom agent:

```python
from src.core.agent import Agent, AgentConfig

class CustomAgent(Agent):
    def __init__(self):
        config = AgentConfig(
            name="custom_agent",
            model="anthropic/claude-sonnet-4",
            tools=["tool1", "tool2"]
        )
        super().__init__(config)

    async def execute(self, task: str, context: Context) -> AgentResponse:
        # Custom execution logic
        pass
```

## Best Practices

1. **Task Clarity**: Provide clear, specific tasks to agents
2. **Context**: Include relevant context for better results
3. **Safety**: Enable appropriate guardrails for your use case
4. **Monitoring**: Track agent performance and handoffs
5. **Error Handling**: Implement proper error recovery

## Performance Tuning

| Parameter | Effect |
|-----------|--------|
| `temperature` | Higher = more creative, Lower = more deterministic |
| `max_iterations` | Limit agent loops |
| `timeout` | Prevent runaway executions |

## Troubleshooting

### Agent Not Responding
- Check API connectivity
- Verify model availability
- Review timeout settings

### Wrong Agent Selected
- Review routing patterns
- Provide clearer task descriptions
- Check context information

### Handoff Failures
- Verify agent registration
- Check max depth limits
- Review context preservation

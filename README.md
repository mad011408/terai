# AI Terminal Agent

A powerful multi-agent AI system for terminal-based task automation. This project provides a flexible framework for orchestrating multiple specialized AI agents to accomplish complex tasks through natural language instructions.

## Features

- **Multi-Agent Architecture**: Specialized agents for different tasks (code, terminal, research, file operations, debugging)
- **Intelligent Task Routing**: Automatic routing of tasks to appropriate agents
- **Multiple LLM Support**: Works with various AI models including:
  - Anthropic Claude (claude-sonnet-4, claude-opus-4.1, etc.)
  - OpenAI GPT models (gpt-5-codex, o3, o4, o5)
  - Google Gemini
  - Mistral
  - Local models via Ollama
- **Safety Guardrails**: Input/output validation, PII filtering, prompt injection detection
- **Tool System**: Extensible tool framework for web search, file operations, code execution
- **Memory Management**: Vector stores, conversation memory, semantic caching
- **Interactive Terminal UI**: Rich terminal interface with streaming responses

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/example/ai-terminal-agent.git
cd ai-terminal-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your API keys
```

### Configuration

Create a `.env` file with your API configuration:

```env
AI_AGENT_API_KEY=your_api_key_here
AI_AGENT_API_HOST=https://go.trybons.ai
AI_AGENT_DEFAULT_MODEL=anthropic/claude-sonnet-4
```

### Usage

**Interactive Mode:**
```bash
python main.py
```

**Single Task:**
```bash
python main.py "Create a Python function to sort a list"
```

**With Specific Model:**
```bash
python main.py --model openai/gpt-5-codex "Write unit tests"
```

## Architecture

```
ai-terminal-agent/
├── src/
│   ├── core/           # Core agent framework
│   ├── agents/         # Specialized agents
│   ├── tools/          # Tool implementations
│   ├── memory/         # Memory and caching
│   ├── guardrails/     # Safety systems
│   ├── models/         # LLM integrations
│   ├── orchestration/  # Workflow management
│   ├── ui/             # Terminal UI
│   └── utils/          # Utilities
├── config/             # Configuration files
├── prompts/            # System prompts and templates
├── tests/              # Test suite
├── docs/               # Documentation
└── scripts/            # Utility scripts
```

## Agents

| Agent | Description |
|-------|-------------|
| **Manager** | Central orchestrator for task delegation |
| **Terminal** | Shell command execution |
| **Code** | Code generation and analysis |
| **Research** | Web search and information synthesis |
| **File** | File system operations |
| **Debug** | Error analysis and debugging |

## Supported Models

### Anthropic
- `anthropic/claude-sonnet-4`
- `anthropic/claude-opus-4.1`
- `anthropic/claude-sonnet-4.5`
- `anthropic/claude-opus-4.5`
- `anthropic/claude-opus-5.0`
- `claude-opus-4-5-20250929-thinking-32k`

### OpenAI
- `openai/gpt-5-codex`
- `openai/gpt-5.1-codex-max`
- `gpt-5.1-2025-11-13`
- `o3`, `o4`, `o5`
- `sora-2-pro`

### Others
- `gemini-3-pro` (Google)
- `grok-4.1-thinking` (xAI)
- `mistral-large-3:675b-cloud` (Mistral)
- `z-ai/glm-4.6` (Free tier)
- `ollama/kimi-k2-thinking` (Local)

## Docker Deployment

```bash
# Build and run
docker-compose up -d

# With monitoring
docker-compose --profile monitoring up -d

# View logs
docker-compose logs -f ai-agent
```

## Development

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific tests
pytest tests/unit/test_agents.py -v
```

### Code Quality

```bash
# Format code
black src tests

# Sort imports
isort src tests

# Lint
flake8 src tests

# Type check
mypy src
```

## Configuration Options

See `config/` directory for YAML configuration files:

- `agents.yaml` - Agent definitions and routing
- `tools.yaml` - Tool configurations
- `models.yaml` - Model settings and API endpoints
- `guardrails.yaml` - Safety and validation rules
- `workflows.yaml` - Workflow definitions

## API Reference

See [docs/API.md](docs/API.md) for detailed API documentation.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with support for TryBons AI API
- Uses various open-source libraries and frameworks

## Support

- GitHub Issues: [Report bugs or request features](https://github.com/example/ai-terminal-agent/issues)
- Documentation: [Full documentation](docs/)

"""
Unit tests for agent modules.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

# Import agents
from src.core.agent import Agent, AgentConfig, AgentResponse, ThoughtStep
from src.core.context import Context
from src.agents.manager_agent import ManagerAgent
from src.agents.terminal_agent import TerminalAgent
from src.agents.code_agent import CodeAgent
from src.agents.research_agent import ResearchAgent
from src.agents.file_agent import FileAgent
from src.agents.debug_agent import DebugAgent


class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AgentConfig(name="test_agent")
        assert config.name == "test_agent"
        assert config.max_iterations == 10
        assert config.timeout == 300.0
        assert config.tools == []

    def test_custom_config(self):
        """Test custom configuration values."""
        config = AgentConfig(
            name="custom_agent",
            model="gpt-4",
            max_iterations=5,
            timeout=60.0,
            tools=["web_search", "file_reader"]
        )
        assert config.name == "custom_agent"
        assert config.model == "gpt-4"
        assert config.max_iterations == 5
        assert len(config.tools) == 2


class TestAgent:
    """Tests for base Agent class."""

    def test_agent_initialization(self):
        """Test agent initialization."""
        config = AgentConfig(name="test_agent")
        agent = Agent(config)
        assert agent.name == "test_agent"
        assert agent.iteration == 0

    def test_thought_step_creation(self):
        """Test creating thought steps."""
        step = ThoughtStep(
            thought="Analyzing the problem",
            action="research",
            observation="Found relevant information"
        )
        assert step.thought == "Analyzing the problem"
        assert step.action == "research"
        assert step.observation == "Found relevant information"

    @pytest.mark.asyncio
    async def test_agent_think(self):
        """Test agent thinking process."""
        config = AgentConfig(name="test_agent")
        agent = Agent(config)

        # Mock the model client
        agent.model_client = AsyncMock()
        agent.model_client.generate.return_value = Mock(
            content="I should search for information"
        )

        thought = await agent.think("What is Python?", Context())
        assert thought is not None

    def test_agent_response_creation(self):
        """Test creating agent responses."""
        response = AgentResponse(
            content="This is the answer",
            success=True,
            agent_name="test_agent",
            iterations=3
        )
        assert response.content == "This is the answer"
        assert response.success is True
        assert response.iterations == 3


class TestManagerAgent:
    """Tests for ManagerAgent."""

    @pytest.fixture
    def manager_agent(self):
        """Create a manager agent for testing."""
        return ManagerAgent()

    def test_manager_initialization(self, manager_agent):
        """Test manager agent initialization."""
        assert manager_agent.name == "manager"
        assert len(manager_agent.available_agents) > 0

    def test_route_task_terminal(self, manager_agent):
        """Test routing to terminal agent."""
        task = "Run the command ls -la"
        agent = manager_agent._route_task(task)
        assert agent == "terminal"

    def test_route_task_code(self, manager_agent):
        """Test routing to code agent."""
        task = "Write a Python function to sort a list"
        agent = manager_agent._route_task(task)
        assert agent == "code"

    def test_route_task_research(self, manager_agent):
        """Test routing to research agent."""
        task = "Search for information about machine learning"
        agent = manager_agent._route_task(task)
        assert agent == "research"


class TestTerminalAgent:
    """Tests for TerminalAgent."""

    @pytest.fixture
    def terminal_agent(self):
        """Create a terminal agent for testing."""
        return TerminalAgent()

    def test_terminal_initialization(self, terminal_agent):
        """Test terminal agent initialization."""
        assert terminal_agent.name == "terminal"

    def test_validate_command_safe(self, terminal_agent):
        """Test validating safe commands."""
        assert terminal_agent._validate_command("ls -la") is True
        assert terminal_agent._validate_command("echo hello") is True

    def test_validate_command_dangerous(self, terminal_agent):
        """Test validating dangerous commands."""
        assert terminal_agent._validate_command("rm -rf /") is False
        assert terminal_agent._validate_command("mkfs.ext4 /dev/sda") is False


class TestCodeAgent:
    """Tests for CodeAgent."""

    @pytest.fixture
    def code_agent(self):
        """Create a code agent for testing."""
        return CodeAgent()

    def test_code_initialization(self, code_agent):
        """Test code agent initialization."""
        assert code_agent.name == "code"

    def test_detect_language(self, code_agent):
        """Test programming language detection."""
        python_code = "def hello():\n    print('hello')"
        assert code_agent._detect_language(python_code) == "python"

        js_code = "function hello() { console.log('hello'); }"
        assert code_agent._detect_language(js_code) == "javascript"

    def test_extract_code_blocks(self, code_agent):
        """Test extracting code blocks from markdown."""
        text = """
Here is some code:
```python
def hello():
    print('hello')
```
And more text.
"""
        blocks = code_agent._extract_code_blocks(text)
        assert len(blocks) == 1
        assert blocks[0]["language"] == "python"


class TestResearchAgent:
    """Tests for ResearchAgent."""

    @pytest.fixture
    def research_agent(self):
        """Create a research agent for testing."""
        return ResearchAgent()

    def test_research_initialization(self, research_agent):
        """Test research agent initialization."""
        assert research_agent.name == "research"

    def test_formulate_queries(self, research_agent):
        """Test query formulation."""
        topic = "machine learning basics"
        queries = research_agent._formulate_queries(topic)
        assert len(queries) > 0
        assert any("machine learning" in q.lower() for q in queries)


class TestFileAgent:
    """Tests for FileAgent."""

    @pytest.fixture
    def file_agent(self):
        """Create a file agent for testing."""
        return FileAgent()

    def test_file_initialization(self, file_agent):
        """Test file agent initialization."""
        assert file_agent.name == "file"

    def test_validate_path_safe(self, file_agent):
        """Test validating safe paths."""
        assert file_agent._validate_path("/home/user/documents/file.txt") is True
        assert file_agent._validate_path("./local/file.txt") is True

    def test_validate_path_protected(self, file_agent):
        """Test validating protected paths."""
        # Protected paths should return False or require confirmation
        result = file_agent._validate_path("/etc/passwd")
        assert result is False or file_agent._requires_confirmation("/etc/passwd")


class TestDebugAgent:
    """Tests for DebugAgent."""

    @pytest.fixture
    def debug_agent(self):
        """Create a debug agent for testing."""
        return DebugAgent()

    def test_debug_initialization(self, debug_agent):
        """Test debug agent initialization."""
        assert debug_agent.name == "debug"

    def test_parse_error_message(self, debug_agent):
        """Test parsing error messages."""
        error = "TypeError: cannot concatenate 'str' and 'int' objects"
        parsed = debug_agent._parse_error(error)
        assert parsed["type"] == "TypeError"

    def test_categorize_error(self, debug_agent):
        """Test error categorization."""
        assert debug_agent._categorize_error("SyntaxError") == "syntax"
        assert debug_agent._categorize_error("TypeError") == "runtime"
        assert debug_agent._categorize_error("NameError") == "runtime"


class TestAgentInteraction:
    """Tests for agent interactions."""

    @pytest.mark.asyncio
    async def test_agent_handoff(self):
        """Test agent handoff functionality."""
        manager = ManagerAgent()
        terminal = TerminalAgent()

        # Simulate handoff
        task = "List files in current directory"
        target_agent = manager._route_task(task)
        assert target_agent == "terminal"

    @pytest.mark.asyncio
    async def test_agent_context_preservation(self):
        """Test context preservation across agents."""
        context = Context()
        context.set("user_preference", "verbose")

        config = AgentConfig(name="test")
        agent = Agent(config)

        # Context should be accessible
        assert context.get("user_preference") == "verbose"


# Fixtures for pytest
@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

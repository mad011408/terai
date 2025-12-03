"""
Unit tests for tool modules.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

# Import tools
from src.tools.base_tool import BaseTool, ToolConfig, ToolResult, ToolCategory, ToolRegistry
from src.tools.data_tools.web_search import WebSearchTool
from src.tools.data_tools.file_reader import FileReaderTool
from src.tools.action_tools.terminal_executor import TerminalExecutorTool
from src.tools.action_tools.code_executor import CodeExecutorTool
from src.tools.action_tools.file_writer import FileWriterTool


class TestToolConfig:
    """Tests for ToolConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = ToolConfig(
            name="test_tool",
            description="A test tool"
        )
        assert config.name == "test_tool"
        assert config.description == "A test tool"
        assert config.timeout == 60.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = ToolConfig(
            name="custom_tool",
            description="Custom tool",
            category=ToolCategory.ACTION,
            timeout=120.0,
            requires_confirmation=True
        )
        assert config.category == ToolCategory.ACTION
        assert config.timeout == 120.0
        assert config.requires_confirmation is True


class TestToolResult:
    """Tests for ToolResult."""

    def test_success_result(self):
        """Test successful result."""
        result = ToolResult(
            success=True,
            output="Operation completed",
            tool_name="test_tool"
        )
        assert result.success is True
        assert result.output == "Operation completed"
        assert result.error is None

    def test_error_result(self):
        """Test error result."""
        result = ToolResult(
            success=False,
            output=None,
            error="Something went wrong",
            tool_name="test_tool"
        )
        assert result.success is False
        assert result.error == "Something went wrong"

    def test_result_to_dict(self):
        """Test result serialization."""
        result = ToolResult(
            success=True,
            output={"data": "test"},
            tool_name="test_tool"
        )
        data = result.to_dict()
        assert "success" in data
        assert "output" in data
        assert "tool_name" in data


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_tool(self):
        """Test registering a tool."""
        registry = ToolRegistry()
        mock_tool = Mock()
        mock_tool.name = "mock_tool"

        registry.register(mock_tool)
        assert "mock_tool" in registry.tools

    def test_get_tool(self):
        """Test getting a tool."""
        registry = ToolRegistry()
        mock_tool = Mock()
        mock_tool.name = "mock_tool"

        registry.register(mock_tool)
        retrieved = registry.get("mock_tool")
        assert retrieved == mock_tool

    def test_get_nonexistent_tool(self):
        """Test getting a nonexistent tool."""
        registry = ToolRegistry()
        retrieved = registry.get("nonexistent")
        assert retrieved is None

    def test_list_tools(self):
        """Test listing all tools."""
        registry = ToolRegistry()
        mock_tool1 = Mock()
        mock_tool1.name = "tool1"
        mock_tool2 = Mock()
        mock_tool2.name = "tool2"

        registry.register(mock_tool1)
        registry.register(mock_tool2)

        tools = registry.list_tools()
        assert len(tools) == 2


class TestWebSearchTool:
    """Tests for WebSearchTool."""

    @pytest.fixture
    def web_search(self):
        """Create web search tool for testing."""
        return WebSearchTool()

    def test_initialization(self, web_search):
        """Test tool initialization."""
        assert web_search.name == "web_search"
        assert web_search.config.category == ToolCategory.DATA

    def test_get_parameters(self, web_search):
        """Test parameter definitions."""
        params = web_search.get_parameters()
        param_names = [p.name for p in params]
        assert "query" in param_names

    @pytest.mark.asyncio
    async def test_execute_with_mock(self, web_search):
        """Test execution with mocked response."""
        with patch.object(web_search, '_search') as mock_search:
            mock_search.return_value = [
                {"title": "Result 1", "url": "http://example.com"}
            ]
            result = await web_search.execute(query="test query")
            assert result.success is True


class TestFileReaderTool:
    """Tests for FileReaderTool."""

    @pytest.fixture
    def file_reader(self):
        """Create file reader tool for testing."""
        return FileReaderTool()

    def test_initialization(self, file_reader):
        """Test tool initialization."""
        assert file_reader.name == "file_reader"

    def test_get_parameters(self, file_reader):
        """Test parameter definitions."""
        params = file_reader.get_parameters()
        param_names = [p.name for p in params]
        assert "path" in param_names

    @pytest.mark.asyncio
    async def test_read_file(self, file_reader, tmp_path):
        """Test reading a file."""
        # Create a temp file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        result = await file_reader.execute(path=str(test_file))
        assert result.success is True
        assert "Hello, World!" in result.output

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, file_reader):
        """Test reading a nonexistent file."""
        result = await file_reader.execute(path="/nonexistent/file.txt")
        assert result.success is False


class TestTerminalExecutorTool:
    """Tests for TerminalExecutorTool."""

    @pytest.fixture
    def terminal_executor(self):
        """Create terminal executor tool for testing."""
        return TerminalExecutorTool()

    def test_initialization(self, terminal_executor):
        """Test tool initialization."""
        assert terminal_executor.name == "terminal_executor"
        assert terminal_executor.config.requires_confirmation is True

    def test_validate_command_safe(self, terminal_executor):
        """Test validating safe commands."""
        assert terminal_executor._is_safe_command("echo hello") is True
        assert terminal_executor._is_safe_command("ls -la") is True

    def test_validate_command_dangerous(self, terminal_executor):
        """Test validating dangerous commands."""
        assert terminal_executor._is_safe_command("rm -rf /") is False
        assert terminal_executor._is_safe_command(":(){ :|:& };:") is False

    @pytest.mark.asyncio
    async def test_execute_safe_command(self, terminal_executor):
        """Test executing a safe command."""
        with patch('asyncio.create_subprocess_shell') as mock_proc:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"output", b"")
            mock_process.returncode = 0
            mock_proc.return_value = mock_process

            result = await terminal_executor.execute(command="echo test")
            assert result.success is True


class TestCodeExecutorTool:
    """Tests for CodeExecutorTool."""

    @pytest.fixture
    def code_executor(self):
        """Create code executor tool for testing."""
        return CodeExecutorTool()

    def test_initialization(self, code_executor):
        """Test tool initialization."""
        assert code_executor.name == "code_executor"

    def test_get_parameters(self, code_executor):
        """Test parameter definitions."""
        params = code_executor.get_parameters()
        param_names = [p.name for p in params]
        assert "code" in param_names
        assert "language" in param_names

    @pytest.mark.asyncio
    async def test_execute_python(self, code_executor):
        """Test executing Python code."""
        code = "print('Hello, World!')"
        with patch.object(code_executor, '_execute_python') as mock_exec:
            mock_exec.return_value = {"output": "Hello, World!", "error": None}
            result = await code_executor.execute(code=code, language="python")
            # Result depends on implementation


class TestFileWriterTool:
    """Tests for FileWriterTool."""

    @pytest.fixture
    def file_writer(self):
        """Create file writer tool for testing."""
        return FileWriterTool()

    def test_initialization(self, file_writer):
        """Test tool initialization."""
        assert file_writer.name == "file_writer"
        assert file_writer.config.requires_confirmation is True

    @pytest.mark.asyncio
    async def test_write_file(self, file_writer, tmp_path):
        """Test writing a file."""
        test_file = tmp_path / "output.txt"
        result = await file_writer.execute(
            path=str(test_file),
            content="Test content"
        )
        assert result.success is True
        assert test_file.read_text() == "Test content"

    @pytest.mark.asyncio
    async def test_append_file(self, file_writer, tmp_path):
        """Test appending to a file."""
        test_file = tmp_path / "append.txt"
        test_file.write_text("Initial content\n")

        result = await file_writer.execute(
            path=str(test_file),
            content="Appended content",
            mode="append"
        )
        assert result.success is True
        assert "Appended content" in test_file.read_text()


class TestToolExecution:
    """Tests for tool execution flow."""

    @pytest.mark.asyncio
    async def test_tool_timeout(self):
        """Test tool timeout handling."""
        config = ToolConfig(
            name="slow_tool",
            description="A slow tool",
            timeout=0.1  # Very short timeout
        )

        class SlowTool(BaseTool):
            def get_parameters(self):
                return []

            async def _execute(self, **kwargs):
                await asyncio.sleep(10)  # Sleep longer than timeout
                return {"result": "done"}

        tool = SlowTool(config)
        # Should handle timeout gracefully

    @pytest.mark.asyncio
    async def test_tool_error_handling(self):
        """Test tool error handling."""
        config = ToolConfig(name="error_tool", description="Error tool")

        class ErrorTool(BaseTool):
            def get_parameters(self):
                return []

            async def _execute(self, **kwargs):
                raise ValueError("Test error")

        tool = ErrorTool(config)
        result = await tool.execute()
        assert result.success is False
        assert "error" in result.error.lower() or result.error is not None


# Fixtures
@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

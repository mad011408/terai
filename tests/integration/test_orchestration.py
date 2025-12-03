"""
Integration tests for agent orchestration.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

# Import orchestration components
from src.orchestration.handoff_manager import HandoffManager, HandoffRequest, HandoffResult
from src.core.context import Context, ContextManager
from src.core.runner import AgentRunner, RunnerConfig
from src.agents.manager_agent import ManagerAgent
from src.agents.terminal_agent import TerminalAgent
from src.agents.code_agent import CodeAgent


class TestHandoffManager:
    """Tests for agent handoff management."""

    @pytest.fixture
    def handoff_manager(self):
        """Create handoff manager for testing."""
        return HandoffManager()

    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for testing."""
        manager = Mock(spec=ManagerAgent)
        manager.name = "manager"

        terminal = Mock(spec=TerminalAgent)
        terminal.name = "terminal"

        code = Mock(spec=CodeAgent)
        code.name = "code"

        return {"manager": manager, "terminal": terminal, "code": code}

    @pytest.mark.asyncio
    async def test_simple_handoff(self, handoff_manager, mock_agents):
        """Test simple agent handoff."""
        handoff_manager.register_agents(mock_agents)

        request = HandoffRequest(
            source_agent="manager",
            target_agent="terminal",
            task="Run ls command",
            context=Context()
        )

        # Mock target agent execution
        mock_agents["terminal"].execute = AsyncMock(return_value={
            "success": True,
            "output": "file1.txt\nfile2.txt"
        })

        result = await handoff_manager.handoff(request)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_handoff_chain(self, handoff_manager, mock_agents):
        """Test chain of handoffs."""
        handoff_manager.register_agents(mock_agents)

        # Manager -> Code -> Terminal
        context = Context()

        # First handoff
        request1 = HandoffRequest(
            source_agent="manager",
            target_agent="code",
            task="Generate script",
            context=context
        )
        mock_agents["code"].execute = AsyncMock(return_value={
            "success": True,
            "output": "print('hello')"
        })
        result1 = await handoff_manager.handoff(request1)

        # Second handoff
        request2 = HandoffRequest(
            source_agent="code",
            target_agent="terminal",
            task="Run script",
            context=context
        )
        mock_agents["terminal"].execute = AsyncMock(return_value={
            "success": True,
            "output": "hello"
        })
        result2 = await handoff_manager.handoff(request2)

        assert result1.success is True
        assert result2.success is True

    @pytest.mark.asyncio
    async def test_handoff_context_preservation(self, handoff_manager, mock_agents):
        """Test that context is preserved during handoffs."""
        handoff_manager.register_agents(mock_agents)

        context = Context()
        context.set("important_data", "should_be_preserved")

        request = HandoffRequest(
            source_agent="manager",
            target_agent="terminal",
            task="Test task",
            context=context
        )

        mock_agents["terminal"].execute = AsyncMock(return_value={
            "success": True,
            "output": "done"
        })

        await handoff_manager.handoff(request)

        # Context should still have the data
        assert context.get("important_data") == "should_be_preserved"

    @pytest.mark.asyncio
    async def test_handoff_max_depth(self, handoff_manager, mock_agents):
        """Test maximum handoff depth limit."""
        handoff_manager.register_agents(mock_agents)
        handoff_manager.max_depth = 3

        context = Context()

        # Simulate deep handoff chain
        for i in range(5):
            request = HandoffRequest(
                source_agent="manager",
                target_agent="terminal",
                task=f"Task {i}",
                context=context
            )

            mock_agents["terminal"].execute = AsyncMock(return_value={
                "success": True,
                "output": f"Result {i}"
            })

            result = await handoff_manager.handoff(request)

            # Should fail after max depth
            if i >= handoff_manager.max_depth:
                assert result.success is False


class TestAgentRunner:
    """Tests for agent runner."""

    @pytest.fixture
    def runner(self):
        """Create agent runner for testing."""
        config = RunnerConfig(
            max_iterations=5,
            timeout=30.0
        )
        return AgentRunner(config)

    @pytest.mark.asyncio
    async def test_run_single_agent(self, runner):
        """Test running a single agent."""
        mock_agent = Mock()
        mock_agent.name = "test_agent"
        mock_agent.execute = AsyncMock(return_value={
            "success": True,
            "content": "Task completed"
        })

        result = await runner.run(mock_agent, "Test task")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_run_with_timeout(self, runner):
        """Test running with timeout."""
        mock_agent = Mock()
        mock_agent.name = "slow_agent"

        async def slow_execute(task):
            await asyncio.sleep(60)
            return {"success": True}

        mock_agent.execute = slow_execute

        runner.config.timeout = 0.1

        # Should timeout
        with pytest.raises(asyncio.TimeoutError):
            await runner.run(mock_agent, "Slow task")

    @pytest.mark.asyncio
    async def test_run_with_max_iterations(self, runner):
        """Test running with iteration limit."""
        mock_agent = Mock()
        mock_agent.name = "iterating_agent"

        iterations = 0

        async def iterating_execute(task):
            nonlocal iterations
            iterations += 1
            if iterations < 10:
                return {"success": False, "continue": True}
            return {"success": True}

        mock_agent.execute = iterating_execute
        runner.config.max_iterations = 3

        result = await runner.run(mock_agent, "Iterating task")

        # Should stop after max iterations
        assert iterations <= runner.config.max_iterations


class TestContextManager:
    """Tests for context management during orchestration."""

    @pytest.fixture
    def context_manager(self):
        """Create context manager for testing."""
        return ContextManager()

    def test_create_context(self, context_manager):
        """Test creating a new context."""
        context = context_manager.create_context("session_1")
        assert context is not None
        assert context.session_id == "session_1"

    def test_context_inheritance(self, context_manager):
        """Test context inheritance."""
        parent = context_manager.create_context("parent")
        parent.set("inherited_value", "from_parent")

        child = context_manager.create_child_context(parent, "child")
        assert child.get("inherited_value") == "from_parent"

    def test_context_isolation(self, context_manager):
        """Test context isolation."""
        ctx1 = context_manager.create_context("ctx1")
        ctx2 = context_manager.create_context("ctx2")

        ctx1.set("value", "context_1")
        ctx2.set("value", "context_2")

        assert ctx1.get("value") == "context_1"
        assert ctx2.get("value") == "context_2"


class TestMultiAgentOrchestration:
    """Tests for multi-agent orchestration scenarios."""

    @pytest.mark.asyncio
    async def test_collaborative_task(self):
        """Test multiple agents collaborating on a task."""
        handoff_manager = HandoffManager()

        # Create mock agents
        manager = Mock(spec=ManagerAgent)
        manager.name = "manager"

        code = Mock(spec=CodeAgent)
        code.name = "code"

        terminal = Mock(spec=TerminalAgent)
        terminal.name = "terminal"

        handoff_manager.register_agents({
            "manager": manager,
            "code": code,
            "terminal": terminal
        })

        # Simulate collaborative task
        context = Context()
        context.set("task", "Create and run a Python script")

        # Step 1: Manager analyzes and delegates to code agent
        code.execute = AsyncMock(return_value={
            "success": True,
            "output": "script.py",
            "code": "print('Hello World')"
        })

        request1 = HandoffRequest(
            source_agent="manager",
            target_agent="code",
            task="Generate Python script",
            context=context
        )
        result1 = await handoff_manager.handoff(request1)
        assert result1.success is True

        # Step 2: Handoff to terminal to run the script
        terminal.execute = AsyncMock(return_value={
            "success": True,
            "output": "Hello World"
        })

        request2 = HandoffRequest(
            source_agent="code",
            target_agent="terminal",
            task="Run script.py",
            context=context
        )
        result2 = await handoff_manager.handoff(request2)
        assert result2.success is True

    @pytest.mark.asyncio
    async def test_error_recovery_orchestration(self):
        """Test error recovery in orchestration."""
        handoff_manager = HandoffManager()

        manager = Mock(spec=ManagerAgent)
        manager.name = "manager"

        terminal = Mock(spec=TerminalAgent)
        terminal.name = "terminal"

        handoff_manager.register_agents({
            "manager": manager,
            "terminal": terminal
        })

        context = Context()

        # First attempt fails
        attempt = 0

        async def failing_then_success(task):
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                raise Exception("Command failed")
            return {"success": True, "output": "success"}

        terminal.execute = failing_then_success

        # First attempt
        request = HandoffRequest(
            source_agent="manager",
            target_agent="terminal",
            task="Run command",
            context=context
        )

        try:
            result1 = await handoff_manager.handoff(request)
        except Exception:
            pass

        # Retry
        result2 = await handoff_manager.handoff(request)
        assert result2.success is True

    @pytest.mark.asyncio
    async def test_parallel_agent_execution(self):
        """Test parallel execution of multiple agents."""
        from src.orchestration.parallel_executor import ParallelExecutor

        executor = ParallelExecutor(max_workers=3)

        # Create mock agent tasks
        async def agent_task(agent_id):
            await asyncio.sleep(0.05)
            return {"agent": agent_id, "result": "success"}

        tasks = [agent_task(i) for i in range(5)]
        results = await executor.execute_all(tasks)

        assert len(results) == 5
        assert all(r["result"] == "success" for r in results)


# Fixtures
@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

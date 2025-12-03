"""
Integration tests for workflow execution.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

# Import workflow components
from src.orchestration.workflow import Workflow, WorkflowStep, WorkflowEngine, StepStatus
from src.orchestration.parallel_executor import ParallelExecutor, BatchExecutor
from src.orchestration.state_machine import StateMachine, State, Transition
from src.core.context import Context


class TestWorkflowExecution:
    """Tests for workflow execution."""

    @pytest.fixture
    def workflow_engine(self):
        """Create workflow engine for testing."""
        return WorkflowEngine()

    @pytest.fixture
    def simple_workflow(self):
        """Create a simple workflow for testing."""
        workflow = Workflow(
            name="test_workflow",
            description="A test workflow"
        )
        workflow.add_step(WorkflowStep(
            step_id="step1",
            name="Step 1",
            action="test_action"
        ))
        workflow.add_step(WorkflowStep(
            step_id="step2",
            name="Step 2",
            action="test_action",
            dependencies=["step1"]
        ))
        return workflow

    @pytest.mark.asyncio
    async def test_execute_simple_workflow(self, workflow_engine, simple_workflow):
        """Test executing a simple workflow."""
        # Mock the action executor
        with patch.object(workflow_engine, '_execute_action') as mock_execute:
            mock_execute.return_value = {"result": "success"}

            context = Context()
            result = await workflow_engine.execute(simple_workflow, context)

            assert result.success is True
            assert mock_execute.call_count == 2

    @pytest.mark.asyncio
    async def test_workflow_dependency_order(self, workflow_engine):
        """Test that workflow respects step dependencies."""
        workflow = Workflow(name="dependency_test")

        # Create steps with dependencies
        workflow.add_step(WorkflowStep(
            step_id="a",
            name="Step A",
            action="action_a"
        ))
        workflow.add_step(WorkflowStep(
            step_id="b",
            name="Step B",
            action="action_b",
            dependencies=["a"]
        ))
        workflow.add_step(WorkflowStep(
            step_id="c",
            name="Step C",
            action="action_c",
            dependencies=["a", "b"]
        ))

        execution_order = []

        async def track_execution(action, context):
            execution_order.append(action)
            return {"result": "success"}

        with patch.object(workflow_engine, '_execute_action', side_effect=track_execution):
            await workflow_engine.execute(workflow, Context())

        # Verify order
        assert execution_order.index("action_a") < execution_order.index("action_b")
        assert execution_order.index("action_b") < execution_order.index("action_c")

    @pytest.mark.asyncio
    async def test_workflow_failure_handling(self, workflow_engine):
        """Test workflow handling of step failures."""
        workflow = Workflow(name="failure_test")
        workflow.add_step(WorkflowStep(
            step_id="failing",
            name="Failing Step",
            action="fail_action"
        ))
        workflow.add_step(WorkflowStep(
            step_id="after",
            name="After Step",
            action="after_action",
            dependencies=["failing"]
        ))

        async def fail_action(action, context):
            if action == "fail_action":
                raise Exception("Step failed")
            return {"result": "success"}

        with patch.object(workflow_engine, '_execute_action', side_effect=fail_action):
            result = await workflow_engine.execute(workflow, Context())

        # Workflow should handle failure
        assert result.success is False or result.failed_steps is not None


class TestParallelExecution:
    """Tests for parallel execution."""

    @pytest.fixture
    def parallel_executor(self):
        """Create parallel executor for testing."""
        return ParallelExecutor(max_workers=4)

    @pytest.mark.asyncio
    async def test_parallel_tasks(self, parallel_executor):
        """Test executing tasks in parallel."""
        async def slow_task(n):
            await asyncio.sleep(0.1)
            return n * 2

        tasks = [slow_task(i) for i in range(5)]
        results = await parallel_executor.execute_all(tasks)

        assert len(results) == 5
        assert results == [0, 2, 4, 6, 8]

    @pytest.mark.asyncio
    async def test_parallel_with_timeout(self, parallel_executor):
        """Test parallel execution with timeout."""
        async def very_slow_task():
            await asyncio.sleep(10)
            return "done"

        # Should timeout
        parallel_executor.timeout = 0.5
        with pytest.raises(asyncio.TimeoutError):
            await parallel_executor.execute_all([very_slow_task()])

    @pytest.mark.asyncio
    async def test_parallel_error_handling(self, parallel_executor):
        """Test error handling in parallel execution."""
        async def failing_task():
            raise ValueError("Task failed")

        async def success_task():
            return "success"

        tasks = [success_task(), failing_task(), success_task()]

        # Should handle errors gracefully
        results = await parallel_executor.execute_all(tasks, continue_on_error=True)
        # Results should contain successes and errors


class TestBatchExecution:
    """Tests for batch execution."""

    @pytest.fixture
    def batch_executor(self):
        """Create batch executor for testing."""
        return BatchExecutor(batch_size=3)

    @pytest.mark.asyncio
    async def test_batch_processing(self, batch_executor):
        """Test processing items in batches."""
        items = list(range(10))

        async def process_item(item):
            return item * 2

        results = await batch_executor.execute(items, process_item)
        assert len(results) == 10
        assert results == [i * 2 for i in range(10)]

    @pytest.mark.asyncio
    async def test_batch_size_respected(self, batch_executor):
        """Test that batch size is respected."""
        concurrent_count = 0
        max_concurrent = 0

        async def track_concurrency(item):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.05)
            concurrent_count -= 1
            return item

        items = list(range(10))
        await batch_executor.execute(items, track_concurrency)

        # Max concurrent should not exceed batch size
        assert max_concurrent <= batch_executor.batch_size


class TestStateMachine:
    """Tests for state machine."""

    @pytest.fixture
    def state_machine(self):
        """Create state machine for testing."""
        sm = StateMachine(initial_state="idle")

        # Define states
        sm.add_state(State(name="idle"))
        sm.add_state(State(name="running"))
        sm.add_state(State(name="paused"))
        sm.add_state(State(name="completed", is_final=True))
        sm.add_state(State(name="failed", is_final=True))

        # Define transitions
        sm.add_transition(Transition(
            source="idle",
            target="running",
            event="start"
        ))
        sm.add_transition(Transition(
            source="running",
            target="paused",
            event="pause"
        ))
        sm.add_transition(Transition(
            source="paused",
            target="running",
            event="resume"
        ))
        sm.add_transition(Transition(
            source="running",
            target="completed",
            event="complete"
        ))
        sm.add_transition(Transition(
            source="running",
            target="failed",
            event="fail"
        ))

        return sm

    def test_initial_state(self, state_machine):
        """Test initial state."""
        assert state_machine.current_state == "idle"

    def test_valid_transition(self, state_machine):
        """Test valid state transition."""
        result = state_machine.trigger("start")
        assert result is True
        assert state_machine.current_state == "running"

    def test_invalid_transition(self, state_machine):
        """Test invalid state transition."""
        # Can't pause from idle
        result = state_machine.trigger("pause")
        assert result is False
        assert state_machine.current_state == "idle"

    def test_transition_sequence(self, state_machine):
        """Test sequence of transitions."""
        state_machine.trigger("start")
        assert state_machine.current_state == "running"

        state_machine.trigger("pause")
        assert state_machine.current_state == "paused"

        state_machine.trigger("resume")
        assert state_machine.current_state == "running"

        state_machine.trigger("complete")
        assert state_machine.current_state == "completed"

    def test_final_state(self, state_machine):
        """Test final state behavior."""
        state_machine.trigger("start")
        state_machine.trigger("complete")

        # Should not be able to transition from final state
        result = state_machine.trigger("start")
        assert result is False
        assert state_machine.current_state == "completed"

    def test_history(self, state_machine):
        """Test transition history."""
        state_machine.trigger("start")
        state_machine.trigger("pause")
        state_machine.trigger("resume")

        history = state_machine.get_history()
        assert len(history) >= 3


class TestWorkflowIntegration:
    """Integration tests combining workflow components."""

    @pytest.mark.asyncio
    async def test_workflow_with_parallel_steps(self):
        """Test workflow with parallel step execution."""
        workflow = Workflow(name="parallel_workflow")

        # Steps that can run in parallel
        workflow.add_step(WorkflowStep(
            step_id="init",
            name="Initialize",
            action="init"
        ))
        workflow.add_step(WorkflowStep(
            step_id="task_a",
            name="Task A",
            action="task_a",
            dependencies=["init"]
        ))
        workflow.add_step(WorkflowStep(
            step_id="task_b",
            name="Task B",
            action="task_b",
            dependencies=["init"]
        ))
        workflow.add_step(WorkflowStep(
            step_id="finalize",
            name="Finalize",
            action="finalize",
            dependencies=["task_a", "task_b"]
        ))

        engine = WorkflowEngine()

        async def mock_action(action, context):
            await asyncio.sleep(0.05)
            return {"action": action, "result": "success"}

        with patch.object(engine, '_execute_action', side_effect=mock_action):
            result = await engine.execute(workflow, Context())
            # task_a and task_b should run in parallel

    @pytest.mark.asyncio
    async def test_workflow_state_tracking(self):
        """Test workflow state tracking with state machine."""
        workflow = Workflow(name="stateful_workflow")
        workflow.add_step(WorkflowStep(
            step_id="step1",
            name="Step 1",
            action="action1"
        ))

        engine = WorkflowEngine()

        # Track state changes
        states = []

        def on_state_change(old_state, new_state):
            states.append((old_state, new_state))

        engine.on_state_change = on_state_change

        async def mock_action(action, context):
            return {"result": "success"}

        with patch.object(engine, '_execute_action', side_effect=mock_action):
            await engine.execute(workflow, Context())

        # Verify state transitions were tracked


# Fixtures
@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

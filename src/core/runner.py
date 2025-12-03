"""
Agent execution loop and orchestration.
Manages agent lifecycle and execution coordination.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Callable, Union
from enum import Enum
import uuid
from datetime import datetime
import traceback

from .agent import Agent, AgentConfig, AgentResponse, AgentState
from .context import Context, ContextManager
from .message import Message, MessageType


class RunnerState(Enum):
    """States for the agent runner."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class ExecutionMode(Enum):
    """Execution modes for the runner."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PIPELINE = "pipeline"
    REACTIVE = "reactive"


@dataclass
class RunnerConfig:
    """Configuration for the agent runner."""
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    max_concurrent_agents: int = 5
    global_timeout: float = 600.0
    retry_on_failure: bool = True
    max_retries: int = 3
    checkpoint_enabled: bool = True
    checkpoint_interval: int = 5
    telemetry_enabled: bool = True
    debug_mode: bool = False


@dataclass
class ExecutionPlan:
    """Plan for executing a task across agents."""
    plan_id: str
    task: str
    steps: List[Dict[str, Any]]
    current_step: int = 0
    status: str = "pending"
    results: List[Any] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Checkpoint:
    """Checkpoint for resumable execution."""
    checkpoint_id: str
    runner_id: str
    state: Dict[str, Any]
    context_snapshot: Dict[str, Any]
    timestamp: datetime
    step_index: int


class AgentRunner:
    """
    Orchestrates agent execution with support for multiple execution modes.
    Handles agent lifecycle, error recovery, and coordination.
    """

    def __init__(self, config: RunnerConfig = None):
        self.config = config or RunnerConfig()
        self.runner_id = str(uuid.uuid4())
        self.state = RunnerState.IDLE
        self.agents: Dict[str, Agent] = {}
        self.context_manager = ContextManager()
        self.execution_history: List[Dict[str, Any]] = []
        self.checkpoints: List[Checkpoint] = []
        self._current_plan: Optional[ExecutionPlan] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._callbacks: Dict[str, List[Callable]] = {
            "on_start": [],
            "on_step": [],
            "on_complete": [],
            "on_error": [],
            "on_checkpoint": []
        }

    def register_agent(self, agent: Agent) -> None:
        """Register an agent with the runner."""
        self.agents[agent.name] = agent

    def register_agents(self, agents: List[Agent]) -> None:
        """Register multiple agents."""
        for agent in agents:
            self.register_agent(agent)

    def get_agent(self, name: str) -> Optional[Agent]:
        """Get a registered agent by name."""
        return self.agents.get(name)

    def register_callback(self, event: str, callback: Callable) -> None:
        """Register a callback for runner events."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    async def _trigger_callbacks(self, event: str, **kwargs) -> None:
        """Trigger all callbacks for an event."""
        for callback in self._callbacks.get(event, []):
            if asyncio.iscoroutinefunction(callback):
                await callback(**kwargs)
            else:
                callback(**kwargs)

    async def run(self, task: str, agent_name: Optional[str] = None,
                  context: Optional[Context] = None) -> AgentResponse:
        """
        Run a task with the specified agent or auto-select based on task.
        """
        self.state = RunnerState.RUNNING
        await self._trigger_callbacks("on_start", task=task)

        try:
            # Select agent
            agent = self._select_agent(task, agent_name)
            if not agent:
                raise ValueError(f"No suitable agent found for task")

            # Create or use provided context
            ctx = context or self.context_manager.create_context()

            # Run the agent
            response = await self._execute_agent(agent, task, ctx)

            # Record execution
            self._record_execution(agent.name, task, response)

            self.state = RunnerState.IDLE
            await self._trigger_callbacks("on_complete", response=response)

            return response

        except Exception as e:
            self.state = RunnerState.ERROR
            await self._trigger_callbacks("on_error", error=e)
            raise

    async def run_pipeline(self, task: str, agent_sequence: List[str],
                           context: Optional[Context] = None) -> List[AgentResponse]:
        """
        Run a task through a sequence of agents (pipeline mode).
        Output of each agent becomes input for the next.
        """
        self.state = RunnerState.RUNNING
        ctx = context or self.context_manager.create_context()
        ctx.set("original_task", task)

        responses: List[AgentResponse] = []
        current_input = task

        for i, agent_name in enumerate(agent_sequence):
            agent = self.agents.get(agent_name)
            if not agent:
                raise ValueError(f"Agent '{agent_name}' not found")

            # Update context with pipeline info
            ctx.set("pipeline_step", i)
            ctx.set("pipeline_input", current_input)

            response = await self._execute_agent(agent, current_input, ctx)
            responses.append(response)

            # Use response as next input
            if response.success and response.result:
                current_input = str(response.result)
            else:
                break  # Stop pipeline on failure

            # Checkpoint if enabled
            if self.config.checkpoint_enabled and (i + 1) % self.config.checkpoint_interval == 0:
                await self._create_checkpoint(ctx, i)

        self.state = RunnerState.IDLE
        return responses

    async def run_parallel(self, task: str, agent_names: List[str],
                           context: Optional[Context] = None) -> List[AgentResponse]:
        """
        Run a task with multiple agents in parallel.
        Useful for getting multiple perspectives or parallel processing.
        """
        self.state = RunnerState.RUNNING
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_agents)

        ctx = context or self.context_manager.create_context()

        async def run_with_semaphore(agent_name: str) -> AgentResponse:
            async with self._semaphore:
                agent = self.agents.get(agent_name)
                if not agent:
                    raise ValueError(f"Agent '{agent_name}' not found")
                # Each parallel agent gets its own context copy
                agent_ctx = ctx.copy()
                return await self._execute_agent(agent, task, agent_ctx)

        tasks = [run_with_semaphore(name) for name in agent_names]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_responses = []
        for r in responses:
            if isinstance(r, AgentResponse):
                valid_responses.append(r)
            elif isinstance(r, Exception):
                # Create error response
                valid_responses.append(AgentResponse(
                    agent_id="error",
                    agent_name="unknown",
                    success=False,
                    result=None,
                    reasoning_trace=[],
                    messages=[],
                    error=str(r)
                ))

        self.state = RunnerState.IDLE
        return valid_responses

    async def run_plan(self, plan: ExecutionPlan) -> List[AgentResponse]:
        """
        Execute a pre-defined execution plan.
        """
        self._current_plan = plan
        self.state = RunnerState.RUNNING
        responses: List[AgentResponse] = []

        ctx = self.context_manager.create_context()
        ctx.set("plan_id", plan.plan_id)
        ctx.set("original_task", plan.task)

        for i, step in enumerate(plan.steps[plan.current_step:], start=plan.current_step):
            plan.current_step = i

            agent_name = step.get("agent")
            step_task = step.get("task", plan.task)
            step_config = step.get("config", {})

            agent = self.agents.get(agent_name)
            if not agent:
                continue

            await self._trigger_callbacks("on_step", step=i, agent=agent_name)

            response = await self._execute_agent(agent, step_task, ctx)
            responses.append(response)
            plan.results.append(response.result)

            # Update context with step results
            ctx.set(f"step_{i}_result", response.result)

            if not response.success and not step.get("continue_on_error", False):
                plan.status = "failed"
                break

            # Checkpoint
            if self.config.checkpoint_enabled and (i + 1) % self.config.checkpoint_interval == 0:
                await self._create_checkpoint(ctx, i)

        if plan.current_step >= len(plan.steps) - 1:
            plan.status = "completed"

        self.state = RunnerState.IDLE
        self._current_plan = None
        return responses

    async def _execute_agent(self, agent: Agent, task: str,
                             context: Context) -> AgentResponse:
        """Execute a single agent with error handling and retries."""
        last_error = None

        for attempt in range(self.config.max_retries if self.config.retry_on_failure else 1):
            try:
                response = await asyncio.wait_for(
                    agent.run(task, context),
                    timeout=self.config.global_timeout
                )
                return response
            except asyncio.TimeoutError:
                last_error = f"Agent execution timed out after {self.config.global_timeout}s"
            except Exception as e:
                last_error = str(e)
                if self.config.debug_mode:
                    traceback.print_exc()

            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

        return AgentResponse(
            agent_id=agent.agent_id,
            agent_name=agent.name,
            success=False,
            result=None,
            reasoning_trace=agent.reasoning_trace,
            messages=[],
            error=last_error
        )

    def _select_agent(self, task: str, agent_name: Optional[str] = None) -> Optional[Agent]:
        """Select the best agent for a task."""
        if agent_name:
            return self.agents.get(agent_name)

        # Simple keyword-based selection
        task_lower = task.lower()

        # Priority mapping
        agent_keywords = {
            "terminal_agent": ["command", "terminal", "shell", "bash", "run", "execute"],
            "code_agent": ["code", "program", "function", "class", "implement", "write code"],
            "research_agent": ["search", "find", "research", "lookup", "information"],
            "file_agent": ["file", "read", "write", "create", "delete", "directory"],
            "debug_agent": ["debug", "error", "fix", "bug", "troubleshoot"],
            "manager_agent": ["plan", "orchestrate", "coordinate", "manage"]
        }

        for agent_name, keywords in agent_keywords.items():
            if any(kw in task_lower for kw in keywords):
                if agent_name in self.agents:
                    return self.agents[agent_name]

        # Default to manager agent or first available
        return self.agents.get("manager_agent") or next(iter(self.agents.values()), None)

    def _record_execution(self, agent_name: str, task: str, response: AgentResponse) -> None:
        """Record execution for history and telemetry."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "task": task,
            "success": response.success,
            "execution_time": response.execution_time,
            "iterations": response.metadata.get("iterations", 0)
        }
        self.execution_history.append(record)

    async def _create_checkpoint(self, context: Context, step_index: int) -> Checkpoint:
        """Create a checkpoint for resumable execution."""
        checkpoint = Checkpoint(
            checkpoint_id=str(uuid.uuid4()),
            runner_id=self.runner_id,
            state={"current_plan": self._current_plan.__dict__ if self._current_plan else None},
            context_snapshot=context.to_dict(),
            timestamp=datetime.now(),
            step_index=step_index
        )
        self.checkpoints.append(checkpoint)
        await self._trigger_callbacks("on_checkpoint", checkpoint=checkpoint)
        return checkpoint

    async def resume_from_checkpoint(self, checkpoint: Checkpoint) -> List[AgentResponse]:
        """Resume execution from a checkpoint."""
        # Restore context
        ctx = Context()
        ctx.from_dict(checkpoint.context_snapshot)

        # Restore plan if exists
        if checkpoint.state.get("current_plan"):
            plan_data = checkpoint.state["current_plan"]
            plan = ExecutionPlan(
                plan_id=plan_data["plan_id"],
                task=plan_data["task"],
                steps=plan_data["steps"],
                current_step=checkpoint.step_index + 1,
                results=plan_data["results"]
            )
            return await self.run_plan(plan)

        return []

    def pause(self) -> None:
        """Pause execution."""
        self.state = RunnerState.PAUSED

    def resume(self) -> None:
        """Resume execution."""
        if self.state == RunnerState.PAUSED:
            self.state = RunnerState.RUNNING

    def stop(self) -> None:
        """Stop execution."""
        self.state = RunnerState.STOPPED

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        total_executions = len(self.execution_history)
        successful = sum(1 for e in self.execution_history if e["success"])

        return {
            "runner_id": self.runner_id,
            "state": self.state.value,
            "registered_agents": list(self.agents.keys()),
            "total_executions": total_executions,
            "successful_executions": successful,
            "success_rate": successful / total_executions if total_executions > 0 else 0,
            "checkpoints": len(self.checkpoints)
        }


class BatchRunner:
    """
    Runs multiple tasks in batch mode.
    """

    def __init__(self, runner: AgentRunner, batch_size: int = 10):
        self.runner = runner
        self.batch_size = batch_size

    async def run_batch(self, tasks: List[Dict[str, Any]]) -> List[AgentResponse]:
        """Run a batch of tasks."""
        responses = []

        for i in range(0, len(tasks), self.batch_size):
            batch = tasks[i:i + self.batch_size]

            # Run batch in parallel
            batch_tasks = [
                self.runner.run(
                    task=t["task"],
                    agent_name=t.get("agent"),
                    context=t.get("context")
                )
                for t in batch
            ]

            batch_responses = await asyncio.gather(*batch_tasks, return_exceptions=True)
            responses.extend(batch_responses)

        return responses

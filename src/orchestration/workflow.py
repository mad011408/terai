"""
Workflow definitions and execution engine.
"""

from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import asyncio
import uuid


class StepType(Enum):
    """Types of workflow steps."""
    AGENT = "agent"
    TOOL = "tool"
    CONDITION = "condition"
    PARALLEL = "parallel"
    LOOP = "loop"
    HUMAN_INPUT = "human_input"
    WAIT = "wait"


class StepStatus(Enum):
    """Status of a workflow step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """A single step in a workflow."""
    step_id: str
    name: str
    step_type: StepType
    config: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    condition: Optional[str] = None
    timeout: float = 300.0
    retry_count: int = 0
    max_retries: int = 3
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "name": self.name,
            "step_type": self.step_type.value,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
        }


@dataclass
class Workflow:
    """A workflow definition."""
    workflow_id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    variables: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Workflow":
        """Create workflow from dictionary."""
        steps = [
            WorkflowStep(
                step_id=s.get("step_id", str(uuid.uuid4())),
                name=s["name"],
                step_type=StepType(s["type"]),
                config=s.get("config", {}),
                dependencies=s.get("dependencies", []),
                condition=s.get("condition"),
                timeout=s.get("timeout", 300.0),
                max_retries=s.get("max_retries", 3)
            )
            for s in data.get("steps", [])
        ]

        return cls(
            workflow_id=data.get("id", str(uuid.uuid4())),
            name=data["name"],
            description=data.get("description", ""),
            steps=steps,
            variables=data.get("variables", {}),
            metadata=data.get("metadata", {})
        )

    def get_step(self, step_id: str) -> Optional[WorkflowStep]:
        """Get a step by ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_ready_steps(self) -> List[WorkflowStep]:
        """Get steps that are ready to run."""
        completed_ids = {s.step_id for s in self.steps if s.status == StepStatus.COMPLETED}

        ready = []
        for step in self.steps:
            if step.status != StepStatus.PENDING:
                continue
            if all(dep in completed_ids for dep in step.dependencies):
                ready.append(step)

        return ready


class WorkflowEngine:
    """
    Executes workflows with support for parallel execution and conditions.
    """

    def __init__(self, agents: Dict[str, Any] = None, tools: Dict[str, Any] = None):
        self.agents = agents or {}
        self.tools = tools or {}
        self.running_workflows: Dict[str, Workflow] = {}
        self.execution_history: List[Dict[str, Any]] = []

    async def run(self, workflow: Workflow, initial_input: Any = None) -> Dict[str, Any]:
        """
        Execute a workflow.

        Args:
            workflow: The workflow to execute
            initial_input: Initial input for the workflow

        Returns:
            Dictionary with results
        """
        workflow_id = workflow.workflow_id
        self.running_workflows[workflow_id] = workflow

        # Set initial input
        if initial_input:
            workflow.variables["input"] = initial_input

        results = {}
        errors = []

        try:
            while True:
                # Get steps ready to run
                ready_steps = workflow.get_ready_steps()

                if not ready_steps:
                    # Check if all steps completed
                    all_done = all(
                        s.status in [StepStatus.COMPLETED, StepStatus.SKIPPED, StepStatus.FAILED]
                        for s in workflow.steps
                    )
                    if all_done:
                        break

                    # Check for deadlock
                    pending = [s for s in workflow.steps if s.status == StepStatus.PENDING]
                    if pending:
                        errors.append("Workflow deadlock detected")
                        break

                # Execute ready steps in parallel
                tasks = [
                    self._execute_step(step, workflow)
                    for step in ready_steps
                ]

                step_results = await asyncio.gather(*tasks, return_exceptions=True)

                for step, result in zip(ready_steps, step_results):
                    if isinstance(result, Exception):
                        errors.append(f"Step {step.name} failed: {str(result)}")
                    else:
                        results[step.step_id] = result

        finally:
            del self.running_workflows[workflow_id]

        # Record execution
        execution_record = {
            "workflow_id": workflow_id,
            "workflow_name": workflow.name,
            "completed_at": datetime.now().isoformat(),
            "results": results,
            "errors": errors,
            "step_count": len(workflow.steps)
        }
        self.execution_history.append(execution_record)

        return {
            "success": len(errors) == 0,
            "results": results,
            "errors": errors,
            "variables": workflow.variables
        }

    async def _execute_step(self, step: WorkflowStep, workflow: Workflow) -> Any:
        """Execute a single workflow step."""
        # Check condition
        if step.condition:
            if not self._evaluate_condition(step.condition, workflow.variables):
                step.status = StepStatus.SKIPPED
                return None

        step.status = StepStatus.RUNNING
        step.started_at = datetime.now()

        try:
            result = await asyncio.wait_for(
                self._run_step(step, workflow),
                timeout=step.timeout
            )

            step.status = StepStatus.COMPLETED
            step.result = result
            step.completed_at = datetime.now()

            # Store result in workflow variables
            workflow.variables[f"step_{step.step_id}_result"] = result

            return result

        except asyncio.TimeoutError:
            step.status = StepStatus.FAILED
            step.error = f"Step timed out after {step.timeout}s"
            raise

        except Exception as e:
            step.retry_count += 1
            if step.retry_count < step.max_retries:
                # Retry
                step.status = StepStatus.PENDING
                return await self._execute_step(step, workflow)

            step.status = StepStatus.FAILED
            step.error = str(e)
            raise

    async def _run_step(self, step: WorkflowStep, workflow: Workflow) -> Any:
        """Run the actual step logic."""
        if step.step_type == StepType.AGENT:
            return await self._run_agent_step(step, workflow)
        elif step.step_type == StepType.TOOL:
            return await self._run_tool_step(step, workflow)
        elif step.step_type == StepType.PARALLEL:
            return await self._run_parallel_step(step, workflow)
        elif step.step_type == StepType.LOOP:
            return await self._run_loop_step(step, workflow)
        elif step.step_type == StepType.WAIT:
            return await self._run_wait_step(step, workflow)
        elif step.step_type == StepType.CONDITION:
            return self._evaluate_condition(step.config.get("expression", "True"), workflow.variables)
        else:
            raise ValueError(f"Unknown step type: {step.step_type}")

    async def _run_agent_step(self, step: WorkflowStep, workflow: Workflow) -> Any:
        """Run an agent step."""
        agent_name = step.config.get("agent")
        task = step.config.get("task", "")

        # Interpolate variables in task
        task = self._interpolate(task, workflow.variables)

        agent = self.agents.get(agent_name)
        if not agent:
            raise ValueError(f"Agent not found: {agent_name}")

        from ..core.context import Context
        ctx = Context()
        ctx.update(workflow.variables)

        response = await agent.run(task, ctx)
        return response.result if hasattr(response, 'result') else response

    async def _run_tool_step(self, step: WorkflowStep, workflow: Workflow) -> Any:
        """Run a tool step."""
        tool_name = step.config.get("tool")
        params = step.config.get("parameters", {})

        # Interpolate variables in parameters
        params = {k: self._interpolate(v, workflow.variables) for k, v in params.items()}

        tool = self.tools.get(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")

        result = await tool.execute(**params)
        return result.result if hasattr(result, 'result') else result

    async def _run_parallel_step(self, step: WorkflowStep, workflow: Workflow) -> List[Any]:
        """Run parallel sub-steps."""
        sub_steps = step.config.get("steps", [])

        tasks = []
        for sub_config in sub_steps:
            sub_step = WorkflowStep(
                step_id=str(uuid.uuid4()),
                name=sub_config.get("name", "parallel_sub"),
                step_type=StepType(sub_config.get("type", "tool")),
                config=sub_config.get("config", {})
            )
            tasks.append(self._run_step(sub_step, workflow))

        return await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_loop_step(self, step: WorkflowStep, workflow: Workflow) -> List[Any]:
        """Run a loop step."""
        items = step.config.get("items", [])
        item_var = step.config.get("item_variable", "item")
        body = step.config.get("body", {})

        # Resolve items if it's a variable reference
        if isinstance(items, str) and items.startswith("$"):
            items = workflow.variables.get(items[1:], [])

        results = []
        for item in items:
            workflow.variables[item_var] = item

            sub_step = WorkflowStep(
                step_id=str(uuid.uuid4()),
                name=f"loop_iteration",
                step_type=StepType(body.get("type", "tool")),
                config=body.get("config", {})
            )

            result = await self._run_step(sub_step, workflow)
            results.append(result)

        return results

    async def _run_wait_step(self, step: WorkflowStep, workflow: Workflow) -> None:
        """Run a wait step."""
        duration = step.config.get("duration", 1.0)
        await asyncio.sleep(duration)
        return None

    def _evaluate_condition(self, condition: str, variables: Dict[str, Any]) -> bool:
        """Evaluate a condition expression."""
        try:
            # Safe evaluation with limited scope
            return eval(condition, {"__builtins__": {}}, variables)
        except:
            return False

    def _interpolate(self, value: Any, variables: Dict[str, Any]) -> Any:
        """Interpolate variables in a value."""
        if isinstance(value, str):
            for var_name, var_value in variables.items():
                value = value.replace(f"${{{var_name}}}", str(var_value))
                value = value.replace(f"${var_name}", str(var_value))
            return value
        elif isinstance(value, dict):
            return {k: self._interpolate(v, variables) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._interpolate(v, variables) for v in value]
        return value

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a running workflow."""
        workflow = self.running_workflows.get(workflow_id)
        if not workflow:
            return None

        return {
            "workflow_id": workflow_id,
            "name": workflow.name,
            "steps": [s.to_dict() for s in workflow.steps],
            "variables": workflow.variables
        }

    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow."""
        if workflow_id in self.running_workflows:
            del self.running_workflows[workflow_id]
            return True
        return False

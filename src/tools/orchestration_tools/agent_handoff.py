"""
Agent handoff tool for transferring control between agents.
Enables decentralized agent coordination.
"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import uuid

from ..base_tool import BaseTool, ToolConfig, ToolParameter, ToolCategory


class HandoffReason(Enum):
    """Reasons for agent handoff."""
    TASK_DELEGATION = "task_delegation"
    EXPERTISE_REQUIRED = "expertise_required"
    TASK_COMPLETION = "task_completion"
    ERROR_ESCALATION = "error_escalation"
    USER_REQUEST = "user_request"
    RESOURCE_LIMIT = "resource_limit"


class HandoffStatus(Enum):
    """Status of a handoff."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class HandoffRequest:
    """A request to hand off to another agent."""
    handoff_id: str
    source_agent: str
    target_agent: str
    task: str
    reason: HandoffReason
    context: Dict[str, Any]
    status: HandoffStatus = HandoffStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None


class AgentHandoffTool(BaseTool):
    """
    Tool for handing off tasks between agents.
    Manages agent-to-agent communication and task transfer.
    """

    def __init__(self, agent_registry: Optional[Dict[str, Any]] = None):
        config = ToolConfig(
            name="agent_handoff",
            description="Hand off tasks to specialized agents.",
            category=ToolCategory.ORCHESTRATION,
            timeout=120.0
        )
        super().__init__(config)
        self.agent_registry = agent_registry or {}
        self.handoff_history: List[HandoffRequest] = []
        self.pending_handoffs: Dict[str, HandoffRequest] = {}
        self._callbacks: Dict[str, Callable] = {}

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="target_agent",
                param_type="string",
                description="Name of the agent to hand off to",
                required=True
            ),
            ToolParameter(
                name="task",
                param_type="string",
                description="Task description for the target agent",
                required=True
            ),
            ToolParameter(
                name="reason",
                param_type="string",
                description="Reason for the handoff",
                required=False,
                default="task_delegation",
                enum_values=["task_delegation", "expertise_required", "task_completion",
                            "error_escalation", "user_request", "resource_limit"]
            ),
            ToolParameter(
                name="context",
                param_type="object",
                description="Context data to pass to the target agent",
                required=False,
                default={}
            ),
            ToolParameter(
                name="wait_for_result",
                param_type="boolean",
                description="Whether to wait for the handoff to complete",
                required=False,
                default=True
            ),
            ToolParameter(
                name="timeout",
                param_type="number",
                description="Timeout for waiting (if wait_for_result is True)",
                required=False,
                default=60.0
            )
        ]

    def register_agent(self, name: str, agent: Any) -> None:
        """Register an agent in the registry."""
        self.agent_registry[name] = agent

    def unregister_agent(self, name: str) -> bool:
        """Unregister an agent."""
        if name in self.agent_registry:
            del self.agent_registry[name]
            return True
        return False

    def list_available_agents(self) -> List[str]:
        """List all available agents."""
        return list(self.agent_registry.keys())

    async def _execute(self, target_agent: str, task: str,
                      reason: str = "task_delegation",
                      context: Dict[str, Any] = None,
                      wait_for_result: bool = True,
                      timeout: float = 60.0) -> Dict[str, Any]:
        """Execute agent handoff."""
        # Check if target agent exists
        if target_agent not in self.agent_registry:
            available = self.list_available_agents()
            return {
                "success": False,
                "error": f"Agent '{target_agent}' not found. Available: {available}"
            }

        # Create handoff request
        handoff = HandoffRequest(
            handoff_id=str(uuid.uuid4()),
            source_agent="current",  # Will be set by calling agent
            target_agent=target_agent,
            task=task,
            reason=HandoffReason(reason),
            context=context or {}
        )

        self.pending_handoffs[handoff.handoff_id] = handoff
        self.handoff_history.append(handoff)

        if wait_for_result:
            # Execute and wait for result
            try:
                result = await self._execute_handoff(handoff, timeout)
                return {
                    "success": True,
                    "handoff_id": handoff.handoff_id,
                    "result": result,
                    "status": handoff.status.value
                }
            except Exception as e:
                handoff.status = HandoffStatus.FAILED
                handoff.error = str(e)
                return {
                    "success": False,
                    "handoff_id": handoff.handoff_id,
                    "error": str(e),
                    "status": handoff.status.value
                }
        else:
            # Fire and forget
            asyncio.create_task(self._execute_handoff(handoff, timeout))
            return {
                "success": True,
                "handoff_id": handoff.handoff_id,
                "status": "pending",
                "message": "Handoff initiated, not waiting for result"
            }

    async def _execute_handoff(self, handoff: HandoffRequest, timeout: float) -> Any:
        """Execute the actual handoff to the target agent."""
        agent = self.agent_registry.get(handoff.target_agent)
        if not agent:
            raise ValueError(f"Agent not found: {handoff.target_agent}")

        handoff.status = HandoffStatus.IN_PROGRESS

        try:
            # Check if agent has a run method
            if hasattr(agent, 'run'):
                # Create context for the agent
                from ...core.context import Context
                ctx = Context()
                ctx.update(handoff.context)
                ctx.set("handoff_id", handoff.handoff_id)
                ctx.set("handoff_source", handoff.source_agent)

                result = await asyncio.wait_for(
                    agent.run(handoff.task, ctx),
                    timeout=timeout
                )

                handoff.status = HandoffStatus.COMPLETED
                handoff.completed_at = datetime.now()
                handoff.result = result.result if hasattr(result, 'result') else result

                return handoff.result
            else:
                raise ValueError(f"Agent {handoff.target_agent} does not have a run method")

        except asyncio.TimeoutError:
            handoff.status = HandoffStatus.FAILED
            handoff.error = f"Handoff timed out after {timeout}s"
            raise

        except Exception as e:
            handoff.status = HandoffStatus.FAILED
            handoff.error = str(e)
            raise

        finally:
            # Remove from pending
            self.pending_handoffs.pop(handoff.handoff_id, None)

    def get_handoff_status(self, handoff_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a handoff."""
        # Check pending
        if handoff_id in self.pending_handoffs:
            handoff = self.pending_handoffs[handoff_id]
            return {
                "handoff_id": handoff_id,
                "status": handoff.status.value,
                "target_agent": handoff.target_agent,
                "task": handoff.task
            }

        # Check history
        for handoff in self.handoff_history:
            if handoff.handoff_id == handoff_id:
                return {
                    "handoff_id": handoff_id,
                    "status": handoff.status.value,
                    "target_agent": handoff.target_agent,
                    "task": handoff.task,
                    "result": handoff.result,
                    "error": handoff.error,
                    "completed_at": handoff.completed_at.isoformat() if handoff.completed_at else None
                }

        return None

    def get_handoff_history(self, limit: int = 10,
                           agent_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get handoff history."""
        history = self.handoff_history

        if agent_filter:
            history = [h for h in history
                      if h.source_agent == agent_filter or h.target_agent == agent_filter]

        return [
            {
                "handoff_id": h.handoff_id,
                "source_agent": h.source_agent,
                "target_agent": h.target_agent,
                "task": h.task[:100],
                "reason": h.reason.value,
                "status": h.status.value,
                "created_at": h.created_at.isoformat()
            }
            for h in history[-limit:]
        ]


class MultiAgentCoordinator:
    """
    Coordinates multiple agents for complex tasks.
    """

    def __init__(self, agents: Dict[str, Any]):
        self.agents = agents
        self.handoff_tool = AgentHandoffTool(agents)
        self.task_results: Dict[str, Any] = {}

    async def run_parallel(self, tasks: List[Dict[str, Any]]) -> List[Any]:
        """Run multiple agent tasks in parallel."""
        async def run_task(task_info: Dict) -> Any:
            agent_name = task_info["agent"]
            task = task_info["task"]
            context = task_info.get("context", {})

            result = await self.handoff_tool._execute(
                target_agent=agent_name,
                task=task,
                context=context,
                wait_for_result=True
            )
            return result

        results = await asyncio.gather(*[run_task(t) for t in tasks], return_exceptions=True)
        return results

    async def run_sequential(self, tasks: List[Dict[str, Any]],
                            pass_context: bool = True) -> List[Any]:
        """Run agent tasks sequentially, optionally passing context."""
        results = []
        accumulated_context = {}

        for task_info in tasks:
            agent_name = task_info["agent"]
            task = task_info["task"]
            context = task_info.get("context", {})

            if pass_context:
                context.update(accumulated_context)

            result = await self.handoff_tool._execute(
                target_agent=agent_name,
                task=task,
                context=context,
                wait_for_result=True
            )

            results.append(result)

            # Update accumulated context
            if pass_context and result.get("success"):
                accumulated_context[f"{agent_name}_result"] = result.get("result")

        return results

    async def run_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Run a defined workflow with conditional logic."""
        steps = workflow.get("steps", [])
        results = {}

        for step in steps:
            step_name = step.get("name", "unnamed")
            agent_name = step.get("agent")
            task = step.get("task")
            condition = step.get("condition")

            # Check condition
            if condition and not self._evaluate_condition(condition, results):
                results[step_name] = {"skipped": True}
                continue

            # Execute step
            result = await self.handoff_tool._execute(
                target_agent=agent_name,
                task=task,
                context={"previous_results": results},
                wait_for_result=True
            )

            results[step_name] = result

            # Check for early termination
            if step.get("terminate_on_failure") and not result.get("success"):
                break

        return results

    def _evaluate_condition(self, condition: str, results: Dict) -> bool:
        """Evaluate a condition string."""
        try:
            # Simple evaluation - in production, use a proper expression parser
            return eval(condition, {"results": results})
        except:
            return True

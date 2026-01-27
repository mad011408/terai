"""
Handoff manager for decentralized agent-to-agent communication.
"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import uuid


class HandoffStatus(Enum):
    """Status of a handoff."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class HandoffMessage:
    """Message passed during handoff."""
    message_id: str
    sender: str
    content: Any
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Handoff:
    """A handoff between agents."""
    handoff_id: str
    source_agent: str
    target_agent: str
    task: str
    context: Dict[str, Any]
    status: HandoffStatus = HandoffStatus.PENDING
    messages: List[HandoffMessage] = field(default_factory=list)
    result: Any = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "handoff_id": self.handoff_id,
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "task": self.task,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


class HandoffManager:
    """
    Manages handoffs between agents.
    Supports both synchronous and asynchronous handoffs.
    """

    def __init__(self):
        self.agents: Dict[str, Any] = {}
        self.pending_handoffs: Dict[str, Handoff] = {}
        self.completed_handoffs: List[Handoff] = []
        self.handoff_callbacks: Dict[str, List[Callable]] = {}
        self._event_handlers: Dict[str, List[Callable]] = {
            "handoff_created": [],
            "handoff_accepted": [],
            "handoff_completed": [],
            "handoff_failed": [],
        }

    def register_agent(self, name: str, agent: Any,
                      capabilities: Optional[List[str]] = None) -> None:
        """Register an agent."""
        self.agents[name] = {
            "agent": agent,
            "capabilities": capabilities or [],
            "available": True
        }

    def unregister_agent(self, name: str) -> bool:
        """Unregister an agent."""
        if name in self.agents:
            del self.agents[name]
            return True
        return False

    def on_event(self, event: str, callback: Callable) -> None:
        """Register an event handler."""
        if event in self._event_handlers:
            self._event_handlers[event].append(callback)

    async def _emit_event(self, event: str, **kwargs) -> None:
        """Emit an event to handlers."""
        for handler in self._event_handlers.get(event, []):
            if asyncio.iscoroutinefunction(handler):
                await handler(**kwargs)
            else:
                handler(**kwargs)

    async def initiate_handoff(self, source_agent: str, target_agent: str,
                              task: str, context: Dict[str, Any] = None,
                              wait_for_result: bool = True,
                              timeout: float = 120.0) -> Handoff:
        """
        Initiate a handoff to another agent.

        Args:
            source_agent: The agent initiating the handoff
            target_agent: The agent to hand off to
            task: The task description
            context: Context to pass to target
            wait_for_result: Whether to wait for completion
            timeout: Timeout for waiting

        Returns:
            The handoff object
        """
        # Validate target agent
        if target_agent not in self.agents:
            raise ValueError(f"Target agent '{target_agent}' not found")

        target_info = self.agents[target_agent]
        if not target_info["available"]:
            raise ValueError(f"Target agent '{target_agent}' is not available")

        # Create handoff
        handoff = Handoff(
            handoff_id=str(uuid.uuid4()),
            source_agent=source_agent,
            target_agent=target_agent,
            task=task,
            context=context or {}
        )

        self.pending_handoffs[handoff.handoff_id] = handoff
        await self._emit_event("handoff_created", handoff=handoff)

        if wait_for_result:
            return await self._execute_handoff(handoff, timeout)
        else:
            # Start async execution
            asyncio.create_task(self._execute_handoff(handoff, timeout))
            return handoff

    async def _execute_handoff(self, handoff: Handoff, timeout: float) -> Handoff:
        """Execute a handoff."""
        target_info = self.agents.get(handoff.target_agent)
        if not target_info:
            handoff.status = HandoffStatus.FAILED
            handoff.error = "Target agent not found"
            return handoff

        target_agent = target_info["agent"]
        handoff.status = HandoffStatus.IN_PROGRESS
        await self._emit_event("handoff_accepted", handoff=handoff)

        try:
            # Create context for agent
            from ..core.context import Context
            ctx = Context()
            ctx.update(handoff.context)
            ctx.set("handoff_id", handoff.handoff_id)
            ctx.set("source_agent", handoff.source_agent)

            # Execute agent
            result = await asyncio.wait_for(
                target_agent.run(handoff.task, ctx),
                timeout=timeout
            )

            handoff.status = HandoffStatus.COMPLETED
            handoff.result = result.result if hasattr(result, 'result') else result
            handoff.completed_at = datetime.now()

            await self._emit_event("handoff_completed", handoff=handoff)

        except asyncio.TimeoutError:
            handoff.status = HandoffStatus.FAILED
            handoff.error = f"Handoff timed out after {timeout}s"
            await self._emit_event("handoff_failed", handoff=handoff)

        except Exception as e:
            handoff.status = HandoffStatus.FAILED
            handoff.error = str(e)
            await self._emit_event("handoff_failed", handoff=handoff)

        finally:
            # Move to completed
            if handoff.handoff_id in self.pending_handoffs:
                del self.pending_handoffs[handoff.handoff_id]
            self.completed_handoffs.append(handoff)

        return handoff

    async def send_message(self, handoff_id: str, sender: str,
                          content: Any, metadata: Dict = None) -> bool:
        """
        Send a message within a handoff.

        Args:
            handoff_id: The handoff ID
            sender: The sender agent name
            content: Message content
            metadata: Optional metadata

        Returns:
            Whether the message was sent
        """
        handoff = self.pending_handoffs.get(handoff_id)
        if not handoff:
            return False

        message = HandoffMessage(
            message_id=str(uuid.uuid4()),
            sender=sender,
            content=content,
            metadata=metadata or {}
        )

        handoff.messages.append(message)
        return True

    def get_handoff(self, handoff_id: str) -> Optional[Handoff]:
        """Get a handoff by ID."""
        handoff = self.pending_handoffs.get(handoff_id)
        if handoff:
            return handoff

        for h in self.completed_handoffs:
            if h.handoff_id == handoff_id:
                return h

        return None

    def get_pending_for_agent(self, agent_name: str) -> List[Handoff]:
        """Get pending handoffs for an agent."""
        return [
            h for h in self.pending_handoffs.values()
            if h.target_agent == agent_name
        ]

    def cancel_handoff(self, handoff_id: str) -> bool:
        """Cancel a pending handoff."""
        handoff = self.pending_handoffs.get(handoff_id)
        if not handoff:
            return False

        handoff.status = HandoffStatus.CANCELLED
        handoff.completed_at = datetime.now()

        del self.pending_handoffs[handoff_id]
        self.completed_handoffs.append(handoff)
        return True

    def find_agent_for_task(self, task: str,
                           required_capabilities: List[str] = None) -> Optional[str]:
        """
        Find the best agent for a task based on capabilities.

        Args:
            task: Task description
            required_capabilities: Required capabilities

        Returns:
            Agent name or None
        """
        required_capabilities = required_capabilities or []

        for name, info in self.agents.items():
            if not info["available"]:
                continue

            agent_caps = set(info["capabilities"])
            required = set(required_capabilities)

            if required.issubset(agent_caps):
                return name

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get handoff statistics."""
        completed = len(self.completed_handoffs)
        successful = sum(1 for h in self.completed_handoffs if h.status == HandoffStatus.COMPLETED)
        failed = sum(1 for h in self.completed_handoffs if h.status == HandoffStatus.FAILED)

        return {
            "registered_agents": len(self.agents),
            "pending_handoffs": len(self.pending_handoffs),
            "completed_handoffs": completed,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / completed if completed > 0 else 0
        }

"""
State machine for managing agent and workflow states.
"""

from typing import Any, Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio


@dataclass
class State:
    """A state in the state machine."""
    name: str
    description: str = ""
    is_initial: bool = False
    is_final: bool = False
    on_enter: Optional[Callable] = None
    on_exit: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Transition:
    """A transition between states."""
    name: str
    from_state: str
    to_state: str
    condition: Optional[Callable] = None
    action: Optional[Callable] = None
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StateChange:
    """Record of a state change."""
    from_state: str
    to_state: str
    transition: str
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)


class StateMachine:
    """
    Finite state machine for managing complex state transitions.
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self.states: Dict[str, State] = {}
        self.transitions: Dict[str, Transition] = {}
        self.current_state: Optional[str] = None
        self.history: List[StateChange] = []
        self.context: Dict[str, Any] = {}
        self._event_handlers: Dict[str, List[Callable]] = {
            "state_enter": [],
            "state_exit": [],
            "transition": [],
        }

    def add_state(self, state: State) -> None:
        """Add a state to the machine."""
        self.states[state.name] = state
        if state.is_initial and self.current_state is None:
            self.current_state = state.name

    def add_transition(self, transition: Transition) -> None:
        """Add a transition."""
        if transition.from_state not in self.states:
            raise ValueError(f"Source state '{transition.from_state}' not found")
        if transition.to_state not in self.states:
            raise ValueError(f"Target state '{transition.to_state}' not found")

        self.transitions[transition.name] = transition

    def on(self, event: str, handler: Callable) -> None:
        """Register an event handler."""
        if event in self._event_handlers:
            self._event_handlers[event].append(handler)

    async def _emit(self, event: str, **kwargs) -> None:
        """Emit an event."""
        for handler in self._event_handlers.get(event, []):
            if asyncio.iscoroutinefunction(handler):
                await handler(**kwargs)
            else:
                handler(**kwargs)

    def get_current_state(self) -> Optional[State]:
        """Get the current state object."""
        if self.current_state:
            return self.states.get(self.current_state)
        return None

    def get_available_transitions(self) -> List[Transition]:
        """Get transitions available from current state."""
        if not self.current_state:
            return []

        available = []
        for trans in self.transitions.values():
            if trans.from_state == self.current_state:
                # Check condition if present
                if trans.condition:
                    try:
                        if trans.condition(self.context):
                            available.append(trans)
                    except:
                        pass
                else:
                    available.append(trans)

        return sorted(available, key=lambda t: -t.priority)

    async def trigger(self, transition_name: str,
                     context_update: Dict[str, Any] = None) -> bool:
        """
        Trigger a transition.

        Args:
            transition_name: Name of the transition
            context_update: Updates to apply to context

        Returns:
            Whether the transition was successful
        """
        transition = self.transitions.get(transition_name)
        if not transition:
            return False

        if transition.from_state != self.current_state:
            return False

        # Check condition
        if transition.condition:
            try:
                if not transition.condition(self.context):
                    return False
            except:
                return False

        # Update context
        if context_update:
            self.context.update(context_update)

        # Exit current state
        current_state_obj = self.states.get(self.current_state)
        if current_state_obj and current_state_obj.on_exit:
            if asyncio.iscoroutinefunction(current_state_obj.on_exit):
                await current_state_obj.on_exit(self.context)
            else:
                current_state_obj.on_exit(self.context)

        await self._emit("state_exit", state=self.current_state)

        # Execute transition action
        if transition.action:
            if asyncio.iscoroutinefunction(transition.action):
                await transition.action(self.context)
            else:
                transition.action(self.context)

        # Record change
        change = StateChange(
            from_state=self.current_state,
            to_state=transition.to_state,
            transition=transition_name,
            timestamp=datetime.now(),
            context=self.context.copy()
        )
        self.history.append(change)

        await self._emit("transition", transition=transition, change=change)

        # Enter new state
        self.current_state = transition.to_state
        new_state_obj = self.states.get(self.current_state)
        if new_state_obj and new_state_obj.on_enter:
            if asyncio.iscoroutinefunction(new_state_obj.on_enter):
                await new_state_obj.on_enter(self.context)
            else:
                new_state_obj.on_enter(self.context)

        await self._emit("state_enter", state=self.current_state)

        return True

    async def trigger_auto(self, context_update: Dict[str, Any] = None) -> Optional[str]:
        """
        Automatically trigger the first valid transition.

        Returns:
            Name of triggered transition or None
        """
        if context_update:
            self.context.update(context_update)

        available = self.get_available_transitions()
        if available:
            transition = available[0]
            if await self.trigger(transition.name):
                return transition.name

        return None

    def is_in_final_state(self) -> bool:
        """Check if current state is final."""
        state = self.get_current_state()
        return state.is_final if state else False

    def can_transition(self, transition_name: str) -> bool:
        """Check if a transition is possible."""
        transition = self.transitions.get(transition_name)
        if not transition:
            return False

        if transition.from_state != self.current_state:
            return False

        if transition.condition:
            try:
                return transition.condition(self.context)
            except:
                return False

        return True

    def reset(self) -> None:
        """Reset to initial state."""
        for state in self.states.values():
            if state.is_initial:
                self.current_state = state.name
                break
        self.history.clear()
        self.context.clear()

    def get_history(self) -> List[Dict[str, Any]]:
        """Get state change history."""
        return [
            {
                "from": c.from_state,
                "to": c.to_state,
                "transition": c.transition,
                "timestamp": c.timestamp.isoformat()
            }
            for c in self.history
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state machine to dictionary."""
        return {
            "name": self.name,
            "current_state": self.current_state,
            "states": [s.name for s in self.states.values()],
            "transitions": [t.name for t in self.transitions.values()],
            "context": self.context,
            "history_length": len(self.history)
        }


class AgentStateMachine(StateMachine):
    """
    Pre-configured state machine for agent lifecycle.
    """

    def __init__(self, agent_name: str):
        super().__init__(name=f"agent_{agent_name}")
        self._setup_agent_states()

    def _setup_agent_states(self) -> None:
        """Setup default agent states."""
        # States
        self.add_state(State(name="idle", description="Agent is idle", is_initial=True))
        self.add_state(State(name="thinking", description="Agent is reasoning"))
        self.add_state(State(name="executing", description="Agent is executing action"))
        self.add_state(State(name="waiting", description="Agent is waiting for input"))
        self.add_state(State(name="completed", description="Agent completed task", is_final=True))
        self.add_state(State(name="error", description="Agent encountered error", is_final=True))

        # Transitions
        self.add_transition(Transition(name="start", from_state="idle", to_state="thinking"))
        self.add_transition(Transition(name="plan", from_state="thinking", to_state="executing"))
        self.add_transition(Transition(name="think_more", from_state="executing", to_state="thinking"))
        self.add_transition(Transition(name="wait", from_state="executing", to_state="waiting"))
        self.add_transition(Transition(name="resume", from_state="waiting", to_state="thinking"))
        self.add_transition(Transition(name="complete", from_state="executing", to_state="completed"))
        self.add_transition(Transition(name="complete", from_state="thinking", to_state="completed"))
        self.add_transition(Transition(name="fail", from_state="thinking", to_state="error"))
        self.add_transition(Transition(name="fail", from_state="executing", to_state="error"))
        self.add_transition(Transition(name="reset", from_state="completed", to_state="idle"))
        self.add_transition(Transition(name="reset", from_state="error", to_state="idle"))


class WorkflowStateMachine(StateMachine):
    """
    Pre-configured state machine for workflow execution.
    """

    def __init__(self, workflow_name: str):
        super().__init__(name=f"workflow_{workflow_name}")
        self._setup_workflow_states()

    def _setup_workflow_states(self) -> None:
        """Setup default workflow states."""
        self.add_state(State(name="pending", description="Workflow pending", is_initial=True))
        self.add_state(State(name="running", description="Workflow running"))
        self.add_state(State(name="paused", description="Workflow paused"))
        self.add_state(State(name="completed", description="Workflow completed", is_final=True))
        self.add_state(State(name="failed", description="Workflow failed", is_final=True))
        self.add_state(State(name="cancelled", description="Workflow cancelled", is_final=True))

        self.add_transition(Transition(name="start", from_state="pending", to_state="running"))
        self.add_transition(Transition(name="pause", from_state="running", to_state="paused"))
        self.add_transition(Transition(name="resume", from_state="paused", to_state="running"))
        self.add_transition(Transition(name="complete", from_state="running", to_state="completed"))
        self.add_transition(Transition(name="fail", from_state="running", to_state="failed"))
        self.add_transition(Transition(name="cancel", from_state="running", to_state="cancelled"))
        self.add_transition(Transition(name="cancel", from_state="paused", to_state="cancelled"))
        self.add_transition(Transition(name="cancel", from_state="pending", to_state="cancelled"))

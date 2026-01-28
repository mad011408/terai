"""
Base agent class with reasoning capabilities.
Provides foundation for all specialized agents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Union
from enum import Enum
import asyncio
import uuid
from datetime import datetime

from .message import Message, MessageType
from .context import Context


class AgentState(Enum):
    """Agent execution states."""
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    WAITING = "waiting"
    COMPLETED = "completed"
    ERROR = "error"


class ReasoningStrategy(Enum):
    """Reasoning strategies for agent decision making."""
    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHT = "tree_of_thought"
    REACT = "react"  # Reasoning + Acting
    REFLEXION = "reflexion"
    PLAN_AND_EXECUTE = "plan_and_execute"


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    name: str
    description: str
    model: str = "anthropic/claude-sonnet-4"
    temperature: float = 0.7
    max_tokens: int = 120000
    reasoning_strategy: ReasoningStrategy = ReasoningStrategy.REACT
    max_iterations: int = 10
    timeout: float = 1600.0
    retry_attempts: int = 3
    tools: List[str] = field(default_factory=list)
    guardrails: List[str] = field(default_factory=list)
    system_prompt: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThoughtStep:
    """Represents a single step in the agent's reasoning process."""
    step_id: str
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class AgentResponse:
    """Response from an agent execution."""
    agent_id: str
    agent_name: str
    success: bool
    result: Any
    reasoning_trace: List[ThoughtStep]
    messages: List[Message]
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time: float = 0.0


class Agent(ABC):
    """
    Base agent class with reasoning capabilities.
    All specialized agents inherit from this class.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent_id = str(uuid.uuid4())
        self.state = AgentState.IDLE
        self.reasoning_trace: List[ThoughtStep] = []
        self.context: Optional[Context] = None
        self.tools: Dict[str, Any] = {}
        self.hooks: Dict[str, List[Callable]] = {
            "pre_think": [],
            "post_think": [],
            "pre_act": [],
            "post_act": [],
            "on_error": [],
            "on_complete": []
        }
        self._iteration_count = 0

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def description(self) -> str:
        return self.config.description

    def register_hook(self, event: str, callback: Callable) -> None:
        """Register a hook for a specific event."""
        if event in self.hooks:
            self.hooks[event].append(callback)

    async def _trigger_hooks(self, event: str, **kwargs) -> None:
        """Trigger all hooks for a specific event."""
        for hook in self.hooks.get(event, []):
            if asyncio.iscoroutinefunction(hook):
                await hook(self, **kwargs)
            else:
                hook(self, **kwargs)

    def add_thought(self, thought: str, action: Optional[str] = None,
                    action_input: Optional[Dict[str, Any]] = None) -> ThoughtStep:
        """Add a thought step to the reasoning trace."""
        step = ThoughtStep(
            step_id=f"step_{len(self.reasoning_trace) + 1}",
            thought=thought,
            action=action,
            action_input=action_input
        )
        self.reasoning_trace.append(step)
        return step

    def update_observation(self, step_id: str, observation: str) -> None:
        """Update the observation for a thought step."""
        for step in self.reasoning_trace:
            if step.step_id == step_id:
                step.observation = observation
                break

    @abstractmethod
    async def think(self, context: Context) -> ThoughtStep:
        """
        Generate the next thought based on current context.
        Returns the reasoning step with potential action.
        """
        pass

    @abstractmethod
    async def act(self, thought_step: ThoughtStep) -> str:
        """
        Execute the action specified in the thought step.
        Returns the observation/result.
        """
        pass

    @abstractmethod
    async def should_continue(self, context: Context) -> bool:
        """
        Determine if the agent should continue reasoning.
        Returns False when task is complete or max iterations reached.
        """
        pass

    async def run(self, task: str, context: Optional[Context] = None) -> AgentResponse:
        """
        Main execution loop for the agent.
        Implements the reasoning-action cycle.
        """
        start_time = datetime.now()
        self.context = context or Context()
        self.context.set("task", task)
        self.state = AgentState.THINKING
        self.reasoning_trace = []
        self._iteration_count = 0

        messages: List[Message] = []
        result = None
        error = None

        try:
            while await self.should_continue(self.context):
                self._iteration_count += 1

                if self._iteration_count > self.config.max_iterations:
                    raise RuntimeError(f"Max iterations ({self.config.max_iterations}) exceeded")

                # Think phase
                await self._trigger_hooks("pre_think", context=self.context)
                self.state = AgentState.THINKING
                thought_step = await self.think(self.context)
                await self._trigger_hooks("post_think", thought=thought_step)

                # Act phase (if action specified)
                if thought_step.action:
                    await self._trigger_hooks("pre_act", action=thought_step.action)
                    self.state = AgentState.EXECUTING
                    observation = await self.act(thought_step)
                    self.update_observation(thought_step.step_id, observation)
                    self.context.set("last_observation", observation)
                    await self._trigger_hooks("post_act", observation=observation)

                # Update context with reasoning
                self.context.append_to_list("reasoning_history", thought_step.to_dict())

            # Get final result
            result = self.context.get("final_result", self.reasoning_trace[-1].observation if self.reasoning_trace else None)
            self.state = AgentState.COMPLETED
            await self._trigger_hooks("on_complete", result=result)

        except Exception as e:
            self.state = AgentState.ERROR
            error = str(e)
            await self._trigger_hooks("on_error", error=e)

        execution_time = (datetime.now() - start_time).total_seconds()

        return AgentResponse(
            agent_id=self.agent_id,
            agent_name=self.name,
            success=error is None,
            result=result,
            reasoning_trace=self.reasoning_trace,
            messages=messages,
            metadata={
                "iterations": self._iteration_count,
                "model": self.config.model,
                "strategy": self.config.reasoning_strategy.value
            },
            error=error,
            execution_time=execution_time
        )

    def reset(self) -> None:
        """Reset the agent state."""
        self.state = AgentState.IDLE
        self.reasoning_trace = []
        self.context = None
        self._iteration_count = 0

    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        if self.config.system_prompt:
            return self.config.system_prompt
        return self._default_system_prompt()

    def _default_system_prompt(self) -> str:
        """Generate default system prompt."""
        return f"""You are {self.name}, an AI agent specialized in {self.description}.

Your reasoning strategy is {self.config.reasoning_strategy.value}.

When solving tasks:
1. Think step by step about the problem
2. Consider multiple approaches before acting
3. Use available tools when necessary
4. Reflect on observations and adjust your approach
5. Provide clear, actionable outputs

Available tools: {', '.join(self.config.tools) if self.config.tools else 'None'}

Always explain your reasoning before taking action."""


class SimpleAgent(Agent):
    """
    A simple implementation of the base Agent class.
    Useful for straightforward tasks without complex reasoning.
    """

    def __init__(self, config: AgentConfig, model_client: Any = None):
        super().__init__(config)
        self.model_client = model_client

    async def think(self, context: Context) -> ThoughtStep:
        """Generate next thought using the model."""
        task = context.get("task", "")
        history = context.get("reasoning_history", [])

        prompt = self._build_think_prompt(task, history)

        # If model client is available, use it
        if self.model_client:
            response = await self.model_client.generate(
                prompt=prompt,
                system=self.get_system_prompt(),
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            thought_text = response.content
        else:
            thought_text = f"Analyzing task: {task}"

        # Parse thought to extract action if present
        action, action_input = self._parse_action(thought_text)

        return self.add_thought(thought_text, action, action_input)

    async def act(self, thought_step: ThoughtStep) -> str:
        """Execute the specified action."""
        action = thought_step.action
        action_input = thought_step.action_input or {}

        if action and action in self.tools:
            tool = self.tools[action]
            result = await tool.execute(**action_input)
            return str(result)

        return "No action taken"

    async def should_continue(self, context: Context) -> bool:
        """Check if agent should continue."""
        if self._iteration_count >= self.config.max_iterations:
            return False

        # Check if task is marked as complete
        if context.get("task_complete", False):
            return False

        # Check for final answer in last reasoning step
        if self.reasoning_trace:
            last_thought = self.reasoning_trace[-1].thought.lower()
            if "final answer" in last_thought or "task complete" in last_thought:
                context.set("task_complete", True)
                return False

        return True

    def _build_think_prompt(self, task: str, history: List[Dict]) -> str:
        """Build prompt for thinking phase."""
        prompt = f"Task: {task}\n\n"

        if history:
            prompt += "Previous reasoning:\n"
            for step in history[-5:]:  # Last 5 steps
                prompt += f"- Thought: {step.get('thought', '')}\n"
                if step.get('observation'):
                    prompt += f"  Observation: {step['observation']}\n"

        prompt += "\nWhat is your next thought and action?"
        return prompt

    def _parse_action(self, thought_text: str) -> tuple[Optional[str], Optional[Dict]]:
        """Parse action from thought text."""
        # Simple parsing - can be enhanced
        action = None
        action_input = None

        if "Action:" in thought_text:
            try:
                action_part = thought_text.split("Action:")[1].split("\n")[0].strip()
                action = action_part.split("(")[0].strip() if "(" in action_part else action_part
            except:
                pass

        return action, action_input

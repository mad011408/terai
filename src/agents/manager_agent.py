"""
Manager Agent - Central orchestrator for task delegation and coordination.
Analyzes tasks and delegates to specialized agents.
"""

from typing import Any, Dict, List, Optional
import asyncio

from ..core.agent import Agent, AgentConfig, ThoughtStep, ReasoningStrategy
from ..core.context import Context
from ..core.message import Message, MessageType


class ManagerAgent(Agent):
    """
    Central orchestrator agent that manages task delegation.
    Analyzes incoming tasks, breaks them down, and coordinates specialized agents.
    """

    def __init__(self, config: Optional[AgentConfig] = None, model_client: Any = None):
        default_config = AgentConfig(
            name="manager_agent",
            description="Central orchestrator for task analysis, planning, and delegation",
            model="anthropic/claude-sonnet-4",
            reasoning_strategy=ReasoningStrategy.REACT,
            max_iterations=1,
            tools=["delegate", "plan", "synthesize", "evaluate"],
            system_prompt=self._get_system_prompt()
        )
        super().__init__(config or default_config)
        self.model_client = model_client
        self.available_agents: Dict[str, Agent] = {}
        self.execution_plan: List[Dict[str, Any]] = []

    def _get_system_prompt(self) -> str:
        return """You are the Manager Agent, the central orchestrator of an AI agent system.

Your responsibilities:
1. ANALYZE incoming tasks to understand requirements and complexity
2. PLAN execution by breaking down complex tasks into subtasks
3. DELEGATE subtasks to appropriate specialized agents
4. COORDINATE agent interactions and data flow
5. SYNTHESIZE results from multiple agents into coherent outputs
6. HANDLE errors and adapt plans when needed

Available specialized agents:
- terminal_agent: Executes shell commands and terminal operations
- code_agent: Writes, analyzes, and modifies code
- research_agent: Searches the web and gathers information
- file_agent: Manages file operations (read, write, create, delete)
- debug_agent: Debugs issues and troubleshoots problems

Decision Framework:
- For simple tasks: Delegate directly to the most appropriate agent
- For complex tasks: Create a multi-step plan with dependencies
- For ambiguous tasks: Gather more information before proceeding
- For risky operations: Add validation steps

Output Format:
When delegating, use: Action: delegate(agent_name, subtask)
When planning, use: Action: plan(steps_list)
When synthesizing, use: Action: synthesize(results)

Always think step-by-step before taking action."""

    def register_agent(self, agent: Agent) -> None:
        """Register a specialized agent."""
        self.available_agents[agent.name] = agent

    def get_available_agents(self) -> List[str]:
        """Get list of available agent names."""
        return list(self.available_agents.keys())

    async def think(self, context: Context) -> ThoughtStep:
        """Analyze the task and plan next action."""
        task = context.get("task", "")
        previous_results = context.get("delegation_results", [])
        current_plan = context.get("execution_plan", [])
        current_step = context.get("current_step", 0)

        # Build analysis prompt
        prompt = self._build_analysis_prompt(task, previous_results, current_plan, current_step)

        if self.model_client:
            response = await self.model_client.generate(
                prompt=prompt,
                system=self.get_system_prompt(),
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            thought_content = response.content
        else:
            # Fallback to simple analysis
            thought_content = self._simple_task_analysis(task, current_plan, current_step)

        # Parse action from thought
        action, action_input = self._parse_manager_action(thought_content)

        thought_step = self.add_thought(
            thought=thought_content,
            action=action,
            action_input=action_input
        )

        return thought_step

    async def act(self, thought_step: ThoughtStep) -> str:
        """Execute the planned action."""
        action = thought_step.action
        action_input = thought_step.action_input or {}

        if action == "delegate":
            return await self._delegate_to_agent(
                agent_name=action_input.get("agent_name"),
                subtask=action_input.get("subtask"),
                context=self.context
            )
        elif action == "plan":
            return self._create_execution_plan(action_input.get("steps", []))
        elif action == "synthesize":
            return self._synthesize_results(action_input.get("results", []))
        elif action == "evaluate":
            return self._evaluate_progress()
        else:
            return "No action taken"

    async def should_continue(self, context: Context) -> bool:
        """Check if task orchestration should continue."""
        if self._iteration_count >= self.config.max_iterations:
            return False

        # Check if final result is set
        if context.get("final_result"):
            return False

        # Check if all plan steps completed
        plan = context.get("execution_plan", [])
        current_step = context.get("current_step", 0)
        if plan and current_step >= len(plan):
            context.set("final_result", self._compile_final_result())
            return False

        # Check for explicit completion
        if context.get("task_complete", False):
            return False

        return True

    async def _delegate_to_agent(self, agent_name: str, subtask: str,
                                  context: Context) -> str:
        """Delegate a subtask to a specialized agent."""
        if agent_name not in self.available_agents:
            return f"Error: Agent '{agent_name}' not available. Available: {self.get_available_agents()}"

        agent = self.available_agents[agent_name]

        # Create child context for the delegated task
        child_context = context.create_child()
        child_context.set("delegated_by", "manager_agent")
        child_context.set("parent_task", context.get("task"))

        try:
            response = await agent.run(subtask, child_context)

            # Store result
            delegation_results = context.get("delegation_results", [])
            delegation_results.append({
                "agent": agent_name,
                "subtask": subtask,
                "success": response.success,
                "result": response.result,
                "error": response.error
            })
            context.set("delegation_results", delegation_results)

            if response.success:
                return f"Agent '{agent_name}' completed: {response.result}"
            else:
                return f"Agent '{agent_name}' failed: {response.error}"

        except Exception as e:
            return f"Delegation error: {str(e)}"

    def _create_execution_plan(self, steps: List[Dict[str, Any]]) -> str:
        """Create an execution plan."""
        self.execution_plan = steps
        self.context.set("execution_plan", steps)
        self.context.set("current_step", 0)
        return f"Created execution plan with {len(steps)} steps"

    def _synthesize_results(self, results: List[Any]) -> str:
        """Synthesize results from multiple agents."""
        if not results:
            results = self.context.get("delegation_results", [])

        synthesis = []
        for r in results:
            if isinstance(r, dict):
                synthesis.append(f"- {r.get('agent', 'unknown')}: {r.get('result', 'No result')}")
            else:
                synthesis.append(f"- {r}")

        synthesized = "\n".join(synthesis)
        self.context.set("synthesized_results", synthesized)
        return f"Synthesized {len(results)} results:\n{synthesized}"

    def _evaluate_progress(self) -> str:
        """Evaluate current progress."""
        plan = self.context.get("execution_plan", [])
        current_step = self.context.get("current_step", 0)
        results = self.context.get("delegation_results", [])

        successful = sum(1 for r in results if r.get("success", False))
        failed = len(results) - successful

        return f"Progress: Step {current_step}/{len(plan)}, Success: {successful}, Failed: {failed}"

    def _compile_final_result(self) -> str:
        """Compile the final result from all delegations."""
        results = self.context.get("delegation_results", [])
        synthesized = self.context.get("synthesized_results", "")

        if synthesized:
            return synthesized

        final_parts = []
        for r in results:
            if r.get("success"):
                final_parts.append(str(r.get("result", "")))

        return "\n".join(final_parts) if final_parts else "Task completed"

    def _build_analysis_prompt(self, task: str, previous_results: List,
                               current_plan: List, current_step: int) -> str:
        """Build prompt for task analysis."""
        prompt = f"Task: {task}\n\n"

        if current_plan:
            prompt += f"Current Plan (Step {current_step + 1}/{len(current_plan)}):\n"
            for i, step in enumerate(current_plan):
                status = "✓" if i < current_step else "→" if i == current_step else "○"
                prompt += f"  {status} {i + 1}. {step.get('description', step)}\n"
            prompt += "\n"

        if previous_results:
            prompt += "Previous Results:\n"
            for r in previous_results[-3:]:  # Last 3 results
                status = "✓" if r.get("success") else "✗"
                prompt += f"  {status} {r.get('agent')}: {r.get('result', r.get('error', 'No result'))[:100]}\n"
            prompt += "\n"

        prompt += f"Available Agents: {', '.join(self.get_available_agents())}\n\n"
        prompt += "What is your next thought and action?"

        return prompt

    def _simple_task_analysis(self, task: str, current_plan: List, current_step: int) -> str:
        """Simple rule-based task analysis."""
        task_lower = task.lower()

        # If we have a plan, execute next step
        if current_plan and current_step < len(current_plan):
            step = current_plan[current_step]
            self.context.set("current_step", current_step + 1)
            return f"Executing plan step {current_step + 1}: {step}\nAction: delegate({step.get('agent', 'code_agent')}, {step.get('task', task)})"

        # Analyze task type and delegate
        if any(kw in task_lower for kw in ["run", "execute", "command", "terminal", "shell"]):
            return f"This task requires terminal execution.\nAction: delegate(terminal_agent, {task})"
        elif any(kw in task_lower for kw in ["write code", "implement", "create function", "program"]):
            return f"This task requires code generation.\nAction: delegate(code_agent, {task})"
        elif any(kw in task_lower for kw in ["search", "find info", "research", "look up"]):
            return f"This task requires research.\nAction: delegate(research_agent, {task})"
        elif any(kw in task_lower for kw in ["read file", "write file", "create file", "delete"]):
            return f"This task requires file operations.\nAction: delegate(file_agent, {task})"
        elif any(kw in task_lower for kw in ["debug", "fix", "error", "bug", "troubleshoot"]):
            return f"This task requires debugging.\nAction: delegate(debug_agent, {task})"
        else:
            # Default to code agent for general tasks
            return f"Delegating task to code agent for analysis.\nAction: delegate(code_agent, {task})"

    def _parse_manager_action(self, thought: str) -> tuple:
        """Parse action from manager's thought."""
        action = None
        action_input = {}

        if "Action:" in thought:
            action_part = thought.split("Action:")[1].strip()

            if "delegate(" in action_part:
                action = "delegate"
                # Extract agent and task
                try:
                    params = action_part.split("delegate(")[1].split(")")[0]
                    parts = params.split(",", 1)
                    action_input["agent_name"] = parts[0].strip().strip("'\"")
                    if len(parts) > 1:
                        action_input["subtask"] = parts[1].strip().strip("'\"")
                except:
                    pass

            elif "plan(" in action_part:
                action = "plan"
                action_input["steps"] = []

            elif "synthesize(" in action_part:
                action = "synthesize"
                action_input["results"] = []

            elif "evaluate" in action_part:
                action = "evaluate"

        return action, action_input

    async def run_multi_agent_task(self, task: str, agents: List[str],
                                    parallel: bool = False) -> Dict[str, Any]:
        """Run a task across multiple agents."""
        context = Context()
        context.set("task", task)
        context.set("multi_agent_mode", True)

        if parallel:
            # Run all agents in parallel
            tasks = []
            for agent_name in agents:
                if agent_name in self.available_agents:
                    agent = self.available_agents[agent_name]
                    agent_context = context.create_child()
                    tasks.append(agent.run(task, agent_context))

            results = await asyncio.gather(*tasks, return_exceptions=True)
            return {
                "mode": "parallel",
                "results": [
                    {"agent": agents[i], "response": r if not isinstance(r, Exception) else str(r)}
                    for i, r in enumerate(results)
                ]
            }
        else:
            # Run sequentially, passing context between agents
            results = []
            for agent_name in agents:
                if agent_name in self.available_agents:
                    agent = self.available_agents[agent_name]
                    response = await agent.run(task, context)
                    results.append({"agent": agent_name, "response": response})
                    # Update context with result for next agent
                    context.set(f"{agent_name}_result", response.result)

            return {"mode": "sequential", "results": results}

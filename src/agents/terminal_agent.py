"""
Terminal Agent - Executes shell commands and manages terminal operations.
Handles command execution, output parsing, and terminal state management.
"""

from typing import Any, Dict, List, Optional, Tuple
import asyncio
import subprocess
import shlex
import os
import platform

from ..core.agent import Agent, AgentConfig, ThoughtStep, ReasoningStrategy
from ..core.context import Context


class TerminalAgent(Agent):
    """
    Specialized agent for terminal command execution.
    Safely executes shell commands with validation and output parsing.
    """

    def __init__(self, config: Optional[AgentConfig] = None, model_client: Any = None):
        default_config = AgentConfig(
            name="terminal_agent",
            description="Terminal command execution and shell operations",
            model="anthropic/claude-sonnet-4",
            reasoning_strategy=ReasoningStrategy.REACT,
            max_iterations=10,
            tools=["execute_command", "check_status", "list_directory", "get_environment"],
            system_prompt=self._get_system_prompt()
        )
        super().__init__(config or default_config)
        self.model_client = model_client
        self.command_history: List[Dict[str, Any]] = []
        self.working_directory = os.getcwd()
        self.environment = os.environ.copy()
        self.blocked_commands = self._get_blocked_commands()

    def _get_system_prompt(self) -> str:
        return """You are the Terminal Agent, specialized in shell command execution.

Your capabilities:
1. Execute shell commands safely
2. Parse and interpret command output
3. Manage working directory and environment
4. Chain commands for complex operations
5. Handle errors and suggest fixes

Safety rules:
- NEVER execute destructive commands without explicit confirmation
- NEVER expose sensitive information (passwords, keys, tokens)
- ALWAYS validate command syntax before execution
- PREFER safe alternatives when available

Blocked commands (require special handling):
- rm -rf / (or any recursive delete of root)
- Format/fdisk operations
- Network attacks
- Privilege escalation without context

Output Format:
For command execution: Action: execute(command)
For directory change: Action: cd(path)
For environment check: Action: env(variable)

Always explain what each command does before executing."""

    def _get_blocked_commands(self) -> List[str]:
        """Get list of blocked dangerous commands."""
        return [
            "rm -rf /",
            "rm -rf /*",
            "mkfs",
            "dd if=/dev/zero",
            ":(){:|:&};:",  # Fork bomb
            "chmod -R 777 /",
            "wget | sh",
            "curl | bash",
        ]

    async def think(self, context: Context) -> ThoughtStep:
        """Analyze task and plan command execution."""
        task = context.get("task", "")
        last_output = context.get("last_command_output", "")
        last_error = context.get("last_command_error", "")

        prompt = self._build_prompt(task, last_output, last_error)

        if self.model_client:
            response = await self.model_client.generate(
                prompt=prompt,
                system=self.get_system_prompt(),
                temperature=0.3,  # Lower temperature for command generation
                max_tokens=1024
            )
            thought_content = response.content
        else:
            thought_content = self._generate_command_thought(task)

        action, action_input = self._parse_terminal_action(thought_content)

        return self.add_thought(
            thought=thought_content,
            action=action,
            action_input=action_input
        )

    async def act(self, thought_step: ThoughtStep) -> str:
        """Execute the terminal action."""
        action = thought_step.action
        action_input = thought_step.action_input or {}

        if action == "execute":
            command = action_input.get("command", "")
            return await self.execute_command(command)
        elif action == "cd":
            path = action_input.get("path", "")
            return self.change_directory(path)
        elif action == "env":
            var = action_input.get("variable")
            return self.get_environment_variable(var)
        elif action == "list":
            path = action_input.get("path", ".")
            return await self.list_directory(path)
        else:
            return "Unknown action"

    async def should_continue(self, context: Context) -> bool:
        """Check if terminal operations should continue."""
        if self._iteration_count >= self.config.max_iterations:
            return False

        if context.get("task_complete", False):
            return False

        # Check if last command succeeded and task is done
        last_output = context.get("last_command_output", "")
        if last_output and not context.get("last_command_error"):
            # Task might be complete
            if self._iteration_count > 0:
                context.set("final_result", last_output)
                return False

        return True

    async def execute_command(self, command: str, timeout: int = 60) -> str:
        """Execute a shell command safely."""
        # Validate command
        validation_result = self._validate_command(command)
        if not validation_result[0]:
            return f"Command blocked: {validation_result[1]}"

        # Record in history
        self.command_history.append({
            "command": command,
            "cwd": self.working_directory,
            "timestamp": asyncio.get_event_loop().time()
        })

        try:
            # Determine shell based on platform
            if platform.system() == "Windows":
                shell = True
                executable = None
            else:
                shell = True
                executable = "/bin/bash"

            # Execute command
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_directory,
                env=self.environment,
                shell=shell
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                return f"Command timed out after {timeout} seconds"

            output = stdout.decode('utf-8', errors='replace')
            error = stderr.decode('utf-8', errors='replace')

            # Update context
            self.context.set("last_command_output", output)
            self.context.set("last_command_error", error if process.returncode != 0 else "")
            self.context.set("last_return_code", process.returncode)

            # Update command history with result
            self.command_history[-1].update({
                "output": output[:1000],  # Truncate for history
                "error": error[:500],
                "return_code": process.returncode
            })

            if process.returncode != 0:
                return f"Command failed (exit code {process.returncode}):\n{error}\n{output}"

            return output if output else "Command executed successfully (no output)"

        except Exception as e:
            return f"Execution error: {str(e)}"

    def _validate_command(self, command: str) -> Tuple[bool, str]:
        """Validate command for safety."""
        command_lower = command.lower().strip()

        # Check blocked commands
        for blocked in self.blocked_commands:
            if blocked in command_lower:
                return False, f"Blocked dangerous command pattern: {blocked}"

        # Check for sudo/admin without context
        if command_lower.startswith("sudo ") or command_lower.startswith("runas "):
            # Allow but warn
            pass

        # Check for pipe to shell (potential code injection)
        if "| bash" in command_lower or "| sh" in command_lower:
            if "curl" in command_lower or "wget" in command_lower:
                return False, "Piping downloaded content to shell is blocked"

        return True, "Command validated"

    def change_directory(self, path: str) -> str:
        """Change working directory."""
        try:
            # Resolve path
            if not os.path.isabs(path):
                path = os.path.join(self.working_directory, path)

            path = os.path.normpath(path)

            if not os.path.exists(path):
                return f"Directory does not exist: {path}"

            if not os.path.isdir(path):
                return f"Not a directory: {path}"

            self.working_directory = path
            return f"Changed directory to: {path}"

        except Exception as e:
            return f"Error changing directory: {str(e)}"

    def get_environment_variable(self, variable: Optional[str] = None) -> str:
        """Get environment variable(s)."""
        if variable:
            value = self.environment.get(variable, "")
            return f"{variable}={value}" if value else f"{variable} is not set"
        else:
            # Return summary of important variables
            important_vars = ["PATH", "HOME", "USER", "SHELL", "PWD", "LANG"]
            return "\n".join(
                f"{var}={self.environment.get(var, 'not set')}"
                for var in important_vars
            )

    def set_environment_variable(self, variable: str, value: str) -> str:
        """Set an environment variable."""
        self.environment[variable] = value
        return f"Set {variable}={value}"

    async def list_directory(self, path: str = ".") -> str:
        """List directory contents."""
        if platform.system() == "Windows":
            command = f'dir "{path}"'
        else:
            command = f'ls -la "{path}"'

        return await self.execute_command(command)

    def get_command_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent command history."""
        return self.command_history[-limit:]

    def _build_prompt(self, task: str, last_output: str, last_error: str) -> str:
        """Build prompt for terminal analysis."""
        prompt = f"Task: {task}\n\n"
        prompt += f"Working Directory: {self.working_directory}\n"
        prompt += f"Platform: {platform.system()}\n\n"

        if last_output:
            prompt += f"Last Command Output:\n{last_output[:500]}\n\n"

        if last_error:
            prompt += f"Last Error:\n{last_error[:300]}\n\n"

        if self.command_history:
            prompt += "Recent Commands:\n"
            for cmd in self.command_history[-3:]:
                prompt += f"  $ {cmd['command']}\n"
            prompt += "\n"

        prompt += "What command should be executed next?"
        return prompt

    def _generate_command_thought(self, task: str) -> str:
        """Generate command thought without model."""
        task_lower = task.lower()

        # Simple pattern matching for common tasks
        if "list" in task_lower and ("file" in task_lower or "dir" in task_lower):
            cmd = "dir" if platform.system() == "Windows" else "ls -la"
            return f"Need to list directory contents.\nAction: execute({cmd})"

        elif "create" in task_lower and "dir" in task_lower:
            # Extract directory name
            return f"Need to create a directory.\nAction: execute(mkdir new_directory)"

        elif "current" in task_lower and "dir" in task_lower:
            cmd = "cd" if platform.system() == "Windows" else "pwd"
            return f"Need to show current directory.\nAction: execute({cmd})"

        elif "python" in task_lower:
            return f"Need to run Python.\nAction: execute(python --version)"

        elif "git" in task_lower:
            if "status" in task_lower:
                return "Checking git status.\nAction: execute(git status)"
            elif "log" in task_lower:
                return "Showing git log.\nAction: execute(git log --oneline -10)"

        elif "install" in task_lower:
            if "pip" in task_lower or "python" in task_lower:
                return f"Installing Python package.\nAction: execute(pip install package_name)"
            elif "npm" in task_lower:
                return f"Installing npm package.\nAction: execute(npm install)"

        # Default: echo the task
        return f"Processing task: {task}\nAction: execute(echo 'Task received: {task}')"

    def _parse_terminal_action(self, thought: str) -> Tuple[Optional[str], Optional[Dict]]:
        """Parse action from terminal thought."""
        action = None
        action_input = {}

        if "Action:" in thought:
            action_part = thought.split("Action:")[1].strip()

            if "execute(" in action_part:
                action = "execute"
                try:
                    command = action_part.split("execute(")[1].split(")")[0]
                    action_input["command"] = command.strip("'\"")
                except:
                    pass

            elif "cd(" in action_part:
                action = "cd"
                try:
                    path = action_part.split("cd(")[1].split(")")[0]
                    action_input["path"] = path.strip("'\"")
                except:
                    pass

            elif "env(" in action_part:
                action = "env"
                try:
                    var = action_part.split("env(")[1].split(")")[0]
                    action_input["variable"] = var.strip("'\"") if var else None
                except:
                    pass

            elif "list(" in action_part:
                action = "list"
                try:
                    path = action_part.split("list(")[1].split(")")[0]
                    action_input["path"] = path.strip("'\"") if path else "."
                except:
                    action_input["path"] = "."

        return action, action_input


class CommandBuilder:
    """Helper class for building complex commands."""

    def __init__(self):
        self.parts: List[str] = []

    def add(self, command: str) -> "CommandBuilder":
        self.parts.append(command)
        return self

    def pipe(self, command: str) -> "CommandBuilder":
        self.parts.append(f"| {command}")
        return self

    def and_then(self, command: str) -> "CommandBuilder":
        self.parts.append(f"&& {command}")
        return self

    def or_else(self, command: str) -> "CommandBuilder":
        self.parts.append(f"|| {command}")
        return self

    def redirect_to(self, file: str, append: bool = False) -> "CommandBuilder":
        op = ">>" if append else ">"
        self.parts.append(f"{op} {file}")
        return self

    def build(self) -> str:
        return " ".join(self.parts)

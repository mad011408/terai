"""
Terminal command executor tool.
Safely executes shell commands with sandboxing and validation.
"""

from typing import Any, Dict, List, Optional
import asyncio
import subprocess
import platform
import os
import shlex
from dataclasses import dataclass

from ..base_tool import BaseTool, ToolConfig, ToolParameter, ToolCategory


@dataclass
class CommandResult:
    """Result of command execution."""
    command: str
    stdout: str
    stderr: str
    return_code: int
    execution_time: float
    timed_out: bool = False


class TerminalExecutorTool(BaseTool):
    """
    Tool for executing terminal commands safely.
    Includes command validation and sandboxing.
    """

    def __init__(self, working_directory: Optional[str] = None,
                 sandbox_enabled: bool = True):
        config = ToolConfig(
            name="terminal_executor",
            description="Execute terminal/shell commands. Use with caution.",
            category=ToolCategory.ACTION,
            timeout=120.0,
            retry_attempts=1,
            requires_confirmation=True
        )
        super().__init__(config)
        self.working_directory = working_directory or os.getcwd()
        self.sandbox_enabled = sandbox_enabled
        self.environment = os.environ.copy()
        self.blocked_commands = self._get_blocked_commands()
        self.allowed_commands = None  # None means all non-blocked allowed

    def _get_blocked_commands(self) -> List[str]:
        """Get list of blocked dangerous commands."""
        return [
            # Destructive file operations
            "rm -rf /",
            "rm -rf /*",
            "rm -rf ~",
            "> /dev/sda",
            "mkfs",
            "dd if=/dev/zero of=/dev",

            # Fork bombs
            ":(){:|:&};:",
            "fork while fork",

            # Network attacks
            ":(){ :|:& };:",

            # Dangerous permission changes
            "chmod -R 777 /",
            "chmod -R 000 /",
            "chown -R",

            # Download and execute
            "curl | bash",
            "curl | sh",
            "wget | bash",
            "wget | sh",

            # History/credential exposure
            "history",
            "cat ~/.bash_history",
            "cat /etc/shadow",
            "cat /etc/passwd",
        ]

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="command",
                param_type="string",
                description="The shell command to execute",
                required=True
            ),
            ToolParameter(
                name="timeout",
                param_type="number",
                description="Command timeout in seconds",
                required=False,
                default=60.0,
                min_value=1,
                max_value=600
            ),
            ToolParameter(
                name="working_directory",
                param_type="string",
                description="Working directory for the command",
                required=False
            ),
            ToolParameter(
                name="environment",
                param_type="object",
                description="Additional environment variables",
                required=False,
                default={}
            ),
            ToolParameter(
                name="capture_output",
                param_type="boolean",
                description="Whether to capture stdout/stderr",
                required=False,
                default=True
            )
        ]

    def _validate_command(self, command: str) -> tuple[bool, Optional[str]]:
        """Validate command for safety."""
        command_lower = command.lower().strip()

        # Check blocked commands
        for blocked in self.blocked_commands:
            if blocked.lower() in command_lower:
                return False, f"Command contains blocked pattern: {blocked}"

        # Check for pipe to shell
        if "| bash" in command_lower or "| sh" in command_lower:
            if "curl" in command_lower or "wget" in command_lower:
                return False, "Piping downloaded content to shell is blocked"

        # Check allowed commands if whitelist is set
        if self.allowed_commands is not None:
            command_name = command.split()[0] if command.split() else ""
            if command_name not in self.allowed_commands:
                return False, f"Command not in allowed list: {command_name}"

        return True, None

    async def _execute(self, command: str, timeout: float = 60.0,
                      working_directory: Optional[str] = None,
                      environment: Dict[str, str] = None,
                      capture_output: bool = True) -> CommandResult:
        """Execute the command."""
        import time

        # Validate command
        is_valid, error = self._validate_command(command)
        if not is_valid:
            return CommandResult(
                command=command,
                stdout="",
                stderr=error,
                return_code=-1,
                execution_time=0
            )

        # Setup working directory
        cwd = working_directory or self.working_directory

        # Setup environment
        env = self.environment.copy()
        if environment:
            env.update(environment)

        start_time = time.time()

        try:
            # Determine shell based on platform
            if platform.system() == "Windows":
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE if capture_output else None,
                    stderr=asyncio.subprocess.PIPE if capture_output else None,
                    cwd=cwd,
                    env=env
                )
            else:
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE if capture_output else None,
                    stderr=asyncio.subprocess.PIPE if capture_output else None,
                    cwd=cwd,
                    env=env,
                    executable="/bin/bash"
                )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )

                return CommandResult(
                    command=command,
                    stdout=stdout.decode('utf-8', errors='replace') if stdout else "",
                    stderr=stderr.decode('utf-8', errors='replace') if stderr else "",
                    return_code=process.returncode,
                    execution_time=time.time() - start_time
                )

            except asyncio.TimeoutError:
                process.kill()
                await process.wait()

                return CommandResult(
                    command=command,
                    stdout="",
                    stderr=f"Command timed out after {timeout} seconds",
                    return_code=-1,
                    execution_time=timeout,
                    timed_out=True
                )

        except Exception as e:
            return CommandResult(
                command=command,
                stdout="",
                stderr=str(e),
                return_code=-1,
                execution_time=time.time() - start_time
            )

    def set_allowed_commands(self, commands: List[str]) -> None:
        """Set whitelist of allowed commands."""
        self.allowed_commands = set(commands)

    def add_blocked_command(self, pattern: str) -> None:
        """Add a pattern to blocked commands."""
        self.blocked_commands.append(pattern)

    def set_working_directory(self, directory: str) -> None:
        """Set the working directory."""
        if os.path.isdir(directory):
            self.working_directory = directory
        else:
            raise ValueError(f"Not a directory: {directory}")

    def set_environment_variable(self, key: str, value: str) -> None:
        """Set an environment variable."""
        self.environment[key] = value


class InteractiveTerminalTool(BaseTool):
    """
    Tool for interactive terminal sessions.
    Maintains state across multiple commands.
    """

    def __init__(self):
        config = ToolConfig(
            name="interactive_terminal",
            description="Interactive terminal session with state persistence.",
            category=ToolCategory.ACTION,
            timeout=300.0
        )
        super().__init__(config)
        self.process: Optional[asyncio.subprocess.Process] = None
        self.working_directory = os.getcwd()
        self.command_history: List[str] = []

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="command",
                param_type="string",
                description="Command to execute in the interactive session",
                required=True
            ),
            ToolParameter(
                name="timeout",
                param_type="number",
                description="Timeout for this command",
                required=False,
                default=30.0
            )
        ]

    async def _start_session(self):
        """Start an interactive shell session."""
        if platform.system() == "Windows":
            self.process = await asyncio.create_subprocess_shell(
                "cmd",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_directory
            )
        else:
            self.process = await asyncio.create_subprocess_shell(
                "/bin/bash",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_directory
            )

    async def _execute(self, command: str, timeout: float = 30.0) -> Dict[str, Any]:
        """Execute command in interactive session."""
        if not self.process or self.process.returncode is not None:
            await self._start_session()

        # Add command to history
        self.command_history.append(command)

        # Send command
        self.process.stdin.write(f"{command}\n".encode())
        await self.process.stdin.drain()

        # Read output (this is simplified - real implementation would be more complex)
        try:
            output = await asyncio.wait_for(
                self.process.stdout.readline(),
                timeout=timeout
            )
            return {
                "command": command,
                "output": output.decode('utf-8', errors='replace'),
                "session_active": True
            }
        except asyncio.TimeoutError:
            return {
                "command": command,
                "output": "Command timed out",
                "session_active": True,
                "timed_out": True
            }

    async def close_session(self):
        """Close the interactive session."""
        if self.process:
            self.process.terminate()
            await self.process.wait()
            self.process = None

    def get_history(self) -> List[str]:
        """Get command history."""
        return self.command_history.copy()

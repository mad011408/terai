"""
Subprocess manager tool for coordinating background processes.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import uuid
import signal
import os

from ..base_tool import BaseTool, ToolConfig, ToolParameter, ToolCategory


class ProcessStatus(Enum):
    """Status of a managed process."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class ManagedProcess:
    """A managed subprocess."""
    process_id: str
    command: str
    status: ProcessStatus
    process: Optional[asyncio.subprocess.Process] = None
    stdout: str = ""
    stderr: str = ""
    return_code: Optional[int] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SubprocessManagerTool(BaseTool):
    """
    Tool for managing multiple subprocesses.
    Supports starting, monitoring, and stopping processes.
    """

    def __init__(self, max_processes: int = 10):
        config = ToolConfig(
            name="subprocess_manager",
            description="Manage background processes - start, monitor, and stop.",
            category=ToolCategory.ORCHESTRATION,
            timeout=300.0
        )
        super().__init__(config)
        self.max_processes = max_processes
        self.processes: Dict[str, ManagedProcess] = {}
        self._cleanup_task: Optional[asyncio.Task] = None

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                param_type="string",
                description="Action to perform",
                required=True,
                enum_values=["start", "stop", "status", "list", "wait", "output"]
            ),
            ToolParameter(
                name="command",
                param_type="string",
                description="Command to execute (for start action)",
                required=False
            ),
            ToolParameter(
                name="process_id",
                param_type="string",
                description="Process ID (for stop, status, wait, output actions)",
                required=False
            ),
            ToolParameter(
                name="timeout",
                param_type="number",
                description="Timeout in seconds",
                required=False,
                default=60.0
            ),
            ToolParameter(
                name="working_directory",
                param_type="string",
                description="Working directory for the process",
                required=False
            ),
            ToolParameter(
                name="environment",
                param_type="object",
                description="Environment variables",
                required=False,
                default={}
            )
        ]

    async def _execute(self, action: str, command: Optional[str] = None,
                      process_id: Optional[str] = None,
                      timeout: float = 60.0,
                      working_directory: Optional[str] = None,
                      environment: Dict[str, str] = None) -> Dict[str, Any]:
        """Execute subprocess management action."""
        if action == "start":
            if not command:
                return {"success": False, "error": "Command required for start action"}
            return await self.start_process(command, working_directory, environment)

        elif action == "stop":
            if not process_id:
                return {"success": False, "error": "Process ID required for stop action"}
            return await self.stop_process(process_id)

        elif action == "status":
            if process_id:
                return self.get_process_status(process_id)
            return {"success": False, "error": "Process ID required for status action"}

        elif action == "list":
            return self.list_processes()

        elif action == "wait":
            if not process_id:
                return {"success": False, "error": "Process ID required for wait action"}
            return await self.wait_for_process(process_id, timeout)

        elif action == "output":
            if not process_id:
                return {"success": False, "error": "Process ID required for output action"}
            return self.get_process_output(process_id)

        else:
            return {"success": False, "error": f"Unknown action: {action}"}

    async def start_process(self, command: str,
                           working_directory: Optional[str] = None,
                           environment: Dict[str, str] = None) -> Dict[str, Any]:
        """Start a new subprocess."""
        # Check process limit
        active_count = sum(1 for p in self.processes.values()
                         if p.status == ProcessStatus.RUNNING)
        if active_count >= self.max_processes:
            return {
                "success": False,
                "error": f"Maximum process limit ({self.max_processes}) reached"
            }

        process_id = str(uuid.uuid4())[:8]

        # Setup environment
        env = os.environ.copy()
        if environment:
            env.update(environment)

        managed = ManagedProcess(
            process_id=process_id,
            command=command,
            status=ProcessStatus.PENDING
        )
        self.processes[process_id] = managed

        try:
            # Start process
            managed.process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_directory,
                env=env
            )
            managed.status = ProcessStatus.RUNNING
            managed.started_at = datetime.now()

            # Start output collection task
            asyncio.create_task(self._collect_output(process_id))

            return {
                "success": True,
                "process_id": process_id,
                "pid": managed.process.pid,
                "status": managed.status.value
            }

        except Exception as e:
            managed.status = ProcessStatus.FAILED
            managed.stderr = str(e)
            return {
                "success": False,
                "process_id": process_id,
                "error": str(e)
            }

    async def _collect_output(self, process_id: str):
        """Collect output from a running process."""
        managed = self.processes.get(process_id)
        if not managed or not managed.process:
            return

        try:
            stdout, stderr = await managed.process.communicate()

            managed.stdout = stdout.decode('utf-8', errors='replace')
            managed.stderr = stderr.decode('utf-8', errors='replace')
            managed.return_code = managed.process.returncode
            managed.completed_at = datetime.now()

            if managed.return_code == 0:
                managed.status = ProcessStatus.COMPLETED
            else:
                managed.status = ProcessStatus.FAILED

        except asyncio.CancelledError:
            managed.status = ProcessStatus.CANCELLED
        except Exception as e:
            managed.status = ProcessStatus.FAILED
            managed.stderr += f"\nError: {str(e)}"

    async def stop_process(self, process_id: str, force: bool = False) -> Dict[str, Any]:
        """Stop a running subprocess."""
        managed = self.processes.get(process_id)
        if not managed:
            return {"success": False, "error": f"Process not found: {process_id}"}

        if managed.status != ProcessStatus.RUNNING:
            return {
                "success": False,
                "error": f"Process not running (status: {managed.status.value})"
            }

        if not managed.process:
            return {"success": False, "error": "No process handle"}

        try:
            if force:
                managed.process.kill()
            else:
                managed.process.terminate()

            # Wait a bit for termination
            try:
                await asyncio.wait_for(managed.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                managed.process.kill()
                await managed.process.wait()

            managed.status = ProcessStatus.CANCELLED
            managed.completed_at = datetime.now()
            managed.return_code = managed.process.returncode

            return {
                "success": True,
                "process_id": process_id,
                "status": managed.status.value
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def wait_for_process(self, process_id: str, timeout: float = 60.0) -> Dict[str, Any]:
        """Wait for a process to complete."""
        managed = self.processes.get(process_id)
        if not managed:
            return {"success": False, "error": f"Process not found: {process_id}"}

        if managed.status not in [ProcessStatus.PENDING, ProcessStatus.RUNNING]:
            return {
                "success": True,
                "process_id": process_id,
                "status": managed.status.value,
                "return_code": managed.return_code
            }

        if not managed.process:
            return {"success": False, "error": "No process handle"}

        try:
            await asyncio.wait_for(managed.process.wait(), timeout=timeout)

            return {
                "success": True,
                "process_id": process_id,
                "status": managed.status.value,
                "return_code": managed.return_code,
                "stdout": managed.stdout[:1000],
                "stderr": managed.stderr[:500]
            }

        except asyncio.TimeoutError:
            managed.status = ProcessStatus.TIMEOUT
            return {
                "success": False,
                "process_id": process_id,
                "error": f"Process timed out after {timeout}s",
                "status": ProcessStatus.TIMEOUT.value
            }

    def get_process_status(self, process_id: str) -> Dict[str, Any]:
        """Get status of a process."""
        managed = self.processes.get(process_id)
        if not managed:
            return {"success": False, "error": f"Process not found: {process_id}"}

        result = {
            "success": True,
            "process_id": process_id,
            "command": managed.command,
            "status": managed.status.value,
            "return_code": managed.return_code,
            "started_at": managed.started_at.isoformat() if managed.started_at else None,
            "completed_at": managed.completed_at.isoformat() if managed.completed_at else None
        }

        if managed.process:
            result["pid"] = managed.process.pid

        return result

    def get_process_output(self, process_id: str) -> Dict[str, Any]:
        """Get output of a process."""
        managed = self.processes.get(process_id)
        if not managed:
            return {"success": False, "error": f"Process not found: {process_id}"}

        return {
            "success": True,
            "process_id": process_id,
            "stdout": managed.stdout,
            "stderr": managed.stderr,
            "status": managed.status.value
        }

    def list_processes(self) -> Dict[str, Any]:
        """List all managed processes."""
        processes = []
        for pid, managed in self.processes.items():
            processes.append({
                "process_id": pid,
                "command": managed.command[:50] + "..." if len(managed.command) > 50 else managed.command,
                "status": managed.status.value,
                "started_at": managed.started_at.isoformat() if managed.started_at else None
            })

        return {
            "success": True,
            "total": len(processes),
            "running": sum(1 for p in processes if p["status"] == "running"),
            "processes": processes
        }

    async def cleanup_completed(self, max_age_seconds: int = 3600) -> int:
        """Clean up completed processes older than max_age_seconds."""
        now = datetime.now()
        to_remove = []

        for process_id, managed in self.processes.items():
            if managed.status not in [ProcessStatus.PENDING, ProcessStatus.RUNNING]:
                if managed.completed_at:
                    age = (now - managed.completed_at).total_seconds()
                    if age > max_age_seconds:
                        to_remove.append(process_id)

        for process_id in to_remove:
            del self.processes[process_id]

        return len(to_remove)

    async def stop_all(self) -> int:
        """Stop all running processes."""
        stopped = 0
        for process_id, managed in list(self.processes.items()):
            if managed.status == ProcessStatus.RUNNING:
                result = await self.stop_process(process_id)
                if result.get("success"):
                    stopped += 1
        return stopped

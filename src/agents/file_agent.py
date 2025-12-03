"""
File Agent - File operations and management.
Handles file reading, writing, creation, deletion, and directory operations.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import os
import shutil
import json
import mimetypes
import hashlib

from ..core.agent import Agent, AgentConfig, ThoughtStep, ReasoningStrategy
from ..core.context import Context


@dataclass
class FileInfo:
    """Information about a file."""
    path: str
    name: str
    extension: str
    size: int
    created: datetime
    modified: datetime
    is_directory: bool
    mime_type: Optional[str] = None
    checksum: Optional[str] = None


@dataclass
class FileOperation:
    """Record of a file operation."""
    operation: str  # read, write, create, delete, move, copy
    source: str
    destination: Optional[str]
    timestamp: datetime
    success: bool
    error: Optional[str] = None


class FileAgent(Agent):
    """
    Specialized agent for file system operations.
    Provides safe file manipulation with backup and validation.
    """

    def __init__(self, config: Optional[AgentConfig] = None, model_client: Any = None):
        default_config = AgentConfig(
            name="file_agent",
            description="File operations including read, write, create, delete",
            model="anthropic/claude-sonnet-4",
            reasoning_strategy=ReasoningStrategy.REACT,
            max_iterations=10,
            tools=["read_file", "write_file", "create_file", "delete_file", "list_directory", "move_file", "copy_file"],
            system_prompt=self._get_system_prompt()
        )
        super().__init__(config or default_config)
        self.model_client = model_client
        self.operation_history: List[FileOperation] = []
        self.working_directory = os.getcwd()
        self.backup_enabled = True
        self.backup_directory = ".file_agent_backup"

    def _get_system_prompt(self) -> str:
        return """You are the File Agent, specialized in file system operations.

Your capabilities:
1. Read files of various formats (text, JSON, CSV, etc.)
2. Write and create new files
3. Delete files safely with confirmation
4. Move and copy files
5. List directory contents
6. Search for files by pattern
7. Manage file permissions

Safety rules:
- ALWAYS validate paths to prevent directory traversal attacks
- BACKUP files before destructive operations
- CONFIRM before deleting multiple files
- CHECK file sizes before reading large files
- PRESERVE file permissions when copying

Output Format:
For reading: Action: read(filepath)
For writing: Action: write(filepath, content)
For creating: Action: create(filepath)
For deleting: Action: delete(filepath)
For listing: Action: list(directory)
For moving: Action: move(source, destination)
For copying: Action: copy(source, destination)

Always report operation results clearly."""

    async def think(self, context: Context) -> ThoughtStep:
        """Plan file operation."""
        task = context.get("task", "")
        last_result = context.get("last_operation_result", "")

        prompt = self._build_file_prompt(task, last_result)

        if self.model_client:
            response = await self.model_client.generate(
                prompt=prompt,
                system=self.get_system_prompt(),
                temperature=0.3,
                max_tokens=1024
            )
            thought_content = response.content
        else:
            thought_content = self._generate_file_thought(task)

        action, action_input = self._parse_file_action(thought_content)

        return self.add_thought(
            thought=thought_content,
            action=action,
            action_input=action_input
        )

    async def act(self, thought_step: ThoughtStep) -> str:
        """Execute file operation."""
        action = thought_step.action
        action_input = thought_step.action_input or {}

        result = ""
        try:
            if action == "read":
                result = self.read_file(action_input.get("path", ""))
            elif action == "write":
                result = self.write_file(
                    action_input.get("path", ""),
                    action_input.get("content", "")
                )
            elif action == "create":
                result = self.create_file(
                    action_input.get("path", ""),
                    action_input.get("content", "")
                )
            elif action == "delete":
                result = self.delete_file(action_input.get("path", ""))
            elif action == "list":
                result = self.list_directory(action_input.get("path", "."))
            elif action == "move":
                result = self.move_file(
                    action_input.get("source", ""),
                    action_input.get("destination", "")
                )
            elif action == "copy":
                result = self.copy_file(
                    action_input.get("source", ""),
                    action_input.get("destination", "")
                )
            elif action == "info":
                result = self.get_file_info(action_input.get("path", ""))
            else:
                result = "Unknown action"
        except Exception as e:
            result = f"Error: {str(e)}"

        self.context.set("last_operation_result", result)
        return result

    async def should_continue(self, context: Context) -> bool:
        """Check if file operations should continue."""
        if self._iteration_count >= self.config.max_iterations:
            return False

        if context.get("task_complete", False):
            return False

        # Single operation tasks are usually done after one iteration
        if self._iteration_count > 0 and context.get("last_operation_result"):
            context.set("final_result", context.get("last_operation_result"))
            return False

        return True

    def _validate_path(self, path: str) -> Tuple[bool, str]:
        """Validate path for safety."""
        try:
            # Resolve to absolute path
            resolved = Path(path).resolve()

            # Check for path traversal
            if ".." in str(resolved):
                return False, "Path traversal detected"

            # Don't allow access to sensitive directories
            sensitive_dirs = ["/etc", "/var", "/usr", "/bin", "/sbin", "/root"]
            for sensitive in sensitive_dirs:
                if str(resolved).startswith(sensitive):
                    return False, f"Access to {sensitive} is restricted"

            return True, str(resolved)
        except Exception as e:
            return False, str(e)

    def read_file(self, path: str, encoding: str = "utf-8") -> str:
        """Read file contents."""
        valid, resolved_path = self._validate_path(path)
        if not valid:
            return f"Invalid path: {resolved_path}"

        if not os.path.exists(resolved_path):
            return f"File not found: {resolved_path}"

        # Check file size
        file_size = os.path.getsize(resolved_path)
        if file_size > 10 * 1024 * 1024:  # 10MB limit
            return f"File too large ({file_size} bytes). Consider reading in chunks."

        try:
            with open(resolved_path, 'r', encoding=encoding) as f:
                content = f.read()

            self._record_operation("read", resolved_path, None, True)
            return content
        except UnicodeDecodeError:
            # Try binary read
            with open(resolved_path, 'rb') as f:
                content = f.read()
            return f"Binary file ({len(content)} bytes)"
        except Exception as e:
            self._record_operation("read", resolved_path, None, False, str(e))
            return f"Error reading file: {str(e)}"

    def write_file(self, path: str, content: str, encoding: str = "utf-8") -> str:
        """Write content to file."""
        valid, resolved_path = self._validate_path(path)
        if not valid:
            return f"Invalid path: {resolved_path}"

        try:
            # Backup if file exists and backup is enabled
            if self.backup_enabled and os.path.exists(resolved_path):
                self._create_backup(resolved_path)

            # Create parent directories if needed
            os.makedirs(os.path.dirname(resolved_path) or ".", exist_ok=True)

            with open(resolved_path, 'w', encoding=encoding) as f:
                f.write(content)

            self._record_operation("write", resolved_path, None, True)
            return f"Successfully wrote {len(content)} bytes to {resolved_path}"
        except Exception as e:
            self._record_operation("write", resolved_path, None, False, str(e))
            return f"Error writing file: {str(e)}"

    def create_file(self, path: str, content: str = "") -> str:
        """Create a new file."""
        valid, resolved_path = self._validate_path(path)
        if not valid:
            return f"Invalid path: {resolved_path}"

        if os.path.exists(resolved_path):
            return f"File already exists: {resolved_path}"

        return self.write_file(resolved_path, content)

    def delete_file(self, path: str, force: bool = False) -> str:
        """Delete a file or directory."""
        valid, resolved_path = self._validate_path(path)
        if not valid:
            return f"Invalid path: {resolved_path}"

        if not os.path.exists(resolved_path):
            return f"File not found: {resolved_path}"

        try:
            # Backup before delete
            if self.backup_enabled:
                self._create_backup(resolved_path)

            if os.path.isdir(resolved_path):
                if force:
                    shutil.rmtree(resolved_path)
                else:
                    os.rmdir(resolved_path)  # Only works if empty
            else:
                os.remove(resolved_path)

            self._record_operation("delete", resolved_path, None, True)
            return f"Successfully deleted: {resolved_path}"
        except Exception as e:
            self._record_operation("delete", resolved_path, None, False, str(e))
            return f"Error deleting file: {str(e)}"

    def list_directory(self, path: str = ".", pattern: str = "*") -> str:
        """List directory contents."""
        valid, resolved_path = self._validate_path(path)
        if not valid:
            return f"Invalid path: {resolved_path}"

        if not os.path.exists(resolved_path):
            return f"Directory not found: {resolved_path}"

        if not os.path.isdir(resolved_path):
            return f"Not a directory: {resolved_path}"

        try:
            entries = []
            for entry in sorted(os.listdir(resolved_path)):
                full_path = os.path.join(resolved_path, entry)
                is_dir = os.path.isdir(full_path)
                size = os.path.getsize(full_path) if not is_dir else 0

                entry_type = "DIR " if is_dir else "FILE"
                size_str = f"{size:>10}" if not is_dir else "         -"
                entries.append(f"{entry_type} {size_str}  {entry}")

            result = f"Directory: {resolved_path}\n"
            result += f"Total: {len(entries)} items\n\n"
            result += "\n".join(entries)

            return result
        except Exception as e:
            return f"Error listing directory: {str(e)}"

    def move_file(self, source: str, destination: str) -> str:
        """Move file or directory."""
        valid_src, resolved_src = self._validate_path(source)
        valid_dst, resolved_dst = self._validate_path(destination)

        if not valid_src:
            return f"Invalid source path: {resolved_src}"
        if not valid_dst:
            return f"Invalid destination path: {resolved_dst}"

        if not os.path.exists(resolved_src):
            return f"Source not found: {resolved_src}"

        try:
            shutil.move(resolved_src, resolved_dst)
            self._record_operation("move", resolved_src, resolved_dst, True)
            return f"Moved {resolved_src} to {resolved_dst}"
        except Exception as e:
            self._record_operation("move", resolved_src, resolved_dst, False, str(e))
            return f"Error moving file: {str(e)}"

    def copy_file(self, source: str, destination: str) -> str:
        """Copy file or directory."""
        valid_src, resolved_src = self._validate_path(source)
        valid_dst, resolved_dst = self._validate_path(destination)

        if not valid_src:
            return f"Invalid source path: {resolved_src}"
        if not valid_dst:
            return f"Invalid destination path: {resolved_dst}"

        if not os.path.exists(resolved_src):
            return f"Source not found: {resolved_src}"

        try:
            if os.path.isdir(resolved_src):
                shutil.copytree(resolved_src, resolved_dst)
            else:
                shutil.copy2(resolved_src, resolved_dst)

            self._record_operation("copy", resolved_src, resolved_dst, True)
            return f"Copied {resolved_src} to {resolved_dst}"
        except Exception as e:
            self._record_operation("copy", resolved_src, resolved_dst, False, str(e))
            return f"Error copying file: {str(e)}"

    def get_file_info(self, path: str) -> str:
        """Get detailed file information."""
        valid, resolved_path = self._validate_path(path)
        if not valid:
            return f"Invalid path: {resolved_path}"

        if not os.path.exists(resolved_path):
            return f"File not found: {resolved_path}"

        try:
            stat = os.stat(resolved_path)
            is_dir = os.path.isdir(resolved_path)

            info = FileInfo(
                path=resolved_path,
                name=os.path.basename(resolved_path),
                extension=os.path.splitext(resolved_path)[1],
                size=stat.st_size,
                created=datetime.fromtimestamp(stat.st_ctime),
                modified=datetime.fromtimestamp(stat.st_mtime),
                is_directory=is_dir,
                mime_type=mimetypes.guess_type(resolved_path)[0] if not is_dir else None
            )

            # Calculate checksum for files
            if not is_dir and stat.st_size < 10 * 1024 * 1024:  # Under 10MB
                with open(resolved_path, 'rb') as f:
                    info.checksum = hashlib.md5(f.read()).hexdigest()

            result = f"""File Information:
- Path: {info.path}
- Name: {info.name}
- Type: {'Directory' if info.is_directory else 'File'}
- Size: {info.size} bytes
- Extension: {info.extension or 'None'}
- MIME Type: {info.mime_type or 'Unknown'}
- Created: {info.created}
- Modified: {info.modified}
- Checksum: {info.checksum or 'N/A'}
"""
            return result
        except Exception as e:
            return f"Error getting file info: {str(e)}"

    def search_files(self, directory: str, pattern: str, recursive: bool = True) -> List[str]:
        """Search for files matching pattern."""
        valid, resolved_path = self._validate_path(directory)
        if not valid:
            return []

        matches = []
        path = Path(resolved_path)

        if recursive:
            for match in path.rglob(pattern):
                matches.append(str(match))
        else:
            for match in path.glob(pattern):
                matches.append(str(match))

        return matches

    def _create_backup(self, path: str) -> Optional[str]:
        """Create backup of a file."""
        try:
            backup_dir = os.path.join(os.path.dirname(path), self.backup_directory)
            os.makedirs(backup_dir, exist_ok=True)

            filename = os.path.basename(path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(backup_dir, f"{filename}.{timestamp}.bak")

            if os.path.isdir(path):
                shutil.copytree(path, backup_path)
            else:
                shutil.copy2(path, backup_path)

            return backup_path
        except Exception:
            return None

    def _record_operation(self, operation: str, source: str,
                         destination: Optional[str], success: bool,
                         error: Optional[str] = None) -> None:
        """Record file operation in history."""
        self.operation_history.append(FileOperation(
            operation=operation,
            source=source,
            destination=destination,
            timestamp=datetime.now(),
            success=success,
            error=error
        ))

    def _build_file_prompt(self, task: str, last_result: str) -> str:
        """Build prompt for file operation planning."""
        prompt = f"Task: {task}\n"
        prompt += f"Working Directory: {self.working_directory}\n\n"

        if last_result:
            prompt += f"Last Operation Result:\n{last_result[:500]}\n\n"

        prompt += "What file operation should be performed?"
        return prompt

    def _generate_file_thought(self, task: str) -> str:
        """Generate file operation thought without model."""
        task_lower = task.lower()

        if "read" in task_lower or "open" in task_lower or "show" in task_lower:
            # Extract filename
            words = task.split()
            for word in words:
                if "." in word or "/" in word or "\\" in word:
                    return f"Need to read file: {word}\nAction: read({word})"
            return f"Need to read a file.\nAction: read(filename)"

        elif "write" in task_lower or "save" in task_lower:
            return f"Need to write to file.\nAction: write(filename, content)"

        elif "create" in task_lower:
            if "directory" in task_lower or "folder" in task_lower:
                return f"Need to create directory.\nAction: create(directory_name)"
            return f"Need to create file.\nAction: create(filename)"

        elif "delete" in task_lower or "remove" in task_lower:
            return f"Need to delete file.\nAction: delete(filepath)"

        elif "list" in task_lower or "ls" in task_lower:
            return f"Need to list directory.\nAction: list(.)"

        elif "move" in task_lower:
            return f"Need to move file.\nAction: move(source, destination)"

        elif "copy" in task_lower:
            return f"Need to copy file.\nAction: copy(source, destination)"

        elif "info" in task_lower:
            return f"Need to get file info.\nAction: info(filepath)"

        else:
            return f"Analyzing file task: {task}\nAction: list(.)"

    def _parse_file_action(self, thought: str) -> Tuple[Optional[str], Optional[Dict]]:
        """Parse action from file thought."""
        action = None
        action_input = {}

        if "Action:" in thought:
            action_part = thought.split("Action:")[1].strip()

            # Parse different actions
            for action_name in ["read", "write", "create", "delete", "list", "move", "copy", "info"]:
                if f"{action_name}(" in action_part:
                    action = action_name
                    try:
                        params = action_part.split(f"{action_name}(")[1].split(")")[0]

                        if action in ["move", "copy"]:
                            parts = params.split(",")
                            action_input["source"] = parts[0].strip().strip("'\"")
                            if len(parts) > 1:
                                action_input["destination"] = parts[1].strip().strip("'\"")
                        elif action == "write":
                            parts = params.split(",", 1)
                            action_input["path"] = parts[0].strip().strip("'\"")
                            if len(parts) > 1:
                                action_input["content"] = parts[1].strip().strip("'\"")
                        else:
                            action_input["path"] = params.strip().strip("'\"")
                    except:
                        pass
                    break

        return action, action_input

    def get_operation_history(self, limit: int = 10) -> List[Dict]:
        """Get recent operation history."""
        return [
            {
                "operation": op.operation,
                "source": op.source,
                "destination": op.destination,
                "timestamp": op.timestamp.isoformat(),
                "success": op.success,
                "error": op.error
            }
            for op in self.operation_history[-limit:]
        ]

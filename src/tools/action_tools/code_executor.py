"""
Code executor tool for running code in sandboxed environments.
Supports multiple languages with safety restrictions.
"""

from typing import Any, Dict, List, Optional
import asyncio
import tempfile
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from ..base_tool import BaseTool, ToolConfig, ToolParameter, ToolCategory


@dataclass
class ExecutionResult:
    """Result of code execution."""
    language: str
    code: str
    stdout: str
    stderr: str
    return_code: int
    execution_time: float
    error: Optional[str] = None


class CodeExecutorTool(BaseTool):
    """
    Tool for executing code in various languages.
    Uses sandboxed environments for safety.
    """

    def __init__(self, sandbox_enabled: bool = True,
                 allowed_languages: Optional[List[str]] = None):
        config = ToolConfig(
            name="code_executor",
            description="Execute code in a sandboxed environment. Supports Python, JavaScript, and more.",
            category=ToolCategory.ACTION,
            timeout=60.0,
            retry_attempts=1,
            requires_confirmation=True
        )
        super().__init__(config)
        self.sandbox_enabled = sandbox_enabled
        self.allowed_languages = allowed_languages or ["python", "javascript", "bash"]
        self.temp_dir = tempfile.mkdtemp(prefix="code_executor_")

        # Language configurations
        self.language_configs = {
            "python": {
                "extension": ".py",
                "command": [sys.executable, "-u"],
                "timeout": 30
            },
            "javascript": {
                "extension": ".js",
                "command": ["node"],
                "timeout": 30
            },
            "typescript": {
                "extension": ".ts",
                "command": ["npx", "ts-node"],
                "timeout": 45
            },
            "bash": {
                "extension": ".sh",
                "command": ["bash"],
                "timeout": 30
            },
            "ruby": {
                "extension": ".rb",
                "command": ["ruby"],
                "timeout": 30
            },
            "go": {
                "extension": ".go",
                "command": ["go", "run"],
                "timeout": 45
            }
        }

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="code",
                param_type="string",
                description="The code to execute",
                required=True
            ),
            ToolParameter(
                name="language",
                param_type="string",
                description="Programming language",
                required=True,
                enum_values=["python", "javascript", "typescript", "bash", "ruby", "go"]
            ),
            ToolParameter(
                name="timeout",
                param_type="number",
                description="Execution timeout in seconds",
                required=False,
                default=30.0,
                min_value=1,
                max_value=300
            ),
            ToolParameter(
                name="stdin",
                param_type="string",
                description="Input to provide via stdin",
                required=False,
                default=""
            )
        ]

    def _validate_code(self, code: str, language: str) -> tuple[bool, Optional[str]]:
        """Validate code for safety."""
        if language not in self.allowed_languages:
            return False, f"Language not allowed: {language}"

        # Common dangerous patterns
        dangerous_patterns = [
            "os.system",
            "subprocess",
            "eval(",
            "exec(",
            "__import__",
            "importlib",
            "rm -rf",
            ":(){ :|:& };:",
            "fork()",
            "/dev/",
            "socket.socket",
        ]

        code_lower = code.lower()
        for pattern in dangerous_patterns:
            if pattern.lower() in code_lower:
                return False, f"Code contains potentially dangerous pattern: {pattern}"

        return True, None

    async def _execute(self, code: str, language: str,
                      timeout: float = 30.0, stdin: str = "") -> ExecutionResult:
        """Execute code in sandboxed environment."""
        import time

        # Validate code
        is_valid, error = self._validate_code(code, language)
        if not is_valid:
            return ExecutionResult(
                language=language,
                code=code,
                stdout="",
                stderr=error,
                return_code=-1,
                execution_time=0,
                error=error
            )

        lang_config = self.language_configs.get(language)
        if not lang_config:
            return ExecutionResult(
                language=language,
                code=code,
                stdout="",
                stderr=f"Unsupported language: {language}",
                return_code=-1,
                execution_time=0,
                error=f"Unsupported language: {language}"
            )

        # Write code to temp file
        file_path = os.path.join(
            self.temp_dir,
            f"code_{id(code)}{lang_config['extension']}"
        )

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(code)

            # Build command
            command = lang_config['command'] + [file_path]

            start_time = time.time()

            # Execute
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.temp_dir
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input=stdin.encode() if stdin else None),
                    timeout=timeout
                )

                return ExecutionResult(
                    language=language,
                    code=code,
                    stdout=stdout.decode('utf-8', errors='replace'),
                    stderr=stderr.decode('utf-8', errors='replace'),
                    return_code=process.returncode,
                    execution_time=time.time() - start_time
                )

            except asyncio.TimeoutError:
                process.kill()
                await process.wait()

                return ExecutionResult(
                    language=language,
                    code=code,
                    stdout="",
                    stderr=f"Execution timed out after {timeout} seconds",
                    return_code=-1,
                    execution_time=timeout,
                    error="Timeout"
                )

        except Exception as e:
            return ExecutionResult(
                language=language,
                code=code,
                stdout="",
                stderr=str(e),
                return_code=-1,
                execution_time=0,
                error=str(e)
            )

        finally:
            # Cleanup temp file
            try:
                os.unlink(file_path)
            except:
                pass

    def cleanup(self):
        """Cleanup temp directory."""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass


class PythonREPLTool(BaseTool):
    """
    Python REPL tool for interactive code execution.
    Maintains state across executions.
    """

    def __init__(self):
        config = ToolConfig(
            name="python_repl",
            description="Interactive Python REPL with state persistence.",
            category=ToolCategory.ACTION,
            timeout=30.0
        )
        super().__init__(config)
        self.namespace: Dict[str, Any] = {}
        self.execution_history: List[str] = []

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="code",
                param_type="string",
                description="Python code to execute",
                required=True
            ),
            ToolParameter(
                name="reset_state",
                param_type="boolean",
                description="Reset the REPL state before execution",
                required=False,
                default=False
            )
        ]

    async def _execute(self, code: str, reset_state: bool = False) -> Dict[str, Any]:
        """Execute Python code in REPL."""
        import io
        import contextlib
        import traceback

        if reset_state:
            self.namespace = {}
            self.execution_history = []

        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        result = None
        error = None

        try:
            # Try to evaluate as expression first
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                try:
                    result = eval(code, self.namespace)
                except SyntaxError:
                    # Not an expression, execute as statement
                    exec(code, self.namespace)

            self.execution_history.append(code)

        except Exception as e:
            error = traceback.format_exc()

        return {
            "code": code,
            "result": repr(result) if result is not None else None,
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue(),
            "error": error,
            "variables": list(self.namespace.keys())
        }

    def reset(self):
        """Reset REPL state."""
        self.namespace = {}
        self.execution_history = []

    def get_history(self) -> List[str]:
        """Get execution history."""
        return self.execution_history.copy()

    def get_variables(self) -> Dict[str, str]:
        """Get current variable names and types."""
        return {
            name: type(value).__name__
            for name, value in self.namespace.items()
            if not name.startswith('_')
        }

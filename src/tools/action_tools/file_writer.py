"""
File writer tool for creating and modifying files.
Includes backup and validation features.
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
import os
import shutil
from datetime import datetime

from ..base_tool import BaseTool, ToolConfig, ToolParameter, ToolCategory


class FileWriterTool(BaseTool):
    """
    Tool for writing and creating files.
    Includes automatic backup and validation.
    """

    def __init__(self, backup_enabled: bool = True,
                 backup_directory: str = ".backups"):
        config = ToolConfig(
            name="file_writer",
            description="Write content to files. Creates backups automatically.",
            category=ToolCategory.ACTION,
            timeout=30.0,
            requires_confirmation=True
        )
        super().__init__(config)
        self.backup_enabled = backup_enabled
        self.backup_directory = backup_directory
        self.write_history: List[Dict[str, Any]] = []

        # Blocked paths
        self.blocked_paths = [
            "/etc",
            "/usr",
            "/bin",
            "/sbin",
            "/var",
            "/root",
            "C:\\Windows",
            "C:\\Program Files",
        ]

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="file_path",
                param_type="string",
                description="Path to the file to write",
                required=True
            ),
            ToolParameter(
                name="content",
                param_type="string",
                description="Content to write to the file",
                required=True
            ),
            ToolParameter(
                name="mode",
                param_type="string",
                description="Write mode",
                required=False,
                default="write",
                enum_values=["write", "append", "overwrite"]
            ),
            ToolParameter(
                name="encoding",
                param_type="string",
                description="File encoding",
                required=False,
                default="utf-8"
            ),
            ToolParameter(
                name="create_directories",
                param_type="boolean",
                description="Create parent directories if they don't exist",
                required=False,
                default=True
            ),
            ToolParameter(
                name="backup",
                param_type="boolean",
                description="Create backup before writing",
                required=False,
                default=True
            )
        ]

    def _validate_path(self, path: str) -> tuple[bool, Optional[str]]:
        """Validate the file path for safety."""
        resolved = Path(path).resolve()
        resolved_str = str(resolved)

        # Check blocked paths
        for blocked in self.blocked_paths:
            if resolved_str.startswith(blocked):
                return False, f"Path is in blocked directory: {blocked}"

        # Check for path traversal attempts
        if ".." in path:
            return False, "Path traversal detected"

        return True, None

    async def _execute(self, file_path: str, content: str,
                      mode: str = "write", encoding: str = "utf-8",
                      create_directories: bool = True,
                      backup: bool = True) -> Dict[str, Any]:
        """Write content to file."""
        # Validate path
        is_valid, error = self._validate_path(file_path)
        if not is_valid:
            raise ValueError(error)

        path = Path(file_path).resolve()

        # Create parent directories
        if create_directories:
            path.parent.mkdir(parents=True, exist_ok=True)

        # Create backup if file exists and backup is enabled
        backup_path = None
        if backup and self.backup_enabled and path.exists():
            backup_path = self._create_backup(path)

        # Determine write mode
        if mode == "append":
            file_mode = 'a'
        else:
            file_mode = 'w'

        # Write content
        try:
            with open(path, file_mode, encoding=encoding) as f:
                f.write(content)

            file_size = path.stat().st_size

            # Record in history
            record = {
                "file_path": str(path),
                "mode": mode,
                "content_length": len(content),
                "file_size": file_size,
                "backup_path": backup_path,
                "timestamp": datetime.now().isoformat()
            }
            self.write_history.append(record)

            return {
                "success": True,
                "file_path": str(path),
                "bytes_written": len(content.encode(encoding)),
                "file_size": file_size,
                "backup_path": backup_path,
                "mode": mode
            }

        except Exception as e:
            # Restore from backup if write failed
            if backup_path and os.path.exists(backup_path):
                shutil.copy2(backup_path, path)

            raise Exception(f"Failed to write file: {str(e)}")

    def _create_backup(self, path: Path) -> Optional[str]:
        """Create a backup of the file."""
        try:
            backup_dir = path.parent / self.backup_directory
            backup_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{path.name}.{timestamp}.bak"
            backup_path = backup_dir / backup_name

            shutil.copy2(path, backup_path)
            return str(backup_path)

        except Exception:
            return None

    def restore_from_backup(self, backup_path: str, target_path: str) -> bool:
        """Restore a file from backup."""
        try:
            shutil.copy2(backup_path, target_path)
            return True
        except Exception:
            return False

    def get_write_history(self, limit: int = 10) -> List[Dict]:
        """Get recent write history."""
        return self.write_history[-limit:]


class FileCreatorTool(BaseTool):
    """
    Tool for creating new files with templates.
    """

    def __init__(self):
        config = ToolConfig(
            name="file_creator",
            description="Create new files with optional templates.",
            category=ToolCategory.ACTION,
            timeout=30.0
        )
        super().__init__(config)
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load file templates."""
        return {
            "python": '''"""
{description}
"""


def main():
    """Main function."""
    pass


if __name__ == "__main__":
    main()
''',
            "python_class": '''"""
{description}
"""

from typing import Any, Dict, List, Optional


class {class_name}:
    """
    {description}
    """

    def __init__(self):
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
''',
            "javascript": '''/**
 * {description}
 */

function main() {{
    // Implementation
}}

module.exports = {{ main }};
''',
            "typescript": '''/**
 * {description}
 */

export function main(): void {{
    // Implementation
}}
''',
            "html": '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
</head>
<body>
    <h1>{title}</h1>
</body>
</html>
''',
            "css": '''/* {description} */

* {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}}

body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
}}
''',
            "json": '''{{
    "name": "{name}",
    "version": "1.0.0",
    "description": "{description}"
}}
''',
            "yaml": '''# {description}
name: {name}
version: "1.0.0"
''',
            "markdown": '''# {title}

## Description

{description}

## Usage

```
# Usage example
```

## License

MIT
''',
            "dockerfile": '''FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
''',
            "gitignore": '''# Python
__pycache__/
*.py[cod]
*$py.class
.Python
venv/
.env

# IDE
.idea/
.vscode/
*.swp
*.swo

# Build
dist/
build/
*.egg-info/

# Logs
*.log
logs/
''',
        }

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="file_path",
                param_type="string",
                description="Path for the new file",
                required=True
            ),
            ToolParameter(
                name="template",
                param_type="string",
                description="Template to use",
                required=False,
                default=None,
                enum_values=list(self.templates.keys())
            ),
            ToolParameter(
                name="variables",
                param_type="object",
                description="Variables to substitute in template",
                required=False,
                default={}
            ),
            ToolParameter(
                name="overwrite",
                param_type="boolean",
                description="Overwrite if file exists",
                required=False,
                default=False
            )
        ]

    async def _execute(self, file_path: str, template: Optional[str] = None,
                      variables: Dict[str, str] = None,
                      overwrite: bool = False) -> Dict[str, Any]:
        """Create a new file."""
        path = Path(file_path).resolve()

        # Check if file exists
        if path.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {file_path}")

        # Get content
        variables = variables or {}
        if template and template in self.templates:
            content = self.templates[template].format(**variables)
        else:
            content = ""

        # Create parent directories
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

        return {
            "success": True,
            "file_path": str(path),
            "template_used": template,
            "file_size": path.stat().st_size
        }

    def list_templates(self) -> List[str]:
        """List available templates."""
        return list(self.templates.keys())

    def add_template(self, name: str, content: str) -> None:
        """Add a custom template."""
        self.templates[name] = content

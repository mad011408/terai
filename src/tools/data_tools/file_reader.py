"""
File reader tool for reading and parsing various file formats.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import os
import json
import csv
import mimetypes

from ..base_tool import BaseTool, ToolConfig, ToolParameter, ToolCategory


class FileReaderTool(BaseTool):
    """
    Tool for reading files in various formats.
    Supports text, JSON, CSV, YAML, and more.
    """

    def __init__(self, allowed_extensions: Optional[List[str]] = None,
                 max_file_size: int = 10 * 1024 * 1024):  # 10MB default
        config = ToolConfig(
            name="file_reader",
            description="Read and parse files in various formats (text, JSON, CSV, YAML).",
            category=ToolCategory.DATA,
            timeout=30.0
        )
        super().__init__(config)
        self.allowed_extensions = allowed_extensions or [
            ".txt", ".json", ".csv", ".yaml", ".yml", ".md",
            ".py", ".js", ".ts", ".html", ".css", ".xml", ".log"
        ]
        self.max_file_size = max_file_size

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="file_path",
                param_type="string",
                description="Path to the file to read",
                required=True
            ),
            ToolParameter(
                name="encoding",
                param_type="string",
                description="File encoding",
                required=False,
                default="utf-8"
            ),
            ToolParameter(
                name="parse_format",
                param_type="string",
                description="How to parse the file content",
                required=False,
                default="auto",
                enum_values=["auto", "text", "json", "csv", "yaml", "lines"]
            ),
            ToolParameter(
                name="start_line",
                param_type="number",
                description="Start reading from this line (1-indexed)",
                required=False,
                default=1,
                min_value=1
            ),
            ToolParameter(
                name="max_lines",
                param_type="number",
                description="Maximum number of lines to read",
                required=False,
                default=1000,
                min_value=1,
                max_value=100000
            )
        ]

    async def _execute(self, file_path: str, encoding: str = "utf-8",
                      parse_format: str = "auto", start_line: int = 1,
                      max_lines: int = 1000) -> Dict[str, Any]:
        """Read and parse a file."""
        # Validate file path
        path = Path(file_path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not path.is_file():
            raise ValueError(f"Not a file: {file_path}")

        # Check file size
        file_size = path.stat().st_size
        if file_size > self.max_file_size:
            raise ValueError(f"File too large: {file_size} bytes (max: {self.max_file_size})")

        # Check extension
        if self.allowed_extensions and path.suffix.lower() not in self.allowed_extensions:
            raise ValueError(f"File extension not allowed: {path.suffix}")

        # Determine format
        if parse_format == "auto":
            parse_format = self._detect_format(path)

        # Read file
        content = self._read_file(path, encoding, start_line, max_lines)

        # Parse content
        parsed = self._parse_content(content, parse_format, path)

        return {
            "file_path": str(path),
            "file_name": path.name,
            "file_size": file_size,
            "format": parse_format,
            "content": parsed,
            "mime_type": mimetypes.guess_type(str(path))[0]
        }

    def _detect_format(self, path: Path) -> str:
        """Detect file format from extension."""
        extension = path.suffix.lower()

        format_map = {
            ".json": "json",
            ".csv": "csv",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".txt": "text",
            ".md": "text",
            ".log": "lines",
            ".py": "text",
            ".js": "text",
            ".ts": "text",
            ".html": "text",
            ".xml": "text",
            ".css": "text",
        }

        return format_map.get(extension, "text")

    def _read_file(self, path: Path, encoding: str,
                   start_line: int, max_lines: int) -> str:
        """Read file content."""
        try:
            with open(path, 'r', encoding=encoding) as f:
                # Read with line limits
                lines = []
                for i, line in enumerate(f, 1):
                    if i < start_line:
                        continue
                    if len(lines) >= max_lines:
                        break
                    lines.append(line)
                return ''.join(lines)
        except UnicodeDecodeError:
            # Try binary read
            with open(path, 'rb') as f:
                content = f.read()
                return f"[Binary file: {len(content)} bytes]"

    def _parse_content(self, content: str, parse_format: str, path: Path) -> Any:
        """Parse content based on format."""
        if parse_format == "json":
            return json.loads(content)

        elif parse_format == "csv":
            import io
            reader = csv.DictReader(io.StringIO(content))
            return list(reader)

        elif parse_format == "yaml":
            import yaml
            return yaml.safe_load(content)

        elif parse_format == "lines":
            return content.split('\n')

        else:  # text
            return content


class DirectoryReaderTool(BaseTool):
    """
    Tool for reading directory contents and file metadata.
    """

    def __init__(self):
        config = ToolConfig(
            name="directory_reader",
            description="List directory contents with file metadata.",
            category=ToolCategory.DATA,
            timeout=30.0
        )
        super().__init__(config)

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="directory_path",
                param_type="string",
                description="Path to the directory to read",
                required=True
            ),
            ToolParameter(
                name="pattern",
                param_type="string",
                description="Glob pattern to filter files",
                required=False,
                default="*"
            ),
            ToolParameter(
                name="recursive",
                param_type="boolean",
                description="Whether to read subdirectories",
                required=False,
                default=False
            ),
            ToolParameter(
                name="include_hidden",
                param_type="boolean",
                description="Whether to include hidden files",
                required=False,
                default=False
            ),
            ToolParameter(
                name="max_depth",
                param_type="number",
                description="Maximum recursion depth",
                required=False,
                default=3,
                min_value=1,
                max_value=10
            )
        ]

    async def _execute(self, directory_path: str, pattern: str = "*",
                      recursive: bool = False, include_hidden: bool = False,
                      max_depth: int = 3) -> Dict[str, Any]:
        """Read directory contents."""
        path = Path(directory_path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        if not path.is_dir():
            raise ValueError(f"Not a directory: {directory_path}")

        entries = self._scan_directory(path, pattern, recursive, include_hidden, max_depth, 0)

        # Calculate summary
        total_files = sum(1 for e in entries if e["type"] == "file")
        total_dirs = sum(1 for e in entries if e["type"] == "directory")
        total_size = sum(e["size"] for e in entries if e["type"] == "file")

        return {
            "directory_path": str(path),
            "total_files": total_files,
            "total_directories": total_dirs,
            "total_size": total_size,
            "entries": entries
        }

    def _scan_directory(self, path: Path, pattern: str, recursive: bool,
                       include_hidden: bool, max_depth: int, current_depth: int) -> List[Dict]:
        """Scan directory and return entries."""
        entries = []

        try:
            for entry in path.glob(pattern):
                # Skip hidden files
                if not include_hidden and entry.name.startswith('.'):
                    continue

                stat = entry.stat()

                entry_info = {
                    "name": entry.name,
                    "path": str(entry),
                    "type": "directory" if entry.is_dir() else "file",
                    "size": stat.st_size if entry.is_file() else 0,
                    "modified": stat.st_mtime,
                    "created": stat.st_ctime,
                }

                if entry.is_file():
                    entry_info["extension"] = entry.suffix
                    entry_info["mime_type"] = mimetypes.guess_type(str(entry))[0]

                entries.append(entry_info)

                # Recurse into directories
                if recursive and entry.is_dir() and current_depth < max_depth:
                    sub_entries = self._scan_directory(
                        entry, "*", recursive, include_hidden,
                        max_depth, current_depth + 1
                    )
                    entries.extend(sub_entries)

        except PermissionError:
            pass  # Skip directories we can't access

        return entries


class CodeFileReaderTool(BaseTool):
    """
    Specialized tool for reading source code files with syntax awareness.
    """

    def __init__(self):
        config = ToolConfig(
            name="code_file_reader",
            description="Read source code files with syntax analysis.",
            category=ToolCategory.DATA,
            timeout=30.0
        )
        super().__init__(config)

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="file_path",
                param_type="string",
                description="Path to the source code file",
                required=True
            ),
            ToolParameter(
                name="include_line_numbers",
                param_type="boolean",
                description="Include line numbers in output",
                required=False,
                default=True
            ),
            ToolParameter(
                name="extract_functions",
                param_type="boolean",
                description="Extract function definitions",
                required=False,
                default=False
            ),
            ToolParameter(
                name="extract_classes",
                param_type="boolean",
                description="Extract class definitions",
                required=False,
                default=False
            )
        ]

    async def _execute(self, file_path: str, include_line_numbers: bool = True,
                      extract_functions: bool = False,
                      extract_classes: bool = False) -> Dict[str, Any]:
        """Read and analyze source code file."""
        path = Path(file_path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')

        result = {
            "file_path": str(path),
            "file_name": path.name,
            "language": self._detect_language(path),
            "total_lines": len(lines),
            "non_empty_lines": sum(1 for line in lines if line.strip()),
        }

        if include_line_numbers:
            result["content"] = '\n'.join(
                f"{i:4d}: {line}" for i, line in enumerate(lines, 1)
            )
        else:
            result["content"] = content

        if extract_functions:
            result["functions"] = self._extract_functions(content, result["language"])

        if extract_classes:
            result["classes"] = self._extract_classes(content, result["language"])

        return result

    def _detect_language(self, path: Path) -> str:
        """Detect programming language from extension."""
        extension_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".java": "java",
            ".c": "c",
            ".cpp": "cpp",
            ".h": "c",
            ".hpp": "cpp",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".cs": "csharp",
        }
        return extension_map.get(path.suffix.lower(), "unknown")

    def _extract_functions(self, content: str, language: str) -> List[Dict]:
        """Extract function definitions from code."""
        import re

        functions = []
        patterns = {
            "python": r"(?:async\s+)?def\s+(\w+)\s*\([^)]*\)",
            "javascript": r"(?:async\s+)?function\s+(\w+)\s*\([^)]*\)|const\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>",
            "typescript": r"(?:async\s+)?function\s+(\w+)\s*\([^)]*\)|const\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>",
            "java": r"(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\([^)]*\)",
            "go": r"func\s+(?:\([^)]+\)\s+)?(\w+)\s*\([^)]*\)",
        }

        pattern = patterns.get(language)
        if pattern:
            for match in re.finditer(pattern, content):
                # Get the first non-None group
                name = next((g for g in match.groups() if g), None)
                if name:
                    functions.append({
                        "name": name,
                        "position": match.start()
                    })

        return functions

    def _extract_classes(self, content: str, language: str) -> List[Dict]:
        """Extract class definitions from code."""
        import re

        classes = []
        patterns = {
            "python": r"class\s+(\w+)\s*(?:\([^)]*\))?:",
            "javascript": r"class\s+(\w+)\s*(?:extends\s+\w+\s*)?{",
            "typescript": r"class\s+(\w+)\s*(?:extends\s+\w+\s*)?(?:implements\s+[^{]+)?{",
            "java": r"class\s+(\w+)\s*(?:extends\s+\w+\s*)?(?:implements\s+[^{]+)?{",
        }

        pattern = patterns.get(language)
        if pattern:
            for match in re.finditer(pattern, content):
                classes.append({
                    "name": match.group(1),
                    "position": match.start()
                })

        return classes

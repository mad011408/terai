"""
Output formatting and markdown rendering utilities.
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import re
import textwrap


class OutputFormat(Enum):
    """Output format types."""
    PLAIN = "plain"
    MARKDOWN = "markdown"
    JSON = "json"
    TABLE = "table"
    TREE = "tree"


class Color(Enum):
    """ANSI color codes."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"

    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"


@dataclass
class TableColumn:
    """Table column configuration."""
    name: str
    width: Optional[int] = None
    align: str = "left"  # left, right, center
    color: Optional[Color] = None


class TextFormatter:
    """
    Formats text with colors and styles.
    """

    def __init__(self, use_colors: bool = True):
        self.use_colors = use_colors

    def colorize(self, text: str, color: Color) -> str:
        """Apply color to text."""
        if not self.use_colors:
            return text
        return f"{color.value}{text}{Color.RESET.value}"

    def bold(self, text: str) -> str:
        """Make text bold."""
        if not self.use_colors:
            return text
        return f"{Color.BOLD.value}{text}{Color.RESET.value}"

    def dim(self, text: str) -> str:
        """Make text dim."""
        if not self.use_colors:
            return text
        return f"{Color.DIM.value}{text}{Color.RESET.value}"

    def italic(self, text: str) -> str:
        """Make text italic."""
        if not self.use_colors:
            return text
        return f"{Color.ITALIC.value}{text}{Color.RESET.value}"

    def underline(self, text: str) -> str:
        """Underline text."""
        if not self.use_colors:
            return text
        return f"{Color.UNDERLINE.value}{text}{Color.RESET.value}"

    def success(self, text: str) -> str:
        """Format success message."""
        return self.colorize(text, Color.GREEN)

    def error(self, text: str) -> str:
        """Format error message."""
        return self.colorize(text, Color.RED)

    def warning(self, text: str) -> str:
        """Format warning message."""
        return self.colorize(text, Color.YELLOW)

    def info(self, text: str) -> str:
        """Format info message."""
        return self.colorize(text, Color.CYAN)

    def highlight(self, text: str) -> str:
        """Highlight text."""
        return self.colorize(self.bold(text), Color.MAGENTA)


class MarkdownRenderer:
    """
    Renders markdown to terminal-friendly output.
    """

    def __init__(self, formatter: Optional[TextFormatter] = None, width: int = 80):
        self.formatter = formatter or TextFormatter()
        self.width = width

    def render(self, markdown: str) -> str:
        """Render markdown to formatted text."""
        lines = markdown.split('\n')
        rendered_lines = []
        in_code_block = False
        code_language = ""
        code_buffer = []

        for line in lines:
            # Code block handling
            if line.startswith('```'):
                if not in_code_block:
                    in_code_block = True
                    code_language = line[3:].strip()
                    code_buffer = []
                else:
                    in_code_block = False
                    rendered_lines.append(self._render_code_block(
                        '\n'.join(code_buffer), code_language
                    ))
                continue

            if in_code_block:
                code_buffer.append(line)
                continue

            # Headers
            if line.startswith('######'):
                rendered_lines.append(self._render_h6(line[6:].strip()))
            elif line.startswith('#####'):
                rendered_lines.append(self._render_h5(line[5:].strip()))
            elif line.startswith('####'):
                rendered_lines.append(self._render_h4(line[4:].strip()))
            elif line.startswith('###'):
                rendered_lines.append(self._render_h3(line[3:].strip()))
            elif line.startswith('##'):
                rendered_lines.append(self._render_h2(line[2:].strip()))
            elif line.startswith('#'):
                rendered_lines.append(self._render_h1(line[1:].strip()))

            # Horizontal rule
            elif line.strip() in ['---', '***', '___']:
                rendered_lines.append(self._render_hr())

            # Unordered list
            elif line.strip().startswith('- ') or line.strip().startswith('* '):
                indent = len(line) - len(line.lstrip())
                content = line.strip()[2:]
                rendered_lines.append(self._render_list_item(content, indent))

            # Ordered list
            elif re.match(r'^\s*\d+\.\s', line):
                match = re.match(r'^(\s*)(\d+)\.\s(.*)$', line)
                if match:
                    indent = len(match.group(1))
                    number = match.group(2)
                    content = match.group(3)
                    rendered_lines.append(self._render_ordered_item(content, number, indent))

            # Blockquote
            elif line.strip().startswith('>'):
                content = line.strip()[1:].strip()
                rendered_lines.append(self._render_blockquote(content))

            # Regular paragraph
            else:
                rendered_lines.append(self._render_inline(line))

        return '\n'.join(rendered_lines)

    def _render_h1(self, text: str) -> str:
        """Render H1 header."""
        formatted = self.formatter.bold(self.formatter.colorize(text.upper(), Color.CYAN))
        return f"\n{formatted}\n{'=' * min(len(text), self.width)}\n"

    def _render_h2(self, text: str) -> str:
        """Render H2 header."""
        formatted = self.formatter.bold(self.formatter.colorize(text, Color.CYAN))
        return f"\n{formatted}\n{'-' * min(len(text), self.width)}\n"

    def _render_h3(self, text: str) -> str:
        """Render H3 header."""
        formatted = self.formatter.bold(self.formatter.colorize(f"### {text}", Color.BLUE))
        return f"\n{formatted}\n"

    def _render_h4(self, text: str) -> str:
        """Render H4 header."""
        formatted = self.formatter.bold(f"#### {text}")
        return f"\n{formatted}\n"

    def _render_h5(self, text: str) -> str:
        """Render H5 header."""
        formatted = self.formatter.underline(f"##### {text}")
        return f"{formatted}\n"

    def _render_h6(self, text: str) -> str:
        """Render H6 header."""
        return f"{self.formatter.dim(f'###### {text}')}\n"

    def _render_hr(self) -> str:
        """Render horizontal rule."""
        return self.formatter.dim('─' * self.width)

    def _render_list_item(self, text: str, indent: int = 0) -> str:
        """Render unordered list item."""
        bullet = self.formatter.colorize('•', Color.CYAN)
        return f"{' ' * indent}{bullet} {self._render_inline(text)}"

    def _render_ordered_item(self, text: str, number: str, indent: int = 0) -> str:
        """Render ordered list item."""
        num = self.formatter.colorize(f"{number}.", Color.CYAN)
        return f"{' ' * indent}{num} {self._render_inline(text)}"

    def _render_blockquote(self, text: str) -> str:
        """Render blockquote."""
        bar = self.formatter.colorize('│', Color.DIM)
        content = self.formatter.italic(text)
        return f"  {bar} {content}"

    def _render_code_block(self, code: str, language: str = "") -> str:
        """Render code block."""
        header = self.formatter.dim(f"┌─ {language or 'code'} " + "─" * (self.width - len(language) - 5))
        footer = self.formatter.dim("└" + "─" * (self.width - 1))

        lines = []
        lines.append(header)
        for line in code.split('\n'):
            bar = self.formatter.dim('│')
            lines.append(f"{bar} {self.formatter.colorize(line, Color.GREEN)}")
        lines.append(footer)

        return '\n'.join(lines)

    def _render_inline(self, text: str) -> str:
        """Render inline markdown elements."""
        # Bold
        text = re.sub(
            r'\*\*(.+?)\*\*',
            lambda m: self.formatter.bold(m.group(1)),
            text
        )
        text = re.sub(
            r'__(.+?)__',
            lambda m: self.formatter.bold(m.group(1)),
            text
        )

        # Italic
        text = re.sub(
            r'\*(.+?)\*',
            lambda m: self.formatter.italic(m.group(1)),
            text
        )
        text = re.sub(
            r'_(.+?)_',
            lambda m: self.formatter.italic(m.group(1)),
            text
        )

        # Inline code
        text = re.sub(
            r'`(.+?)`',
            lambda m: self.formatter.colorize(m.group(1), Color.GREEN),
            text
        )

        # Links [text](url)
        text = re.sub(
            r'\[(.+?)\]\((.+?)\)',
            lambda m: f"{self.formatter.underline(m.group(1))} ({self.formatter.dim(m.group(2))})",
            text
        )

        # Strikethrough
        text = re.sub(
            r'~~(.+?)~~',
            lambda m: self.formatter.dim(m.group(1)),
            text
        )

        return text


class TableFormatter:
    """
    Formats data as tables.
    """

    def __init__(self, formatter: Optional[TextFormatter] = None):
        self.formatter = formatter or TextFormatter()

    def format(self, data: List[Dict[str, Any]],
               columns: Optional[List[TableColumn]] = None) -> str:
        """Format data as a table."""
        if not data:
            return ""

        # Auto-detect columns if not provided
        if not columns:
            columns = [TableColumn(name=key) for key in data[0].keys()]

        # Calculate column widths
        widths = {}
        for col in columns:
            if col.width:
                widths[col.name] = col.width
            else:
                max_width = len(col.name)
                for row in data:
                    value = str(row.get(col.name, ""))
                    max_width = max(max_width, len(value))
                widths[col.name] = min(max_width, 50)  # Cap at 50

        # Build table
        lines = []

        # Header
        header_cells = []
        for col in columns:
            cell = self._align_text(col.name, widths[col.name], col.align)
            header_cells.append(self.formatter.bold(cell))
        lines.append("│ " + " │ ".join(header_cells) + " │")

        # Separator
        sep_cells = ["─" * widths[col.name] for col in columns]
        lines.append("├─" + "─┼─".join(sep_cells) + "─┤")

        # Data rows
        for row in data:
            row_cells = []
            for col in columns:
                value = str(row.get(col.name, ""))
                cell = self._align_text(value[:widths[col.name]], widths[col.name], col.align)
                if col.color:
                    cell = self.formatter.colorize(cell, col.color)
                row_cells.append(cell)
            lines.append("│ " + " │ ".join(row_cells) + " │")

        # Top border
        top_cells = ["─" * widths[col.name] for col in columns]
        lines.insert(0, "┌─" + "─┬─".join(top_cells) + "─┐")

        # Bottom border
        bottom_cells = ["─" * widths[col.name] for col in columns]
        lines.append("└─" + "─┴─".join(bottom_cells) + "─┘")

        return "\n".join(lines)

    def _align_text(self, text: str, width: int, align: str) -> str:
        """Align text within width."""
        if align == "right":
            return text.rjust(width)
        elif align == "center":
            return text.center(width)
        else:
            return text.ljust(width)


class TreeFormatter:
    """
    Formats hierarchical data as trees.
    """

    def __init__(self, formatter: Optional[TextFormatter] = None):
        self.formatter = formatter or TextFormatter()

    def format(self, data: Dict[str, Any], prefix: str = "", is_last: bool = True) -> str:
        """Format data as a tree."""
        lines = []

        if isinstance(data, dict):
            items = list(data.items())
            for i, (key, value) in enumerate(items):
                is_last_item = (i == len(items) - 1)
                connector = "└── " if is_last_item else "├── "
                extension = "    " if is_last_item else "│   "

                key_str = self.formatter.bold(str(key))

                if isinstance(value, dict):
                    lines.append(f"{prefix}{connector}{key_str}/")
                    lines.append(self.format(value, prefix + extension, is_last_item))
                elif isinstance(value, list):
                    lines.append(f"{prefix}{connector}{key_str}/")
                    for j, item in enumerate(value):
                        is_last_list = (j == len(value) - 1)
                        list_connector = "└── " if is_last_list else "├── "
                        if isinstance(item, dict):
                            lines.append(f"{prefix}{extension}{list_connector}[{j}]")
                            lines.append(self.format(item, prefix + extension + ("    " if is_last_list else "│   ")))
                        else:
                            lines.append(f"{prefix}{extension}{list_connector}{item}")
                else:
                    value_str = self.formatter.colorize(str(value), Color.GREEN)
                    lines.append(f"{prefix}{connector}{key_str}: {value_str}")

        return "\n".join(filter(None, lines))


class ProgressBar:
    """
    Creates progress bars.
    """

    def __init__(self, total: int, width: int = 40,
                 formatter: Optional[TextFormatter] = None):
        self.total = total
        self.width = width
        self.current = 0
        self.formatter = formatter or TextFormatter()

    def update(self, value: int) -> str:
        """Update progress and return bar string."""
        self.current = min(value, self.total)
        return self.render()

    def increment(self, amount: int = 1) -> str:
        """Increment progress and return bar string."""
        return self.update(self.current + amount)

    def render(self) -> str:
        """Render the progress bar."""
        if self.total == 0:
            percentage = 100
        else:
            percentage = (self.current / self.total) * 100

        filled = int(self.width * self.current / self.total) if self.total > 0 else self.width
        empty = self.width - filled

        bar = "█" * filled + "░" * empty

        # Color based on progress
        if percentage < 33:
            bar = self.formatter.colorize(bar, Color.RED)
        elif percentage < 66:
            bar = self.formatter.colorize(bar, Color.YELLOW)
        else:
            bar = self.formatter.colorize(bar, Color.GREEN)

        return f"[{bar}] {percentage:5.1f}% ({self.current}/{self.total})"


class Spinner:
    """
    Creates spinner animations.
    """

    SPINNERS = {
        "dots": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
        "line": ["-", "\\", "|", "/"],
        "circle": ["◐", "◓", "◑", "◒"],
        "square": ["◰", "◳", "◲", "◱"],
        "arrow": ["←", "↖", "↑", "↗", "→", "↘", "↓", "↙"],
        "bounce": ["⠁", "⠂", "⠄", "⠂"],
        "growing": ["▁", "▃", "▄", "▅", "▆", "▇", "█", "▇", "▆", "▅", "▄", "▃"],
    }

    def __init__(self, style: str = "dots", formatter: Optional[TextFormatter] = None):
        self.frames = self.SPINNERS.get(style, self.SPINNERS["dots"])
        self.current = 0
        self.formatter = formatter or TextFormatter()

    def next(self) -> str:
        """Get next spinner frame."""
        frame = self.frames[self.current]
        self.current = (self.current + 1) % len(self.frames)
        return self.formatter.colorize(frame, Color.CYAN)

    def reset(self) -> None:
        """Reset spinner to first frame."""
        self.current = 0


class OutputFormatter:
    """
    Main output formatter combining all formatters.
    """

    def __init__(self, use_colors: bool = True, width: int = 80):
        self.text_formatter = TextFormatter(use_colors)
        self.markdown_renderer = MarkdownRenderer(self.text_formatter, width)
        self.table_formatter = TableFormatter(self.text_formatter)
        self.tree_formatter = TreeFormatter(self.text_formatter)
        self.width = width

    def format(self, content: Any, format_type: OutputFormat = OutputFormat.PLAIN) -> str:
        """Format content according to type."""
        if format_type == OutputFormat.MARKDOWN:
            return self.markdown_renderer.render(str(content))
        elif format_type == OutputFormat.JSON:
            import json
            return self._format_json(content)
        elif format_type == OutputFormat.TABLE:
            if isinstance(content, list):
                return self.table_formatter.format(content)
            return str(content)
        elif format_type == OutputFormat.TREE:
            if isinstance(content, dict):
                return self.tree_formatter.format(content)
            return str(content)
        else:
            return str(content)

    def _format_json(self, data: Any) -> str:
        """Format JSON with syntax highlighting."""
        import json
        json_str = json.dumps(data, indent=2, default=str)

        # Highlight JSON syntax
        json_str = re.sub(
            r'"([^"]+)":',
            lambda m: self.text_formatter.colorize(f'"{m.group(1)}"', Color.CYAN) + ':',
            json_str
        )
        json_str = re.sub(
            r': "([^"]*)"',
            lambda m: ': ' + self.text_formatter.colorize(f'"{m.group(1)}"', Color.GREEN),
            json_str
        )
        json_str = re.sub(
            r': (\d+)',
            lambda m: f': {self.text_formatter.colorize(m.group(1), Color.YELLOW)}',
            json_str
        )
        json_str = re.sub(
            r': (true|false|null)',
            lambda m: f': {self.text_formatter.colorize(m.group(1), Color.MAGENTA)}',
            json_str
        )

        return json_str

    def box(self, content: str, title: Optional[str] = None,
            style: str = "single") -> str:
        """Wrap content in a box."""
        if style == "double":
            chars = {"tl": "╔", "tr": "╗", "bl": "╚", "br": "╝", "h": "═", "v": "║"}
        elif style == "rounded":
            chars = {"tl": "╭", "tr": "╮", "bl": "╰", "br": "╯", "h": "─", "v": "│"}
        else:
            chars = {"tl": "┌", "tr": "┐", "bl": "└", "br": "┘", "h": "─", "v": "│"}

        lines = content.split('\n')
        max_len = max(len(line) for line in lines) if lines else 0

        if title:
            max_len = max(max_len, len(title) + 4)

        result = []

        # Top border
        if title:
            title_str = f" {title} "
            padding = max_len - len(title_str)
            result.append(
                f"{chars['tl']}{chars['h']}{self.text_formatter.bold(title_str)}"
                f"{chars['h'] * padding}{chars['h']}{chars['tr']}"
            )
        else:
            result.append(f"{chars['tl']}{chars['h'] * (max_len + 2)}{chars['tr']}")

        # Content
        for line in lines:
            padding = max_len - len(line)
            result.append(f"{chars['v']} {line}{' ' * padding} {chars['v']}")

        # Bottom border
        result.append(f"{chars['bl']}{chars['h'] * (max_len + 2)}{chars['br']}")

        return '\n'.join(result)

    def divider(self, char: str = "─", label: Optional[str] = None) -> str:
        """Create a divider line."""
        if label:
            padding = (self.width - len(label) - 2) // 2
            return f"{char * padding} {label} {char * padding}"
        return char * self.width

    def indent(self, text: str, spaces: int = 2) -> str:
        """Indent text."""
        return textwrap.indent(text, ' ' * spaces)

    def wrap(self, text: str, width: Optional[int] = None) -> str:
        """Wrap text to width."""
        return textwrap.fill(text, width or self.width)

    def truncate(self, text: str, max_length: int, suffix: str = "...") -> str:
        """Truncate text to max length."""
        if len(text) <= max_length:
            return text
        return text[:max_length - len(suffix)] + suffix

    def columns(self, items: List[str], num_columns: int = 3) -> str:
        """Format items in columns."""
        if not items:
            return ""

        col_width = self.width // num_columns
        rows = []

        for i in range(0, len(items), num_columns):
            row_items = items[i:i + num_columns]
            row = ""
            for item in row_items:
                row += item.ljust(col_width)
            rows.append(row.rstrip())

        return '\n'.join(rows)


"""
Streaming response handler for real-time output.
"""

from typing import Any, AsyncGenerator, Callable, Optional
import asyncio
import sys


class StreamHandler:
    """
    Handles streaming responses from LLMs.
    """

    def __init__(self, on_token: Optional[Callable[[str], None]] = None,
                 on_complete: Optional[Callable[[str], None]] = None):
        self.on_token = on_token or self._default_token_handler
        self.on_complete = on_complete
        self.buffer = []
        self.total_content = ""

    def _default_token_handler(self, token: str) -> None:
        """Default token handler - print to stdout."""
        print(token, end="", flush=True)

    async def handle_stream(self, stream: AsyncGenerator[str, None]) -> str:
        """
        Handle a streaming response.

        Args:
            stream: Async generator of tokens

        Returns:
            Complete response text
        """
        self.buffer = []
        self.total_content = ""

        async for token in stream:
            self.buffer.append(token)
            self.total_content += token
            self.on_token(token)

        if self.on_complete:
            self.on_complete(self.total_content)

        return self.total_content

    def get_content(self) -> str:
        """Get accumulated content."""
        return self.total_content


class StreamRenderer:
    """
    Renders streaming content with formatting.
    """

    def __init__(self, console: Any = None):
        self.console = console
        self.current_line = ""
        self.in_code_block = False
        self.code_language = ""
        self.code_buffer = []

    def render_token(self, token: str) -> None:
        """
        Render a single token with formatting.

        Args:
            token: The token to render
        """
        self.current_line += token

        # Check for code block markers
        if "```" in self.current_line:
            if not self.in_code_block:
                # Starting code block
                self.in_code_block = True
                parts = self.current_line.split("```")
                if len(parts) > 1:
                    self.code_language = parts[1].split("\n")[0].strip()
                self._print_text(parts[0])
                self.current_line = ""
                self.code_buffer = []
            else:
                # Ending code block
                self.in_code_block = False
                self._print_code_block()
                self.current_line = self.current_line.split("```")[-1]
                self.code_buffer = []
            return

        if self.in_code_block:
            self.code_buffer.append(token)
        else:
            self._print_token(token)

    def _print_token(self, token: str) -> None:
        """Print a token."""
        if self.console:
            self.console.print(token, end="")
        else:
            print(token, end="", flush=True)

    def _print_text(self, text: str) -> None:
        """Print text."""
        if self.console:
            self.console.print(text, end="")
        else:
            print(text, end="", flush=True)

    def _print_code_block(self) -> None:
        """Print accumulated code block."""
        code = "".join(self.code_buffer).strip()
        if self.console and hasattr(self.console, 'print_code'):
            self.console.print_code(code, self.code_language or "text")
        else:
            print(f"\n```{self.code_language}")
            print(code)
            print("```\n")

    def finish(self) -> None:
        """Finish rendering."""
        if self.in_code_block and self.code_buffer:
            self._print_code_block()
        print()  # Final newline


class TypewriterEffect:
    """
    Creates typewriter effect for text output.
    """

    def __init__(self, delay: float = 0.02, console: Any = None):
        self.delay = delay
        self.console = console

    async def print(self, text: str) -> None:
        """
        Print text with typewriter effect.

        Args:
            text: Text to print
        """
        for char in text:
            if self.console:
                self.console.print(char, end="")
            else:
                print(char, end="", flush=True)
            await asyncio.sleep(self.delay)
        print()  # Final newline

    async def print_lines(self, lines: list, line_delay: float = 0.1) -> None:
        """
        Print lines with typewriter effect.

        Args:
            lines: Lines to print
            line_delay: Delay between lines
        """
        for line in lines:
            await self.print(line)
            await asyncio.sleep(line_delay)


class ProgressStream:
    """
    Shows progress while streaming.
    """

    def __init__(self, total_expected: Optional[int] = None):
        self.total_expected = total_expected
        self.received = 0
        self.tokens = []

    def update(self, token: str) -> None:
        """Update with new token."""
        self.tokens.append(token)
        self.received += len(token)
        self._show_progress()

    def _show_progress(self) -> None:
        """Show progress indicator."""
        if self.total_expected:
            percentage = min(100, (self.received / self.total_expected) * 100)
            bar_length = 30
            filled = int(bar_length * percentage / 100)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"\r[{bar}] {percentage:.1f}%", end="", flush=True)
        else:
            # Show token count
            print(f"\rTokens received: {len(self.tokens)}", end="", flush=True)

    def finish(self) -> str:
        """Finish and return content."""
        print()  # Clear progress line
        return "".join(self.tokens)


class StreamBuffer:
    """
    Buffer for streaming content with line-based output.
    """

    def __init__(self, on_line: Optional[Callable[[str], None]] = None):
        self.buffer = ""
        self.lines = []
        self.on_line = on_line

    def add(self, token: str) -> Optional[str]:
        """
        Add token to buffer.

        Returns:
            Complete line if available
        """
        self.buffer += token

        # Check for complete lines
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self.lines.append(line)
            if self.on_line:
                self.on_line(line)
            return line

        return None

    def flush(self) -> str:
        """Flush remaining buffer."""
        if self.buffer:
            self.lines.append(self.buffer)
            if self.on_line:
                self.on_line(self.buffer)
            remaining = self.buffer
            self.buffer = ""
            return remaining
        return ""

    def get_all(self) -> str:
        """Get all content."""
        return "\n".join(self.lines) + ("\n" + self.buffer if self.buffer else "")

"""
Rich terminal interface for the AI agent.
Premium NEXUS-style UI with advanced graphics.
"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
import sys
import asyncio
import os
from datetime import datetime

try:
    from rich.console import Console as RichConsole
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    from rich.box import DOUBLE, HEAVY, ROUNDED
    from rich.align import Align
    from rich.columns import Columns
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# NEXUS AI GRAPHICS & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

NEXUS_LOGO = """
                  ▐▛███▜▌
                 ▝▜█████▛▘
                   ▘▘ ▝▝
"""

NEXUS_LOGO_SMALL = "◆ NEXUS"

BRAND_NAME = "NEXUS AI"
VERSION = "v2.0.0"

# Color scheme
COLORS = {
    "primary": "#00D4FF",       # Cyan
    "secondary": "#FF6B6B",     # Coral
    "accent": "#A855F7",        # Purple
    "success": "#10B981",       # Green
    "warning": "#F59E0B",       # Amber
    "error": "#EF4444",         # Red
    "muted": "#6B7280",         # Gray
    "bg": "#1a1a2e",            # Dark blue
}


class Console:
    """
    Console abstraction for terminal output.
    Uses rich if available, falls back to basic print.
    """

    def __init__(self, use_rich: bool = True):
        self.use_rich = use_rich and RICH_AVAILABLE
        if self.use_rich:
            self.console = RichConsole()
        else:
            self.console = None

    def print(self, *args, **kwargs) -> None:
        """Print to console."""
        if self.use_rich:
            self.console.print(*args, **kwargs)
        else:
            print(*args)

    def print_markdown(self, text: str) -> None:
        """Print markdown formatted text."""
        if self.use_rich:
            self.console.print(Markdown(text))
        else:
            print(text)

    def print_code(self, code: str, language: str = "python") -> None:
        """Print syntax-highlighted code."""
        if self.use_rich:
            self.console.print(Syntax(code, language, theme="monokai", line_numbers=True))
        else:
            print(f"```{language}\n{code}\n```")

    def print_panel(self, content: str, title: str = "", style: str = "blue") -> None:
        """Print content in a panel."""
        if self.use_rich:
            self.console.print(Panel(content, title=title, border_style=style))
        else:
            print(f"\n=== {title} ===\n{content}\n{'=' * (len(title) + 8)}\n")

    def print_error(self, message: str) -> None:
        """Print error message."""
        if self.use_rich:
            self.console.print(f"[bold red]✗ Error:[/bold red] {message}")
        else:
            print(f"Error: {message}", file=sys.stderr)

    def print_success(self, message: str) -> None:
        """Print success message."""
        if self.use_rich:
            self.console.print(f"[bold green]✓ Success:[/bold green] {message}")
        else:
            print(f"Success: {message}")

    def print_warning(self, message: str) -> None:
        """Print warning message."""
        if self.use_rich:
            self.console.print(f"[bold yellow]⚠ Warning:[/bold yellow] {message}")
        else:
            print(f"Warning: {message}")

    def print_info(self, message: str) -> None:
        """Print info message."""
        if self.use_rich:
            self.console.print(f"[bold cyan]◈ Info:[/bold cyan] {message}")
        else:
            print(f"Info: {message}")

    def print_table(self, headers: List[str], rows: List[List[str]],
                   title: str = "") -> None:
        """Print a table."""
        if self.use_rich:
            table = Table(title=title, box=ROUNDED, border_style="cyan")
            for header in headers:
                table.add_column(header, style="bold cyan")
            for row in rows:
                table.add_row(*row)
            self.console.print(table)
        else:
            # Simple ASCII table
            if title:
                print(f"\n{title}")
            print(" | ".join(headers))
            print("-" * (sum(len(h) for h in headers) + 3 * len(headers)))
            for row in rows:
                print(" | ".join(row))

    def input(self, prompt: str = "") -> str:
        """Get input from user."""
        if self.use_rich:
            return self.console.input(prompt)
        else:
            return input(prompt)

    def clear(self) -> None:
        """Clear the console."""
        if self.use_rich:
            self.console.clear()
        else:
            print("\033[H\033[J", end="")


class TerminalUI:
    """
    Full terminal UI for the AI agent.
    Premium NEXUS-style interface.
    """

    def __init__(self, title: str = "NEXUS AI"):
        self.title = title
        self.console = Console()
        self.spinner_active = False
        self._progress = None
        self._live = None
        self._prompt_session = None
        self.current_model = "claude-opus-4-5-20251101"
        self.current_path = os.getcwd()

        # Setup prompt_toolkit session with key bindings
        if PROMPT_TOOLKIT_AVAILABLE:
            self._setup_prompt_session()

    def _setup_prompt_session(self):
        """Setup prompt_toolkit session with Alt+Enter binding."""
        try:
            bindings = KeyBindings()

            @bindings.add('escape', 'enter')  # Alt+Enter
            def _(event):
                event.app.exit(result=event.app.current_buffer.text)

            @bindings.add('escape', 'escape')  # Double Escape to cancel
            def _(event):
                event.app.exit(result="")

            self._prompt_session = PromptSession(
                multiline=True,
                key_bindings=bindings,
            )
        except Exception:
            # Fallback if prompt_toolkit can't initialize (e.g., no console)
            self._prompt_session = None

    def _get_header_panel(self) -> str:
        """Generate the NEXUS AI header."""
        model_short = self.current_model.split("/")[-1] if "/" in self.current_model else self.current_model
        
        header = f"""
╭─── {BRAND_NAME} {VERSION} ──────────────────────────────────────────────────────────╮
│                                           │ Tips for getting started         │
│               Welcome back!               │ Run /help to see commands...     │
│                                           │ ────────────────────────────     │
│                  ▐▛███▜▌                  │ Recent activity                  │
│                 ▝▜█████▛▘                 │ Ready for your commands          │
│                   ▘▘ ▝▝                   │                                  │
│                                           │                                  │
│       {model_short[:20]:<20} · API Connected    │                                  │
│   {self.current_path[:40]:<40}│                                  │
╰──────────────────────────────────────────────────────────────────────────────╯
"""
        return header

    def show_welcome(self) -> None:
        """Show premium NEXUS welcome message."""
        if self.use_rich:
            self.console.print(f"\n[bold cyan]{self._get_header_panel()}[/bold cyan]")
            self.console.print("\n[bold cyan]◈[/bold cyan] Initializing NEXUS AI Agent...\n")
            self.console.print("[bold green]✓[/bold green] Ready! Type your message below.\n")
        else:
            print(self._get_header_panel())
            print("\n◈ Initializing NEXUS AI Agent...")
            print("✓ Ready! Type your message below.\n")

    @property
    def use_rich(self) -> bool:
        """Check if rich is available."""
        return self.console.use_rich

    def show_input_box(self) -> None:
        """Show the styled input box."""
        model_short = self.current_model.split("/")[-1] if "/" in self.current_model else self.current_model
        path_short = self.current_path[-50:] if len(self.current_path) > 50 else self.current_path
        
        box = f"""
  ████████████████████████████████████████████████████████████████████████████
  █══════════════════════════════════════════════════════════════════════════█▓
  █║  [bold cyan]◆ NEXUS AI[/bold cyan] │ [yellow]{model_short:<30}[/yellow]                        ║█▓
  █║  [dim]◈ {path_short:<68}[/dim]║█▓
  █╠══════════════════════════════════════════════════════════════════════════╣█▓
  █║  [dim]Press Enter to add line, Alt+Enter to SEND[/dim]                              ║█▓
"""
        if self.use_rich:
            self.console.print(box)

    def show_input_received(self) -> None:
        """Show input received confirmation."""
        box_end = """  █╠══════════════════════════════════════════════════════════════════════════╣█▓
  █║  [bold green]✓ Message received[/bold green]                                                    ║█▓
  █══════════════════════════════════════════════════════════════════════════█▓
  ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
"""
        if self.use_rich:
            self.console.print(box_end)

    async def show_prompt_async(self) -> str:
        """
        Async version: Show input prompt and get user input.
        - Enter = New line
        - Alt+Enter = Send message
        - Escape+Escape = Cancel
        """
        # Show input box
        self.show_input_box()
        
        if PROMPT_TOOLKIT_AVAILABLE and self._prompt_session:
            try:
                text = await self._prompt_session.prompt_async(
                    "  █║  ❯ ",
                )
                if text:
                    self.show_input_received()
                return text.strip() if text else ""

            except (EOFError, KeyboardInterrupt):
                return "/exit"
            except Exception as e:
                # Fallback to simple input
                print(f"\n[Prompt error: {e}]")
                return input("\n❯ ")
        else:
            # Fallback for systems without prompt_toolkit
            try:
                return input("  █║  ❯ ")
            except EOFError:
                return "/exit"

    def show_prompt(self) -> str:
        """
        Sync version: Show input prompt and get user input.
        Use show_prompt_async() in async contexts.
        """
        self.show_input_box()
        try:
            result = input("  █║  ❯ ")
            if result:
                self.show_input_received()
            return result
        except EOFError:
            return "/exit"

    def show_multiline_prompt(self) -> str:
        """Show multi-line input prompt. Enter twice (empty line) to send."""
        self.show_input_box()
        print("  █║  [dim]Multi-line mode - empty line to send[/dim]")
        lines = []
        while True:
            try:
                line = input("  █║  ... ")
                if line == "":
                    break
                lines.append(line)
            except EOFError:
                break
        if lines:
            self.show_input_received()
        return "\n".join(lines)

    def show_response(self, response: str, agent_name: str = "NEXUS") -> None:
        """Show agent response with styling."""
        if self.use_rich:
            self.console.print(f"\n[bold cyan]◆ {agent_name}:[/bold cyan]")
            self.console.print_markdown(response)
        else:
            print(f"\n◆ {agent_name}:")
            print(response)

    def show_thinking(self, thought: str) -> None:
        """Show agent thinking."""
        self.console.print(f"[dim italic]◈ Thinking: {thought}[/dim italic]")

    def show_tool_call(self, tool_name: str, params: Dict[str, Any]) -> None:
        """Show tool being called."""
        if self.use_rich:
            self.console.print(f"[bold magenta]⚡ Using tool:[/bold magenta] {tool_name}")
        else:
            print(f"⚡ Using tool: {tool_name}")
        if params:
            for key, value in params.items():
                display_value = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                self.console.print(f"  [dim]{key}:[/dim] {display_value}")

    def show_tool_result(self, result: Any) -> None:
        """Show tool result."""
        result_str = str(result)[:500]
        if len(str(result)) > 500:
            result_str += "..."
        self.console.print(f"[dim]◈ Result: {result_str}[/dim]")

    def start_spinner(self, message: str = "Processing...") -> None:
        """Start a spinner."""
        if RICH_AVAILABLE:
            self._progress = Progress(
                SpinnerColumn(spinner_name="dots12"),
                TextColumn("[cyan]{task.description}[/cyan]"),
                console=self.console.console
            )
            self._progress.add_task(description=f"◈ {message}", total=None)
            self._progress.start()
            self.spinner_active = True

    def stop_spinner(self) -> None:
        """Stop the spinner."""
        if self._progress:
            self._progress.stop()
            self._progress = None
            self.spinner_active = False

    def show_agents_list(self, agents: List[Dict[str, str]]) -> None:
        """Show list of available agents."""
        self.console.print("\n[bold cyan]╭─── Available Agents ───╮[/bold cyan]")
        for agent in agents:
            self.console.print(f"  [bold cyan]◆[/bold cyan] [bold]{agent['name']}[/bold]: {agent['description']}")
        self.console.print("[bold cyan]╰─────────────────────────╯[/bold cyan]")

    def show_tools_list(self, tools: List[Dict[str, str]]) -> None:
        """Show list of available tools."""
        headers = ["Tool", "Description", "Category"]
        rows = [[t["name"], t["description"][:40], t.get("category", "general")] for t in tools]
        self.console.print_table(headers, rows, title="⚡ Available Tools")

    def show_error(self, error: str) -> None:
        """Show error message."""
        self.console.print_error(error)

    def show_success(self, message: str) -> None:
        """Show success message."""
        self.console.print_success(message)

    def confirm(self, message: str) -> bool:
        """Ask for confirmation."""
        response = self.console.input(f"[yellow]?[/yellow] {message} [y/N]: ")
        return response.lower() in ["y", "yes"]

    def show_progress(self, current: int, total: int, description: str = "") -> None:
        """Show progress bar."""
        if RICH_AVAILABLE:
            if not self._progress:
                self._progress = Progress(console=self.console.console)
                self._task = self._progress.add_task(description, total=total)
                self._progress.start()
            self._progress.update(self._task, completed=current)
        else:
            percentage = (current / total) * 100
            print(f"\r{description}: {percentage:.1f}%", end="")

    def finish_progress(self) -> None:
        """Finish progress bar."""
        if self._progress:
            self._progress.stop()
            self._progress = None

    def show_code(self, code: str, language: str = "python") -> None:
        """Show syntax-highlighted code."""
        self.console.print_code(code, language)

    def show_diff(self, old: str, new: str) -> None:
        """Show diff between two texts."""
        import difflib
        diff = difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            lineterm=""
        )
        diff_text = "".join(diff)
        self.console.print_code(diff_text, "diff")

    def show_models_list(self, models: List[str], current: str) -> None:
        """Show available models with styling."""
        self.console.print("\n[bold cyan]╭─── Available Models ───╮[/bold cyan]")
        for model in models:
            if model == current:
                self.console.print(f"  [bold green]◆ {model} (current)[/bold green]")
            else:
                self.console.print(f"  [dim]◇ {model}[/dim]")
        self.console.print("[bold cyan]╰─────────────────────────╯[/bold cyan]")
        self.console.print(f"\n[dim]Use: model <name> to switch[/dim]")

    def show_help(self) -> None:
        """Show help with NEXUS styling."""
        help_text = """
╭─── NEXUS AI Commands ───────────────────────────────────────────────────────╮
│                                                                             │
│  [bold cyan]Input Controls[/bold cyan]                                                         │
│    Enter      = New line                                                    │
│    Alt+Enter  = Send message                                                │
│    Esc+Esc    = Cancel input                                                │
│                                                                             │
│  [bold cyan]Commands[/bold cyan]                                                               │
│    /help           - Show this help                                         │
│    /models         - List available models                                  │
│    /prompts        - List saved prompts                                     │
│    /prompt <name>  - Load a saved prompt                                    │
│    /system         - Show/set system prompt                                 │
│    /search <query> - Web search                                             │
│    /ml             - Multi-line input mode                                  │
│    /exit           - Exit the application                                   │
│                                                                             │
│  [bold magenta]Turbo Mode (Ultra Performance)[/bold magenta]                                      │
│    /turbo          - Toggle turbo mode ON/OFF                               │
│    /quality <lvl>  - Set quality: ultra, high, balanced, fast               │
│    /mode <type>    - Set mode: turbo, deep, creative, code, analysis        │
│    /stats          - Show performance statistics                            │
│                                                                             │
│  [bold cyan]Prompt Shortcuts[/bold cyan]                                                       │
│    coder, security, devops, researcher, creative                            │
│                                                                             │
╰─────────────────────────────────────────────────────────────────────────────╯
"""
        self.console.print(help_text)

    def set_model(self, model: str) -> None:
        """Set current model for display."""
        self.current_model = model


class InteractiveSession:
    """
    Interactive session manager.
    """

    def __init__(self, ui: TerminalUI, agent: Any):
        self.ui = ui
        self.agent = agent
        self.running = False
        self.history: List[Dict[str, str]] = []

    async def start(self) -> None:
        """Start interactive session."""
        self.running = True
        self.ui.show_welcome()

        while self.running:
            try:
                user_input = self.ui.show_prompt()

                if not user_input.strip():
                    continue

                if user_input.startswith("/"):
                    await self._handle_command(user_input)
                else:
                    await self._handle_message(user_input)

            except KeyboardInterrupt:
                self.ui.console.print("\n[yellow]⚠ Interrupted. Type /exit to quit.[/yellow]")
            except EOFError:
                break

    async def _handle_command(self, command: str) -> None:
        """Handle slash commands."""
        cmd = command.lower().strip()

        if cmd == "/exit" or cmd == "/quit":
            self.running = False
            self.ui.console.print("[cyan]◈ Goodbye! See you next time.[/cyan]")
        elif cmd == "/help":
            self.ui.show_help()
        elif cmd == "/clear":
            self.ui.console.clear()
        elif cmd == "/history":
            self._show_history()
        else:
            self.ui.show_error(f"Unknown command: {command}")

    async def _handle_message(self, message: str) -> None:
        """Handle user message."""
        self.history.append({"role": "user", "content": message})

        self.ui.start_spinner("Thinking...")

        try:
            from ..core.context import Context
            ctx = Context()
            response = await self.agent.run(message, ctx)

            self.ui.stop_spinner()

            result = response.result if hasattr(response, 'result') else str(response)
            self.ui.show_response(result)
            self.history.append({"role": "assistant", "content": result})

        except Exception as e:
            self.ui.stop_spinner()
            self.ui.show_error(str(e))

    def _show_history(self) -> None:
        """Show conversation history."""
        self.ui.console.print("\n[bold cyan]╭─── Conversation History ───╮[/bold cyan]")
        for msg in self.history[-10:]:
            role = msg["role"]
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            icon = "◆" if role == "assistant" else "❯"
            color = "cyan" if role == "assistant" else "white"
            self.ui.console.print(f"  [{color}]{icon} {role}:[/{color}] {content}")
        self.ui.console.print("[bold cyan]╰─────────────────────────────╯[/bold cyan]")

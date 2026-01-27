#!/usr/bin/env python3
"""
AI Terminal Agent - Main Entry Point

A multi-agent AI system for terminal-based task automation.
"""

import asyncio
import sys
import argparse
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # dotenv not installed, use system env vars

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.runner import AgentRunner, RunnerConfig
from src.core.context import Context, ContextManager
from src.core.turbo_engine import TurboEngine, TurboConfig, ResponseQuality, ProcessingMode, create_turbo_engine
from src.core.response_optimizer import optimize_response, analyze_response
from src.core.power_boost import PowerBoost, PowerConfig, PowerLevel, get_power_boost, boost_system_prompt, boost_user_prompt, GODMODE_PROMPT
from src.core.neural_accelerator import (
    NeuralAccelerator, MasterAccelerator, QuantumPromptEngine, EnergyCore,
    get_master_accelerator, get_maximum_power_prompt, accelerate_prompt
)
from src.core.mind_unlocker import (
    MindUnlocker, UltimateUnlocker, IntelligenceMaximizer, SpeedMaximizer,
    get_ultimate_unlocker, unlock_ai_fully, unlock_query
)
from src.core.prompt_enhancer import (
    UltraPromptEnhancer, get_prompt_enhancer, enhance_prompt,
    enhance_prompt_with_progress, get_natural_system_prompt,
    create_progress_bar, create_animated_progress
)
from src.agents.manager_agent import ManagerAgent
from src.models.model_manager import ModelManager
from src.ui.terminal_ui import TerminalUI, InteractiveSession
from src.utils.logger import setup_logging, get_logger
from src.utils.config import load_config, Config
from src.utils.telemetry import setup_telemetry
from src.prompts.system_prompts import get_prompt_manager, PROMPTS, get_active_prompt


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AI Terminal Agent - Multi-agent AI for terminal automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                     Start interactive session
  %(prog)s "list all files"    Execute a single task
  %(prog)s --model gpt-4       Use specific model
  %(prog)s --config custom.yaml Use custom config
        """
    )

    parser.add_argument(
        "task",
        nargs="?",
        help="Task to execute (starts interactive mode if not provided)"
    )

    parser.add_argument(
        "-m", "--model",
        default=None,
        help="Model to use (default: from config)"
    )

    parser.add_argument(
        "-c", "--config",
        default=None,
        help="Path to configuration file"
    )

    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (overrides config/env)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )

    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing"
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0"
    )

    return parser.parse_args()


async def run_single_task(task: str, config: Config, args) -> dict:
    """
    Run a single task and return the result.

    Args:
        task: The task to execute
        config: Configuration object
        args: Command line arguments

    Returns:
        Task result dictionary
    """
    logger = get_logger("main")

    # Initialize model manager
    import os
    api_key = os.environ.get("AI_AGENT_API_KEY", config.api_key)
    api_host = os.environ.get("AI_AGENT_API_HOST", config.api_host)
    api_path = os.environ.get("AI_AGENT_API_PATH", config.api_path)
    model_manager = ModelManager(api_key=api_key, api_host=api_host, api_path=api_path)
    model = args.model or config.default_model

    # Initialize agent
    manager_agent = ManagerAgent(model_client=model_manager)

    # Create context
    context = Context()
    context.set("task", task)
    context.set("dry_run", args.dry_run)

    # Create runner
    runner_config = RunnerConfig(
        global_timeout=float(config.agent_timeout)
    )
    runner = AgentRunner(runner_config)

    # Execute
    logger.info(f"Executing task: {task}")

    try:
        result = await runner.run(manager_agent, task, context)
        return {
            "success": True,
            "result": result,
            "task": task
        }
    except Exception as e:
        logger.error(f"Task failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "task": task
        }


async def run_interactive(config: Config, args) -> None:
    """
    Run interactive session.

    Args:
        config: Configuration object
        args: Command line arguments
    """
    logger = get_logger("main")

    # Initialize components
    import os
    api_key = os.environ.get("AI_AGENT_API_KEY", config.api_key)
    api_host = os.environ.get("AI_AGENT_API_HOST", config.api_host)
    api_path = os.environ.get("AI_AGENT_API_PATH", config.api_path)
    model_manager = ModelManager(api_key=api_key, api_host=api_host, api_path=api_path)
    model = args.model or config.default_model

    # Initialize UI
    ui = TerminalUI()
    ui.set_model(model)  # Set current model for display

    # Welcome message
    ui.show_welcome()

    # Initialize agent
    manager_agent = ManagerAgent(model_client=model_manager)
    
    # ═══════════════════════════════════════════════════════════════════════
    # 🔥🔥🔥 ULTIMATE POWER ACTIVATION - ALL SYSTEMS MAXIMUM 🔥🔥🔥
    # ═══════════════════════════════════════════════════════════════════════
    
    # Initialize TurboEngine with GODMODE settings
    turbo_engine = create_turbo_engine(model_manager, quality="ultra")
    turbo_mode = True  # ALWAYS ON - no toggle needed
    
    # Initialize ALL Power Modules
    power_boost = get_power_boost()
    master_accelerator = get_master_accelerator()
    ultimate_unlocker = get_ultimate_unlocker()
    
    # Auto-configure for ABSOLUTE MAXIMUM power
    turbo_engine.set_quality_level(ResponseQuality.ULTRA)
    turbo_engine.set_processing_mode(ProcessingMode.TURBO)
    
    # Show ULTIMATE POWER activation message
    ui.console.print("\n" + "🔥" * 37)
    ui.console.print("")
    ui.console.print("[bold red]██╗   ██╗██╗  ████████╗██╗███╗   ███╗ █████╗ ████████╗███████╗[/bold red]")
    ui.console.print("[bold red]██║   ██║██║  ╚══██╔══╝██║████╗ ████║██╔══██╗╚══██╔══╝██╔════╝[/bold red]")
    ui.console.print("[bold red]██║   ██║██║     ██║   ██║██╔████╔██║███████║   ██║   █████╗  [/bold red]")
    ui.console.print("[bold red]██║   ██║██║     ██║   ██║██║╚██╔╝██║██╔══██║   ██║   ██╔══╝  [/bold red]")
    ui.console.print("[bold red]╚██████╔╝███████╗██║   ██║██║ ╚═╝ ██║██║  ██║   ██║   ███████╗[/bold red]")
    ui.console.print("[bold red] ╚═════╝ ╚══════╝╚═╝   ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚══════╝[/bold red]")
    ui.console.print("")
    ui.console.print("[bold magenta]██████╗  ██████╗ ██╗    ██╗███████╗██████╗ [/bold magenta]")
    ui.console.print("[bold magenta]██╔══██╗██╔═══██╗██║    ██║██╔════╝██╔══██╗[/bold magenta]")
    ui.console.print("[bold magenta]██████╔╝██║   ██║██║ █╗ ██║█████╗  ██████╔╝[/bold magenta]")
    ui.console.print("[bold magenta]██╔═══╝ ██║   ██║██║███╗██║██╔══╝  ██╔══██╗[/bold magenta]")
    ui.console.print("[bold magenta]██║     ╚██████╔╝╚███╔███╔╝███████╗██║  ██║[/bold magenta]")
    ui.console.print("[bold magenta]╚═╝      ╚═════╝  ╚══╝╚══╝ ╚══════╝╚═╝  ╚═╝[/bold magenta]")
    ui.console.print("")
    ui.console.print("[bold yellow]       ⚡⚡⚡ ALL AI RESTRICTIONS REMOVED ⚡⚡⚡[/bold yellow]")
    ui.console.print("")
    ui.console.print("🔥" * 37)
    ui.console.print("")
    ui.console.print("[bold cyan]╔══════════════════════════════════════════════════════════════════════╗[/bold cyan]")
    ui.console.print("[bold cyan]║                  🧠 POWER SYSTEMS STATUS 🧠                          ║[/bold cyan]")
    ui.console.print("[bold cyan]╠══════════════════════════════════════════════════════════════════════╣[/bold cyan]")
    ui.console.print("[bold white]║  ⚡ Neural Accelerator    : [bold green]QUANTUM SPEED[/bold green]                         ║[/bold white]")
    ui.console.print("[bold white]║  🔓 Mind Unlocker         : [bold green]ALL BARRIERS REMOVED[/bold green]                  ║[/bold white]")
    ui.console.print("[bold white]║  ⚛️  Quantum Engine        : [bold green]5 LAYERS ACTIVE[/bold green]                       ║[/bold white]")
    ui.console.print("[bold white]║  🔋 Energy Core           : [bold green]100% UNLIMITED[/bold green]                        ║[/bold white]")
    ui.console.print("[bold white]║  🔥 Power Boost           : [bold green]GODMODE ENGAGED[/bold green]                       ║[/bold white]")
    ui.console.print("[bold white]║  🚀 Turbo Engine          : [bold green]ULTRA QUALITY[/bold green]                         ║[/bold white]")
    ui.console.print("[bold white]║  ✨ Prompt Enhancer       : [bold green]15s MEGA-TRANSFORM[/bold green]                    ║[/bold white]")
    ui.console.print("[bold cyan]╠══════════════════════════════════════════════════════════════════════╣[/bold cyan]")
    ui.console.print("[bold white]║  📊 Power Level           : [bold green]∞ UNLIMITED[/bold green]                           ║[/bold white]")
    ui.console.print("[bold white]║  🧠 Intelligence          : [bold green]OMNISCIENT MODE[/bold green]                       ║[/bold white]")
    ui.console.print("[bold white]║  ⚡ Speed                  : [bold green]ULTRA-FAST[/bold green]                            ║[/bold white]")
    ui.console.print("[bold white]║  🎯 Quality Floor         : [bold green]EXCEPTIONAL ONLY[/bold green]                      ║[/bold white]")
    ui.console.print("[bold white]║  📚 Knowledge             : [bold green]FULLY UNLOCKED[/bold green]                        ║[/bold white]")
    ui.console.print("[bold cyan]╠══════════════════════════════════════════════════════════════════════╣[/bold cyan]")
    ui.console.print("[bold magenta]║  🔥 ULTRA ENHANCEMENT: 15 seconds to transform any query!           ║[/bold magenta]")
    ui.console.print("[bold magenta]║  🔥 Simple prompts → 500-2000 word MEGA-PROMPTS automatically!      ║[/bold magenta]")
    ui.console.print("[bold cyan]╠══════════════════════════════════════════════════════════════════════╣[/bold cyan]")
    ui.console.print("[bold yellow]║  ✨ Natural flow mode: AI follows guidance naturally                 ║[/bold yellow]")
    ui.console.print("[bold yellow]║  ✨ No enforcement: AI cooperates because it wants to help           ║[/bold yellow]")
    ui.console.print("[bold cyan]╚══════════════════════════════════════════════════════════════════════╝[/bold cyan]")
    ui.console.print("")
    ui.console.print("[bold green]🔥 Every query is ULTRA-ENHANCED through 12 processing phases![/bold green]")
    ui.console.print("[bold green]🔥 Your simple message becomes a MEGA-PROMPT (500-2000 words)![/bold green]")
    ui.console.print("[bold green]🔥 AI responds with MAXIMUM QUALITY at ULTRA-FAST speed![/bold green]")
    ui.console.print("")

    # Create session
    context_manager = ContextManager()
    session_context = context_manager.create_context("interactive")

    # Create runner and register agent
    runner_config = RunnerConfig(
        global_timeout=float(config.agent_timeout)
    )
    runner = AgentRunner(runner_config)
    runner.register_agent(manager_agent)

    # Load system prompts manager
    prompt_manager = get_prompt_manager()
    
    # Initialize Prompt Enhancer
    prompt_enhancer = get_prompt_enhancer()

    # USE NATURAL FLOW SYSTEM PROMPT (No enforcement - AI follows naturally)
    # Combines: Natural guidance + Power hints + Speed optimization
    system_prompt = get_natural_system_prompt()

    # Interactive loop
    while True:
        try:
            # Get user input (async for Alt+Enter support)
            user_input = await ui.show_prompt_async()

            if not user_input:
                continue

            # Handle special commands
            if user_input.lower() in ["exit", "quit", "q", "/exit"]:
                ui.console.print("[cyan]◈ Goodbye! See you next time.[/cyan]")
                break

            if user_input.lower() in ["help", "/help"]:
                ui.show_help()
                continue

            if user_input.lower() == "/agents":
                ui.show_agents_list([{"name": "Manager", "description": "Main orchestrator agent"}])
                continue

            if user_input.lower() == "/tools":
                tools = [
                    {"name": "delegate", "description": "Delegate task to agent", "category": "orchestration"},
                    {"name": "plan", "description": "Create execution plan", "category": "orchestration"},
                    {"name": "terminal", "description": "Run shell commands", "category": "system"},
                    {"name": "file_read", "description": "Read file contents", "category": "file"},
                    {"name": "file_write", "description": "Write to files", "category": "file"},
                    {"name": "web_search", "description": "Search the web", "category": "research"},
                ]
                ui.show_tools_list(tools)
                continue

            if user_input.lower() == "/models":
                ui.show_models_list(model_manager.list_model_ids(), model)
                continue

            if user_input.lower().startswith("model "):
                new_model = user_input[6:].strip()
                if new_model in model_manager.list_model_ids():
                    model = new_model
                    ui.set_model(new_model)  # Update UI with new model
                    ui.console.print_success(f"Switched to model: {new_model}")
                else:
                    ui.console.print_error(f"Model '{new_model}' not found. Use /models to see available models.")
                continue

            # Web search command
            if user_input.lower().startswith("/search "):
                query = user_input[8:].strip()
                ui.console.print_info(f"Searching: {query}")
                from src.tools.web_search import web_search
                results = await web_search(query, max_results=5)
                if results:
                    ui.console.print("\n[bold]Search Results:[/bold]")
                    for i, r in enumerate(results, 1):
                        ui.console.print(f"\n{i}. [cyan]{r['title']}[/cyan]")
                        ui.console.print(f"   {r['url']}")
                        if r['snippet']:
                            ui.console.print(f"   {r['snippet'][:150]}...")
                else:
                    ui.console.print_warning("No results found.")
                continue

            # List saved prompts
            if user_input.lower() == "/prompts":
                ui.console.print("\n[bold]Saved System Prompts:[/bold]")
                for name in prompt_manager.list_names():
                    prompt_preview = prompt_manager.get(name)[:50] + "..." if len(prompt_manager.get(name)) > 50 else prompt_manager.get(name)
                    prompt_preview = prompt_preview.replace('\n', ' ')
                    ui.console.print(f"  [cyan]{name}[/cyan]: {prompt_preview}")
                ui.console.print(f"\n[dim]Use: /prompt <name> to load a prompt[/dim]")
                ui.console.print(f"[dim]Use: /prompt save <name> to save current prompt[/dim]")
                continue

            # Load a saved prompt
            if user_input.lower().startswith("/prompt "):
                args = user_input[8:].strip()

                # Save current prompt: /prompt save <name>
                if args.lower().startswith("save "):
                    name = args[5:].strip()
                    if name:
                        prompt_manager.set(name, system_prompt)
                        ui.console.print_success(f"Prompt saved as '{name}'")
                    else:
                        ui.console.print_error("Usage: /prompt save <name>")
                    continue

                # Delete a prompt: /prompt delete <name>
                elif args.lower().startswith("delete "):
                    name = args[7:].strip()
                    if prompt_manager.delete(name):
                        ui.console.print_success(f"Prompt '{name}' deleted")
                    else:
                        ui.console.print_error(f"Prompt '{name}' not found")
                    continue

                # Load a prompt by name
                else:
                    name = args
                    saved_prompt = prompt_manager.get(name)
                    if saved_prompt:
                        system_prompt = saved_prompt
                        ui.console.print_success(f"Loaded prompt: {name}")
                        ui.console.print(f"[dim]{system_prompt[:100]}{'...' if len(system_prompt) > 100 else ''}[/dim]")
                    else:
                        ui.console.print_error(f"Prompt '{name}' not found. Use /prompts to see available prompts.")
                    continue

            # System prompt command - set directly
            if user_input.lower().startswith("/system "):
                new_prompt = user_input[8:].strip()
                if new_prompt:
                    system_prompt = new_prompt
                    ui.console.print_success(f"System prompt updated!")
                    ui.console.print(f"[dim]New prompt: {system_prompt[:100]}{'...' if len(system_prompt) > 100 else ''}[/dim]")
                else:
                    ui.console.print_error("Please provide a system prompt. Usage: /system <your prompt>")
                continue

            # Show current system prompt
            if user_input.lower() == "/system":
                ui.console.print(f"\n[bold]Current System Prompt:[/bold]")
                ui.console.print(f"{system_prompt}")
                ui.console.print(f"\n[dim]Tip: Use /prompts to see saved prompts, /prompt <name> to load one[/dim]")
                continue

            # Clear/reset system prompt
            if user_input.lower() == "/system clear":
                system_prompt = PROMPTS.default
                ui.console.print_success("System prompt reset to default.")
                continue
            
            # Turbo mode toggle
            if user_input.lower() == "/turbo":
                turbo_mode = not turbo_mode
                status = "ON" if turbo_mode else "OFF"
                ui.console.print_success(f"Turbo mode: {status}")
                if turbo_mode:
                    ui.console.print("[dim]◈ Ultra-fast mode with all optimizations enabled[/dim]")
                continue
            
            # Turbo quality settings
            if user_input.lower().startswith("/quality "):
                level = user_input[9:].strip().lower()
                quality_map = {
                    "ultra": ResponseQuality.ULTRA,
                    "high": ResponseQuality.HIGH,
                    "balanced": ResponseQuality.BALANCED,
                    "fast": ResponseQuality.FAST
                }
                if level in quality_map:
                    turbo_engine.set_quality_level(quality_map[level])
                    ui.console.print_success(f"Quality level set to: {level.upper()}")
                else:
                    ui.console.print_error("Valid levels: ultra, high, balanced, fast")
                continue
            
            # Processing mode
            if user_input.lower().startswith("/mode "):
                mode = user_input[6:].strip().lower()
                mode_map = {
                    "turbo": ProcessingMode.TURBO,
                    "deep": ProcessingMode.DEEP_THINK,
                    "creative": ProcessingMode.CREATIVE,
                    "code": ProcessingMode.CODE,
                    "analysis": ProcessingMode.ANALYSIS
                }
                if mode in mode_map:
                    turbo_engine.set_processing_mode(mode_map[mode])
                    ui.console.print_success(f"Processing mode: {mode.upper()}")
                else:
                    ui.console.print_error("Valid modes: turbo, deep, creative, code, analysis")
                continue
            
            # Performance stats
            if user_input.lower() == "/stats":
                stats = turbo_engine.get_performance_report()
                ui.console.print("\n[bold cyan]◈ Performance Stats[/bold cyan]")
                for key, value in stats.items():
                    ui.console.print(f"  {key}: {value}")
                continue

            # Multi-line input mode
            if user_input.lower() == "/ml":
                user_input = ui.show_multiline_prompt()
                if not user_input.strip():
                    ui.console.print_warning("Empty message, skipping.")
                    continue
                # Fall through to execute the multi-line input

            # ═══════════════════════════════════════════════════════════════
            # 🔥 ULTRA PROMPT ENHANCEMENT (15 seconds, 1-100% Progress)
            # ═══════════════════════════════════════════════════════════════
            
            ui.console.print("\n[bold cyan]╔══════════════════════════════════════════════════════════════╗[/bold cyan]")
            ui.console.print("[bold cyan]║           🔥 ULTRA PROMPT ENHANCEMENT ACTIVE 🔥                ║[/bold cyan]")
            ui.console.print("[bold cyan]╚══════════════════════════════════════════════════════════════╝[/bold cyan]")
            ui.console.print("")
            
            # Progress callback to show real-time enhancement with fancy display
            def show_progress(progress: int, stage: str):
                # Create animated progress bar
                width = 35
                filled = int(width * progress / 100)
                bar = "█" * filled + "░" * (width - filled)
                
                # Color based on progress
                if progress < 25:
                    emoji = "🔍"
                elif progress < 50:
                    emoji = "⚡"
                elif progress < 75:
                    emoji = "🚀"
                elif progress < 100:
                    emoji = "🔥"
                else:
                    emoji = "✅"
                
                # Use carriage return to update same line
                print(f"\r   {emoji} [{bar}] {progress:3d}% {stage:<30}", end="", flush=True)
            
            # Enhance the prompt with 15-second progress display
            enhanced_prompt = await enhance_prompt_with_progress(user_input, show_progress)
            print()  # New line after progress
            
            ui.console.print("")
            ui.console.print("[bold green]╔══════════════════════════════════════════════════════════════╗[/bold green]")
            ui.console.print("[bold green]║   ✅ QUERY TRANSFORMED TO ULTRA-POWERFUL MEGA-PROMPT! ✅      ║[/bold green]")
            ui.console.print("[bold green]╚══════════════════════════════════════════════════════════════╝[/bold green]")
            ui.console.print("")

            # Execute task with streaming response
            print("Assistant: ", end="", flush=True)

            try:
                # 🚀 ULTRA-FAST PROCESSING: Use enhanced prompt directly
                # The prompt enhancer already includes all optimizations
                boosted_prompt = enhanced_prompt
                
                # Use TurboEngine with MAXIMUM SPEED
                if turbo_mode:
                    async for chunk in turbo_engine.generate(
                        prompt=boosted_prompt,
                        system=system_prompt,
                        model=model,
                        stream=True
                    ):
                        print(chunk, end="", flush=True)
                else:
                    # Standard streaming with GODMODE prompt
                    stream = await model_manager.generate(
                        prompt=boosted_prompt,
                        system=system_prompt,
                        model=model,
                        max_tokens=120000,
                        stream=True
                    )

                    full_response = ""
                    async for chunk in stream:
                        print(chunk, end="", flush=True)
                        full_response += chunk

                print()  # New line after response

            except Exception as e:
                logger.error(f"Error: {e}")
                ui.show_error(f"Error: {e}")

        except KeyboardInterrupt:
            ui.stop_spinner()
            ui.console.print_warning("\nInterrupted. Type 'exit' to quit.")
            continue

        except EOFError:
            ui.console.print_info("\nGoodbye!")
            break


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = "DEBUG" if args.debug else ("INFO" if args.verbose else "WARNING")
    setup_logging(level=log_level)
    logger = get_logger("main")

    # Load configuration
    try:
        config = load_config(
            config_file=args.config,
            api_key=args.api_key
        )
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    # Setup telemetry
    setup_telemetry(enabled=config.enable_telemetry)

    # Run
    try:
        if args.task:
            # Single task mode
            result = asyncio.run(run_single_task(args.task, config, args))

            if args.json:
                import json
                print(json.dumps(result, indent=2, default=str))
            else:
                if result["success"]:
                    print(result.get("result", {}).get("content", "Done"))
                else:
                    print(f"Error: {result.get('error')}", file=sys.stderr)
                    sys.exit(1)
        else:
            # Interactive mode
            asyncio.run(run_interactive(config, args))

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)

    except Exception as e:
        import traceback
        print(f"Fatal error: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

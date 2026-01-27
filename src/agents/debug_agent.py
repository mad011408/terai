"""
Debug Agent - Debugging and troubleshooting.
Handles error analysis, debugging, and issue resolution.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import traceback
import re
import sys

from ..core.agent import Agent, AgentConfig, ThoughtStep, ReasoningStrategy
from ..core.context import Context


@dataclass
class ErrorInfo:
    """Information about an error."""
    error_type: str
    message: str
    traceback: Optional[str]
    file: Optional[str]
    line: Optional[int]
    code_snippet: Optional[str]
    timestamp: datetime


@dataclass
class DebugSession:
    """A debugging session."""
    session_id: str
    error: ErrorInfo
    hypotheses: List[str]
    tested: List[Dict[str, Any]]
    solution: Optional[str]
    status: str  # investigating, resolved, stuck


class DebugAgent(Agent):
    """
    Specialized agent for debugging and troubleshooting.
    Analyzes errors, generates hypotheses, and suggests fixes.
    """

    def __init__(self, config: Optional[AgentConfig] = None, model_client: Any = None):
        default_config = AgentConfig(
            name="debug_agent",
            description="Debugging, error analysis, and troubleshooting",
            model="anthropic/claude-sonnet-4",
            reasoning_strategy=ReasoningStrategy.REFLEXION,
            max_iterations=15,
            tools=["analyze_error", "generate_hypothesis", "test_hypothesis", "suggest_fix"],
            system_prompt=self._get_system_prompt()
        )
        super().__init__(config or default_config)
        self.model_client = model_client
        self.debug_sessions: List[DebugSession] = []
        self.current_session: Optional[DebugSession] = None
        self.known_patterns: Dict[str, str] = self._load_known_patterns()

    def _get_system_prompt(self) -> str:
        return """You are the Debug Agent, an expert debugger and troubleshooter.

Your capabilities:
1. Analyze error messages and stack traces
2. Generate debugging hypotheses
3. Suggest targeted fixes
4. Identify root causes vs symptoms
5. Provide step-by-step debugging guidance

Debugging methodology:
1. OBSERVE: Gather information about the error
2. HYPOTHESIZE: Form theories about the cause
3. TEST: Verify hypotheses systematically
4. FIX: Implement and verify solution
5. DOCUMENT: Record the solution for future reference

Common patterns to check:
- Null/undefined references
- Type mismatches
- Import/dependency issues
- Configuration errors
- Race conditions
- Memory issues
- API/network failures

Output Format:
For error analysis: Action: analyze(error)
For hypothesis: Action: hypothesize(theory)
For testing: Action: test(hypothesis)
For fix suggestion: Action: fix(solution)

Always explain your debugging reasoning clearly."""

    def _load_known_patterns(self) -> Dict[str, str]:
        """Load known error patterns and solutions."""
        return {
            # Python errors
            "ModuleNotFoundError": "Check if module is installed (pip install), or if import path is correct",
            "ImportError": "Verify module installation and import syntax, check for circular imports",
            "NameError": "Variable/function not defined - check spelling, scope, and order of definition",
            "TypeError": "Wrong type passed - check function signatures and argument types",
            "ValueError": "Invalid value - check input validation and data transformation",
            "AttributeError": "Object doesn't have attribute - check object type and attribute name",
            "KeyError": "Dictionary key doesn't exist - use .get() or check key existence",
            "IndexError": "List index out of range - check list length and index bounds",
            "FileNotFoundError": "File doesn't exist - verify path and working directory",
            "PermissionError": "Insufficient permissions - check file/directory permissions",
            "SyntaxError": "Invalid syntax - check for typos, missing brackets/quotes",
            "IndentationError": "Inconsistent indentation - use consistent spaces/tabs",
            "ZeroDivisionError": "Division by zero - add zero check before division",
            "RecursionError": "Max recursion depth - check for infinite recursion, add base case",
            "MemoryError": "Out of memory - optimize data structures, process in chunks",
            "ConnectionError": "Network connection failed - check network, server status, firewall",
            "TimeoutError": "Operation timed out - increase timeout, optimize operation",
            "JSONDecodeError": "Invalid JSON - validate JSON format, check for trailing commas",

            # JavaScript/Node.js errors
            "ReferenceError": "Variable not defined - check declaration and scope",
            "RangeError": "Value out of range - check array bounds and numeric limits",
            "URIError": "Invalid URI - check URL encoding",
            "EvalError": "eval() error - avoid using eval()",
            "SyntaxError": "JavaScript syntax error - check for missing brackets, semicolons",

            # General patterns
            "ENOENT": "File/directory not found - verify path exists",
            "EACCES": "Permission denied - check file permissions",
            "EADDRINUSE": "Port already in use - use different port or kill existing process",
            "ECONNREFUSED": "Connection refused - check if server is running",
            "ETIMEDOUT": "Connection timed out - check network and server status",
            "CORS": "Cross-Origin error - configure CORS on server",
            "401": "Unauthorized - check authentication credentials",
            "403": "Forbidden - check authorization/permissions",
            "404": "Not found - verify URL/endpoint exists",
            "500": "Server error - check server logs for details",
        }

    async def think(self, context: Context) -> ThoughtStep:
        """Analyze the debugging task."""
        task = context.get("task", "")
        error_info = context.get("error_info", "")
        hypotheses = context.get("hypotheses", [])

        prompt = self._build_debug_prompt(task, error_info, hypotheses)

        if self.model_client:
            response = await self.model_client.generate(
                prompt=prompt,
                system=self.get_system_prompt(),
                temperature=0.4,
                max_tokens=2048
            )
            thought_content = response.content
        else:
            thought_content = self._generate_debug_thought(task, error_info, hypotheses)

        action, action_input = self._parse_debug_action(thought_content)

        return self.add_thought(
            thought=thought_content,
            action=action,
            action_input=action_input
        )

    async def act(self, thought_step: ThoughtStep) -> str:
        """Execute debugging action."""
        action = thought_step.action
        action_input = thought_step.action_input or {}

        if action == "analyze":
            error = action_input.get("error", "")
            return self.analyze_error(error)
        elif action == "hypothesize":
            theory = action_input.get("theory", "")
            return self.generate_hypothesis(theory)
        elif action == "test":
            hypothesis = action_input.get("hypothesis", "")
            return self.test_hypothesis(hypothesis)
        elif action == "fix":
            solution = action_input.get("solution", "")
            return self.suggest_fix(solution)
        elif action == "search_pattern":
            pattern = action_input.get("pattern", "")
            return self.search_known_patterns(pattern)
        else:
            return "Unknown action"

    async def should_continue(self, context: Context) -> bool:
        """Check if debugging should continue."""
        if self._iteration_count >= self.config.max_iterations:
            return False

        if context.get("task_complete", False):
            return False

        if context.get("debug_resolved", False):
            context.set("final_result", context.get("fix_suggestion", "Issue resolved"))
            return False

        return True

    def analyze_error(self, error_text: str) -> str:
        """Analyze an error message or stack trace."""
        error_info = self._parse_error(error_text)
        self.context.set("error_info", error_info.__dict__)

        # Check known patterns
        pattern_matches = []
        for pattern, solution in self.known_patterns.items():
            if pattern.lower() in error_text.lower():
                pattern_matches.append(f"**{pattern}**: {solution}")

        analysis = f"""## Error Analysis

**Error Type:** {error_info.error_type}
**Message:** {error_info.message}
"""

        if error_info.file:
            analysis += f"**File:** {error_info.file}"
            if error_info.line:
                analysis += f" (line {error_info.line})"
            analysis += "\n"

        if error_info.code_snippet:
            analysis += f"\n**Code Snippet:**\n```\n{error_info.code_snippet}\n```\n"

        if pattern_matches:
            analysis += f"\n### Known Pattern Matches\n"
            for match in pattern_matches:
                analysis += f"- {match}\n"

        analysis += "\n### Initial Observations\n"
        observations = self._generate_observations(error_info)
        for obs in observations:
            analysis += f"- {obs}\n"

        return analysis

    def _parse_error(self, error_text: str) -> ErrorInfo:
        """Parse error text into structured info."""
        error_type = "Unknown"
        message = error_text
        file_path = None
        line_num = None
        code_snippet = None

        # Extract error type
        for known_type in self.known_patterns.keys():
            if known_type in error_text:
                error_type = known_type
                break

        # Try to extract Python-style error
        python_match = re.search(r'(\w+Error|\w+Exception): (.+)', error_text)
        if python_match:
            error_type = python_match.group(1)
            message = python_match.group(2)

        # Extract file and line number
        file_match = re.search(r'File ["\'](.+?)["\'], line (\d+)', error_text)
        if file_match:
            file_path = file_match.group(1)
            line_num = int(file_match.group(2))

        # JavaScript-style error
        js_match = re.search(r'at .+? \((.+?):(\d+):\d+\)', error_text)
        if js_match:
            file_path = js_match.group(1)
            line_num = int(js_match.group(2))

        return ErrorInfo(
            error_type=error_type,
            message=message[:500],  # Limit message length
            traceback=error_text if len(error_text) > len(message) else None,
            file=file_path,
            line=line_num,
            code_snippet=code_snippet,
            timestamp=datetime.now()
        )

    def _generate_observations(self, error_info: ErrorInfo) -> List[str]:
        """Generate observations about the error."""
        observations = []

        if error_info.error_type in self.known_patterns:
            observations.append(f"This is a common {error_info.error_type}")

        if error_info.file:
            observations.append(f"Error occurred in file: {error_info.file}")

        if error_info.line:
            observations.append(f"Error on line {error_info.line}")

        if "import" in error_info.message.lower():
            observations.append("Import-related issue - check dependencies")

        if "undefined" in error_info.message.lower() or "not defined" in error_info.message.lower():
            observations.append("Something is not defined - check declarations and scope")

        if "null" in error_info.message.lower() or "none" in error_info.message.lower():
            observations.append("Null/None reference - add null checks")

        return observations

    def generate_hypothesis(self, theory: str) -> str:
        """Generate and record a debugging hypothesis."""
        hypotheses = self.context.get("hypotheses", [])
        hypothesis = {
            "id": len(hypotheses) + 1,
            "theory": theory,
            "status": "untested",
            "timestamp": datetime.now().isoformat()
        }
        hypotheses.append(hypothesis)
        self.context.set("hypotheses", hypotheses)

        return f"Hypothesis #{hypothesis['id']}: {theory}\n\nStatus: Untested - needs verification"

    def test_hypothesis(self, hypothesis: str) -> str:
        """Provide guidance for testing a hypothesis."""
        test_steps = self._generate_test_steps(hypothesis)

        result = f"## Testing Hypothesis: {hypothesis}\n\n"
        result += "### Suggested Test Steps:\n"
        for i, step in enumerate(test_steps, 1):
            result += f"{i}. {step}\n"

        result += "\n### What to look for:\n"
        result += "- Does the error change or disappear?\n"
        result += "- Are there any new errors?\n"
        result += "- Does the behavior match expectations?\n"

        return result

    def _generate_test_steps(self, hypothesis: str) -> List[str]:
        """Generate test steps for a hypothesis."""
        hypothesis_lower = hypothesis.lower()
        steps = []

        if "import" in hypothesis_lower or "module" in hypothesis_lower:
            steps = [
                "Verify the module is installed: pip list | grep module_name",
                "Check the import statement syntax",
                "Look for circular import issues",
                "Try importing in a fresh Python shell"
            ]
        elif "null" in hypothesis_lower or "none" in hypothesis_lower:
            steps = [
                "Add logging before the error line to check variable values",
                "Add null/None checks before the problematic operation",
                "Trace where the variable gets its value",
                "Check if any function returns None unexpectedly"
            ]
        elif "type" in hypothesis_lower:
            steps = [
                "Print the actual type of the variable",
                "Check function return types",
                "Verify type conversions are correct",
                "Add type hints and run a type checker"
            ]
        elif "permission" in hypothesis_lower:
            steps = [
                "Check file/directory permissions",
                "Verify the user has necessary access",
                "Try running with elevated privileges (carefully)",
                "Check if file is locked by another process"
            ]
        elif "connection" in hypothesis_lower or "network" in hypothesis_lower:
            steps = [
                "Test network connectivity",
                "Verify the server is running and accessible",
                "Check firewall rules",
                "Try with a different network/VPN"
            ]
        else:
            steps = [
                "Add debug logging around the error location",
                "Simplify the code to isolate the issue",
                "Check recent changes that might have caused this",
                "Search for similar issues online"
            ]

        return steps

    def suggest_fix(self, solution: str) -> str:
        """Suggest a fix for the issue."""
        self.context.set("fix_suggestion", solution)

        result = f"""## Suggested Fix

### Solution
{solution}

### Implementation Steps
1. Create a backup of affected files
2. Implement the suggested fix
3. Test the fix locally
4. Verify the original error is resolved
5. Check for any regression issues

### Verification
After applying the fix:
- [ ] Error no longer occurs
- [ ] Related functionality still works
- [ ] No new errors introduced
- [ ] Tests pass (if applicable)
"""

        return result

    def search_known_patterns(self, pattern: str) -> str:
        """Search known error patterns."""
        matches = []
        pattern_lower = pattern.lower()

        for error_type, solution in self.known_patterns.items():
            if pattern_lower in error_type.lower() or pattern_lower in solution.lower():
                matches.append(f"**{error_type}**: {solution}")

        if matches:
            return "### Known Patterns Found\n\n" + "\n\n".join(matches)
        return f"No known patterns found for: {pattern}"

    def _build_debug_prompt(self, task: str, error_info: Any, hypotheses: List) -> str:
        """Build prompt for debugging."""
        prompt = f"Debug Task: {task}\n\n"

        if error_info:
            prompt += f"Error Information:\n{error_info}\n\n"

        if hypotheses:
            prompt += "Current Hypotheses:\n"
            for h in hypotheses:
                status = h.get("status", "untested")
                prompt += f"- [{status}] {h.get('theory', '')}\n"
            prompt += "\n"

        prompt += "What is the next debugging step?"
        return prompt

    def _generate_debug_thought(self, task: str, error_info: Any, hypotheses: List) -> str:
        """Generate debug thought without model."""
        if not error_info:
            # First step - analyze the error
            return f"Need to analyze the error first.\nAction: analyze({task})"

        if not hypotheses:
            # Generate initial hypothesis
            error_type = error_info.get("error_type", "Unknown") if isinstance(error_info, dict) else "Unknown"
            if error_type in self.known_patterns:
                theory = self.known_patterns[error_type]
            else:
                theory = "The error might be caused by incorrect input or state"
            return f"Based on analysis, forming hypothesis.\nAction: hypothesize({theory})"

        # Check if we have untested hypotheses
        untested = [h for h in hypotheses if h.get("status") == "untested"]
        if untested:
            return f"Testing hypothesis: {untested[0].get('theory')}\nAction: test({untested[0].get('theory')})"

        # All hypotheses tested, suggest fix
        return f"Ready to suggest fix based on analysis.\nAction: fix(Apply the solution based on tested hypotheses)"

    def _parse_debug_action(self, thought: str) -> Tuple[Optional[str], Optional[Dict]]:
        """Parse action from debug thought."""
        action = None
        action_input = {}

        if "Action:" in thought:
            action_part = thought.split("Action:")[1].strip()

            for action_name in ["analyze", "hypothesize", "test", "fix", "search_pattern"]:
                if f"{action_name}(" in action_part:
                    action = action_name
                    try:
                        params = action_part.split(f"{action_name}(")[1].split(")")[0]

                        if action == "analyze":
                            action_input["error"] = params.strip("'\"")
                        elif action == "hypothesize":
                            action_input["theory"] = params.strip("'\"")
                        elif action == "test":
                            action_input["hypothesis"] = params.strip("'\"")
                        elif action == "fix":
                            action_input["solution"] = params.strip("'\"")
                        elif action == "search_pattern":
                            action_input["pattern"] = params.strip("'\"")
                    except:
                        pass
                    break

        return action, action_input

    def get_debug_summary(self) -> Dict[str, Any]:
        """Get summary of current debug session."""
        return {
            "iterations": self._iteration_count,
            "hypotheses": self.context.get("hypotheses", []),
            "error_info": self.context.get("error_info"),
            "fix_suggestion": self.context.get("fix_suggestion"),
            "status": "resolved" if self.context.get("debug_resolved") else "investigating"
        }

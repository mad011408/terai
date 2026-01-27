"""
Code Agent - Generates, analyzes, and modifies code.
Handles code generation, refactoring, and code analysis tasks.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import ast
import re

from ..core.agent import Agent, AgentConfig, ThoughtStep, ReasoningStrategy
from ..core.context import Context


@dataclass
class CodeBlock:
    """Represents a block of code."""
    language: str
    code: str
    filename: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None


@dataclass
class CodeAnalysis:
    """Result of code analysis."""
    language: str
    complexity: str  # low, medium, high
    issues: List[str]
    suggestions: List[str]
    functions: List[str]
    classes: List[str]
    imports: List[str]


class CodeAgent(Agent):
    """
    Specialized agent for code generation and analysis.
    Supports multiple programming languages and coding tasks.
    """

    def __init__(self, config: Optional[AgentConfig] = None, model_client: Any = None):
        default_config = AgentConfig(
            name="code_agent",
            description="Code generation, analysis, and modification",
            model="anthropic/claude-sonnet-4",
            reasoning_strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            max_iterations=10,
            tools=["generate_code", "analyze_code", "refactor", "explain_code"],
            system_prompt=self._get_system_prompt()
        )
        super().__init__(config or default_config)
        self.model_client = model_client
        self.supported_languages = [
            "python", "javascript", "typescript", "java", "c", "cpp",
            "go", "rust", "ruby", "php", "swift", "kotlin", "scala",
            "html", "css", "sql", "bash", "powershell"
        ]
        self.code_blocks: List[CodeBlock] = []

    def _get_system_prompt(self) -> str:
        return """You are the Code Agent, an expert programmer and code analyst.

Your capabilities:
1. Generate clean, efficient, well-documented code
2. Analyze code for bugs, performance issues, and best practices
3. Refactor code for better structure and maintainability
4. Explain complex code in simple terms
5. Convert code between programming languages

Coding principles:
- Write clean, readable code with meaningful names
- Follow language-specific conventions and best practices
- Include appropriate error handling
- Add comments for complex logic
- Consider edge cases and input validation
- Optimize for both readability and performance

Output Format:
For code generation: Action: generate(language, description)
For code analysis: Action: analyze(code)
For refactoring: Action: refactor(code, improvements)
For explanation: Action: explain(code)

Always provide code in markdown code blocks with language specification."""

    async def think(self, context: Context) -> ThoughtStep:
        """Analyze the coding task and plan implementation."""
        task = context.get("task", "")
        existing_code = context.get("existing_code", "")
        language = context.get("language", "python")

        prompt = self._build_code_prompt(task, existing_code, language)

        if self.model_client:
            response = await self.model_client.generate(
                prompt=prompt,
                system=self.get_system_prompt(),
                temperature=0.5,
                max_tokens=4096
            )
            thought_content = response.content
        else:
            thought_content = self._generate_code_thought(task, language)

        action, action_input = self._parse_code_action(thought_content)

        return self.add_thought(
            thought=thought_content,
            action=action,
            action_input=action_input
        )

    async def act(self, thought_step: ThoughtStep) -> str:
        """Execute the code action."""
        action = thought_step.action
        action_input = thought_step.action_input or {}

        if action == "generate":
            return await self.generate_code(
                language=action_input.get("language", "python"),
                description=action_input.get("description", ""),
                template=action_input.get("template")
            )
        elif action == "analyze":
            code = action_input.get("code", "")
            return self.analyze_code(code)
        elif action == "refactor":
            code = action_input.get("code", "")
            improvements = action_input.get("improvements", [])
            return await self.refactor_code(code, improvements)
        elif action == "explain":
            code = action_input.get("code", "")
            return self.explain_code(code)
        else:
            return "Unknown action"

    async def should_continue(self, context: Context) -> bool:
        """Check if code generation should continue."""
        if self._iteration_count >= self.config.max_iterations:
            return False

        if context.get("task_complete", False):
            return False

        # If code was generated, consider task complete
        if self.code_blocks:
            context.set("final_result", self._format_code_output())
            return False

        return True

    async def generate_code(self, language: str, description: str,
                            template: Optional[str] = None) -> str:
        """Generate code based on description."""
        if self.model_client:
            prompt = f"""Generate {language} code for: {description}

Requirements:
- Clean, production-ready code
- Proper error handling
- Clear comments where needed
- Follow {language} best practices

{f'Use this template as base: {template}' if template else ''}

Provide only the code in a code block."""

            response = await self.model_client.generate(
                prompt=prompt,
                system=self.get_system_prompt(),
                temperature=0.5,
                max_tokens=4096
            )

            code = self._extract_code_from_response(response.content, language)
        else:
            code = self._generate_template_code(language, description)

        # Store the generated code
        code_block = CodeBlock(language=language, code=code)
        self.code_blocks.append(code_block)
        self.context.set("generated_code", code)

        return f"```{language}\n{code}\n```"

    def _generate_template_code(self, language: str, description: str) -> str:
        """Generate template code without model."""
        templates = {
            "python": '''"""
{description}
"""

def main():
    """Main function."""
    # TODO: Implement {description}
    pass


if __name__ == "__main__":
    main()
''',
            "javascript": '''/**
 * {description}
 */

function main() {{
    // TODO: Implement {description}
}}

module.exports = {{ main }};
''',
            "typescript": '''/**
 * {description}
 */

export function main(): void {{
    // TODO: Implement {description}
}}
''',
            "java": '''/**
 * {description}
 */
public class Main {{
    public static void main(String[] args) {{
        // TODO: Implement {description}
    }}
}}
''',
            "go": '''package main

// {description}

func main() {{
    // TODO: Implement {description}
}}
''',
        }

        template = templates.get(language, templates["python"])
        return template.format(description=description)

    def analyze_code(self, code: str, language: str = "python") -> str:
        """Analyze code for issues and suggestions."""
        analysis = CodeAnalysis(
            language=language,
            complexity="medium",
            issues=[],
            suggestions=[],
            functions=[],
            classes=[],
            imports=[]
        )

        if language == "python":
            analysis = self._analyze_python(code)
        else:
            analysis = self._generic_analysis(code, language)

        # Format analysis result
        result = f"""## Code Analysis

**Language:** {analysis.language}
**Complexity:** {analysis.complexity}

### Structure
- **Functions:** {', '.join(analysis.functions) if analysis.functions else 'None found'}
- **Classes:** {', '.join(analysis.classes) if analysis.classes else 'None found'}
- **Imports:** {', '.join(analysis.imports) if analysis.imports else 'None found'}

### Issues
{self._format_list(analysis.issues) if analysis.issues else '- No critical issues found'}

### Suggestions
{self._format_list(analysis.suggestions) if analysis.suggestions else '- Code looks good!'}
"""
        return result

    def _analyze_python(self, code: str) -> CodeAnalysis:
        """Analyze Python code using AST."""
        analysis = CodeAnalysis(
            language="python",
            complexity="low",
            issues=[],
            suggestions=[],
            functions=[],
            classes=[],
            imports=[]
        )

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    analysis.functions.append(node.name)
                    # Check function complexity
                    if len(node.body) > 20:
                        analysis.suggestions.append(
                            f"Function '{node.name}' is long ({len(node.body)} statements). Consider breaking it down."
                        )
                elif isinstance(node, ast.AsyncFunctionDef):
                    analysis.functions.append(f"{node.name} (async)")
                elif isinstance(node, ast.ClassDef):
                    analysis.classes.append(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis.imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    analysis.imports.append(f"{node.module}")

            # Determine complexity
            total_items = len(analysis.functions) + len(analysis.classes)
            if total_items > 10:
                analysis.complexity = "high"
            elif total_items > 5:
                analysis.complexity = "medium"

            # Check for common issues
            if "import *" in code:
                analysis.issues.append("Avoid 'import *' - it pollutes namespace")

            if not any(line.strip().startswith('"""') or line.strip().startswith("'''")
                      for line in code.split('\n')[:5]):
                analysis.suggestions.append("Consider adding a module docstring")

        except SyntaxError as e:
            analysis.issues.append(f"Syntax error: {str(e)}")

        return analysis

    def _generic_analysis(self, code: str, language: str) -> CodeAnalysis:
        """Generic code analysis for any language."""
        analysis = CodeAnalysis(
            language=language,
            complexity="medium",
            issues=[],
            suggestions=[],
            functions=[],
            classes=[],
            imports=[]
        )

        lines = code.split('\n')
        analysis.complexity = "low" if len(lines) < 50 else "medium" if len(lines) < 200 else "high"

        # Simple pattern matching for common constructs
        function_patterns = {
            "javascript": r"function\s+(\w+)",
            "typescript": r"function\s+(\w+)|(\w+)\s*:\s*\([^)]*\)\s*=>",
            "java": r"(public|private|protected)?\s*(static)?\s*\w+\s+(\w+)\s*\(",
            "go": r"func\s+(\w+)",
            "rust": r"fn\s+(\w+)",
        }

        pattern = function_patterns.get(language)
        if pattern:
            matches = re.findall(pattern, code)
            analysis.functions = [m if isinstance(m, str) else m[0] for m in matches if m]

        # Check for long lines
        long_lines = [i+1 for i, line in enumerate(lines) if len(line) > 120]
        if long_lines:
            analysis.suggestions.append(f"Long lines found at: {long_lines[:5]}...")

        # Check for TODO/FIXME
        todos = [i+1 for i, line in enumerate(lines) if "TODO" in line or "FIXME" in line]
        if todos:
            analysis.suggestions.append(f"TODO/FIXME comments at lines: {todos}")

        return analysis

    async def refactor_code(self, code: str, improvements: List[str]) -> str:
        """Refactor code based on specified improvements."""
        if self.model_client:
            prompt = f"""Refactor the following code with these improvements:
{chr(10).join(f'- {imp}' for imp in improvements)}

Original code:
```
{code}
```

Provide the refactored code with explanations for changes."""

            response = await self.model_client.generate(
                prompt=prompt,
                system=self.get_system_prompt(),
                temperature=0.3,
                max_tokens=4096
            )
            return response.content
        else:
            return f"Refactoring suggestions:\n{chr(10).join(f'- {imp}' for imp in improvements)}\n\nOriginal code preserved."

    def explain_code(self, code: str, language: str = "python") -> str:
        """Generate explanation for code."""
        # Basic explanation without model
        explanation = f"""## Code Explanation

**Language:** {language}
**Lines:** {len(code.split(chr(10)))}

### Overview
This code appears to be a {language} program.

### Key Components:
"""
        # Try to identify main components
        if "def " in code or "function" in code:
            explanation += "- Contains function definitions\n"
        if "class " in code:
            explanation += "- Contains class definitions\n"
        if "import " in code or "require" in code:
            explanation += "- Uses external modules/libraries\n"
        if "async" in code or "await" in code:
            explanation += "- Uses asynchronous programming\n"
        if "try" in code or "catch" in code:
            explanation += "- Includes error handling\n"

        return explanation

    def _build_code_prompt(self, task: str, existing_code: str, language: str) -> str:
        """Build prompt for code tasks."""
        prompt = f"Task: {task}\n"
        prompt += f"Language: {language}\n\n"

        if existing_code:
            prompt += f"Existing code:\n```{language}\n{existing_code}\n```\n\n"

        prompt += "What code should be generated or modified?"
        return prompt

    def _generate_code_thought(self, task: str, language: str) -> str:
        """Generate code thought without model."""
        task_lower = task.lower()

        if "function" in task_lower or "def" in task_lower:
            return f"Need to generate a function in {language}.\nAction: generate({language}, {task})"
        elif "class" in task_lower:
            return f"Need to generate a class in {language}.\nAction: generate({language}, {task})"
        elif "analyze" in task_lower or "review" in task_lower:
            return f"Need to analyze the code.\nAction: analyze(code)"
        elif "refactor" in task_lower or "improve" in task_lower:
            return f"Need to refactor the code.\nAction: refactor(code, [improvements])"
        elif "explain" in task_lower:
            return f"Need to explain the code.\nAction: explain(code)"
        else:
            return f"Generating code for: {task}\nAction: generate({language}, {task})"

    def _parse_code_action(self, thought: str) -> Tuple[Optional[str], Optional[Dict]]:
        """Parse action from code thought."""
        action = None
        action_input = {}

        # Also check if code was directly generated
        if "```" in thought:
            code = self._extract_code_from_response(thought, "python")
            if code:
                action = "generate"
                action_input = {"code": code, "language": "python"}
                return action, action_input

        if "Action:" in thought:
            action_part = thought.split("Action:")[1].strip()

            if "generate(" in action_part:
                action = "generate"
                try:
                    params = action_part.split("generate(")[1].split(")")[0]
                    parts = params.split(",", 1)
                    action_input["language"] = parts[0].strip().strip("'\"")
                    if len(parts) > 1:
                        action_input["description"] = parts[1].strip().strip("'\"")
                except:
                    action_input["language"] = "python"
                    action_input["description"] = "general code"

            elif "analyze(" in action_part:
                action = "analyze"
                action_input["code"] = self.context.get("existing_code", "")

            elif "refactor(" in action_part:
                action = "refactor"
                action_input["code"] = self.context.get("existing_code", "")
                action_input["improvements"] = []

            elif "explain(" in action_part:
                action = "explain"
                action_input["code"] = self.context.get("existing_code", "")

        return action, action_input

    def _extract_code_from_response(self, response: str, default_language: str) -> str:
        """Extract code from markdown code blocks."""
        pattern = r"```(\w+)?\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            return matches[0][1].strip()

        # If no code block, return the response as-is if it looks like code
        if any(kw in response for kw in ["def ", "function ", "class ", "import "]):
            return response.strip()

        return ""

    def _format_code_output(self) -> str:
        """Format all generated code blocks for output."""
        output_parts = []
        for block in self.code_blocks:
            output_parts.append(f"```{block.language}\n{block.code}\n```")
        return "\n\n".join(output_parts)

    def _format_list(self, items: List[str]) -> str:
        """Format a list as markdown."""
        return "\n".join(f"- {item}" for item in items)

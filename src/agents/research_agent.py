"""
Research Agent - Web search and information gathering.
Handles web searches, information synthesis, and knowledge retrieval.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import asyncio
import re

from ..core.agent import Agent, AgentConfig, ThoughtStep, ReasoningStrategy
from ..core.context import Context


@dataclass
class SearchResult:
    """Represents a search result."""
    title: str
    url: str
    snippet: str
    source: str
    relevance_score: float = 0.0
    timestamp: Optional[datetime] = None


@dataclass
class ResearchReport:
    """A compiled research report."""
    query: str
    summary: str
    key_findings: List[str]
    sources: List[SearchResult]
    confidence: float
    timestamp: datetime


class ResearchAgent(Agent):
    """
    Specialized agent for web research and information gathering.
    Searches multiple sources and synthesizes findings.
    """

    def __init__(self, config: Optional[AgentConfig] = None, model_client: Any = None):
        default_config = AgentConfig(
            name="research_agent",
            description="Web search, research, and information synthesis",
            model="anthropic/claude-sonnet-4",
            reasoning_strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            max_iterations=8,
            tools=["web_search", "fetch_page", "summarize", "extract_facts"],
            system_prompt=self._get_system_prompt()
        )
        super().__init__(config or default_config)
        self.model_client = model_client
        self.search_results: List[SearchResult] = []
        self.research_history: List[ResearchReport] = []
        self.search_tool = None  # Will be set externally

    def _get_system_prompt(self) -> str:
        return """You are the Research Agent, specialized in finding and synthesizing information.

Your capabilities:
1. Search the web for relevant information
2. Extract key facts from search results
3. Synthesize information from multiple sources
4. Verify information accuracy when possible
5. Cite sources appropriately

Research principles:
- Use multiple sources for verification
- Prefer authoritative and recent sources
- Be transparent about uncertainty
- Distinguish facts from opinions
- Provide proper citations

Output Format:
For web search: Action: search(query)
For page fetch: Action: fetch(url)
For summarization: Action: summarize(content)
For fact extraction: Action: extract(content)

Always cite your sources when presenting findings."""

    async def think(self, context: Context) -> ThoughtStep:
        """Plan research strategy."""
        task = context.get("task", "")
        previous_results = context.get("search_results", [])

        prompt = self._build_research_prompt(task, previous_results)

        if self.model_client:
            response = await self.model_client.generate(
                prompt=prompt,
                system=self.get_system_prompt(),
                temperature=0.5,
                max_tokens=2048
            )
            thought_content = response.content
        else:
            thought_content = self._generate_research_thought(task, previous_results)

        action, action_input = self._parse_research_action(thought_content)

        return self.add_thought(
            thought=thought_content,
            action=action,
            action_input=action_input
        )

    async def act(self, thought_step: ThoughtStep) -> str:
        """Execute research action."""
        action = thought_step.action
        action_input = thought_step.action_input or {}

        if action == "search":
            query = action_input.get("query", "")
            return await self.web_search(query)
        elif action == "fetch":
            url = action_input.get("url", "")
            return await self.fetch_page(url)
        elif action == "summarize":
            content = action_input.get("content", "")
            return await self.summarize(content)
        elif action == "extract":
            content = action_input.get("content", "")
            return self.extract_facts(content)
        elif action == "compile":
            return self.compile_report()
        else:
            return "Unknown action"

    async def should_continue(self, context: Context) -> bool:
        """Check if research should continue."""
        if self._iteration_count >= self.config.max_iterations:
            # Compile final report
            context.set("final_result", self.compile_report())
            return False

        if context.get("task_complete", False):
            return False

        # Check if we have enough information
        if len(self.search_results) >= 5:
            context.set("final_result", self.compile_report())
            return False

        return True

    async def web_search(self, query: str, num_results: int = 5) -> str:
        """Perform web search."""
        if self.search_tool:
            # Use actual search tool if available
            results = await self.search_tool.search(query, num_results)
            self.search_results.extend(results)
            return self._format_search_results(results)
        else:
            # Simulate search results
            simulated_results = self._simulate_search(query)
            self.search_results.extend(simulated_results)
            self.context.set("search_results", self.search_results)
            return self._format_search_results(simulated_results)

    def _simulate_search(self, query: str) -> List[SearchResult]:
        """Simulate search results for testing."""
        return [
            SearchResult(
                title=f"Search result 1 for: {query}",
                url=f"https://example.com/result1?q={query.replace(' ', '+')}",
                snippet=f"This is a simulated search result about {query}. It contains relevant information that would help answer the query.",
                source="example.com",
                relevance_score=0.95,
                timestamp=datetime.now()
            ),
            SearchResult(
                title=f"Search result 2 for: {query}",
                url=f"https://example.org/article?topic={query.replace(' ', '-')}",
                snippet=f"Another relevant result discussing {query} with detailed analysis and examples.",
                source="example.org",
                relevance_score=0.85,
                timestamp=datetime.now()
            ),
            SearchResult(
                title=f"Documentation about {query}",
                url=f"https://docs.example.com/{query.replace(' ', '_')}",
                snippet=f"Official documentation and guides related to {query}.",
                source="docs.example.com",
                relevance_score=0.80,
                timestamp=datetime.now()
            ),
        ]

    async def fetch_page(self, url: str) -> str:
        """Fetch and extract content from a web page."""
        # In real implementation, would use httpx or similar
        return f"[Content fetched from {url}]\n\nThis would contain the actual page content."

    async def summarize(self, content: str, max_length: int = 500) -> str:
        """Summarize content."""
        if self.model_client:
            response = await self.model_client.generate(
                prompt=f"Summarize the following in {max_length} characters:\n\n{content}",
                temperature=0.3,
                max_tokens=max_length // 4
            )
            return response.content
        else:
            # Simple truncation
            if len(content) > max_length:
                return content[:max_length] + "..."
            return content

    def extract_facts(self, content: str) -> str:
        """Extract key facts from content."""
        # Simple fact extraction
        sentences = content.split('.')
        facts = []

        for sentence in sentences[:10]:  # Limit to first 10 sentences
            sentence = sentence.strip()
            if len(sentence) > 20:  # Meaningful sentences
                # Look for fact indicators
                if any(indicator in sentence.lower() for indicator in
                       ["is", "are", "was", "were", "found", "showed", "indicates"]):
                    facts.append(f"• {sentence}")

        return "\n".join(facts) if facts else "No clear facts extracted."

    def compile_report(self) -> str:
        """Compile research findings into a report."""
        if not self.search_results:
            return "No research results to compile."

        task = self.context.get("task", "Unknown query")

        report = f"""## Research Report

### Query
{task}

### Summary
Based on {len(self.search_results)} sources, here are the key findings:

### Key Findings
"""
        # Add findings from search results
        for i, result in enumerate(self.search_results[:5], 1):
            report += f"\n{i}. **{result.title}**\n"
            report += f"   - {result.snippet}\n"
            report += f"   - Source: [{result.source}]({result.url})\n"

        report += f"""
### Sources
"""
        for result in self.search_results[:5]:
            report += f"- [{result.title}]({result.url})\n"

        report += f"""
### Confidence
Moderate - Based on {len(self.search_results)} search results.

---
*Research completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        # Store report
        research_report = ResearchReport(
            query=task,
            summary="Research completed",
            key_findings=[r.snippet for r in self.search_results[:5]],
            sources=self.search_results[:5],
            confidence=0.7,
            timestamp=datetime.now()
        )
        self.research_history.append(research_report)

        return report

    def _build_research_prompt(self, task: str, previous_results: List) -> str:
        """Build prompt for research planning."""
        prompt = f"Research Task: {task}\n\n"

        if previous_results:
            prompt += "Previous search results:\n"
            for r in previous_results[-3:]:
                if isinstance(r, SearchResult):
                    prompt += f"- {r.title}: {r.snippet[:100]}...\n"
            prompt += "\n"

        prompt += "What is the next research action?"
        return prompt

    def _generate_research_thought(self, task: str, previous_results: List) -> str:
        """Generate research thought without model."""
        if not previous_results:
            # First iteration - do web search
            query = self._generate_search_query(task)
            return f"Starting research on: {task}\n\nNeed to search for relevant information.\nAction: search({query})"
        elif len(previous_results) < 3:
            # Need more results
            query = self._generate_followup_query(task, previous_results)
            return f"Found {len(previous_results)} results. Searching for more.\nAction: search({query})"
        else:
            # Have enough results, compile report
            return f"Collected {len(previous_results)} results. Ready to compile report.\nAction: compile()"

    def _generate_search_query(self, task: str) -> str:
        """Generate search query from task."""
        # Remove common words and create search query
        stop_words = ["find", "search", "look", "for", "about", "the", "a", "an", "what", "is", "are"]
        words = task.lower().split()
        query_words = [w for w in words if w not in stop_words]
        return " ".join(query_words) if query_words else task

    def _generate_followup_query(self, task: str, previous_results: List) -> str:
        """Generate follow-up search query."""
        base_query = self._generate_search_query(task)
        # Add variation
        return f"{base_query} detailed"

    def _format_search_results(self, results: List[SearchResult]) -> str:
        """Format search results for display."""
        if not results:
            return "No results found."

        formatted = f"Found {len(results)} results:\n\n"
        for i, result in enumerate(results, 1):
            formatted += f"{i}. **{result.title}**\n"
            formatted += f"   URL: {result.url}\n"
            formatted += f"   {result.snippet}\n\n"

        return formatted

    def _parse_research_action(self, thought: str) -> Tuple[Optional[str], Optional[Dict]]:
        """Parse action from research thought."""
        action = None
        action_input = {}

        if "Action:" in thought:
            action_part = thought.split("Action:")[1].strip()

            if "search(" in action_part:
                action = "search"
                try:
                    query = action_part.split("search(")[1].split(")")[0]
                    action_input["query"] = query.strip("'\"")
                except:
                    pass

            elif "fetch(" in action_part:
                action = "fetch"
                try:
                    url = action_part.split("fetch(")[1].split(")")[0]
                    action_input["url"] = url.strip("'\"")
                except:
                    pass

            elif "summarize(" in action_part:
                action = "summarize"
                action_input["content"] = self.context.get("last_content", "")

            elif "extract(" in action_part:
                action = "extract"
                action_input["content"] = self.context.get("last_content", "")

            elif "compile" in action_part:
                action = "compile"

        return action, action_input

    def get_research_history(self) -> List[Dict]:
        """Get research history."""
        return [
            {
                "query": r.query,
                "summary": r.summary,
                "sources_count": len(r.sources),
                "confidence": r.confidence,
                "timestamp": r.timestamp.isoformat()
            }
            for r in self.research_history
        ]

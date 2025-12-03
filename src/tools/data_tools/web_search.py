"""
Web search tool for information retrieval.
Supports multiple search providers.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import asyncio
import aiohttp
from datetime import datetime
import json

from ..base_tool import BaseTool, ToolConfig, ToolParameter, ToolCategory


@dataclass
class SearchResult:
    """A single search result."""
    title: str
    url: str
    snippet: str
    source: str
    timestamp: Optional[datetime] = None
    relevance_score: float = 0.0


class WebSearchTool(BaseTool):
    """
    Web search tool supporting multiple search providers.
    """

    def __init__(self, api_key: Optional[str] = None, provider: str = "google"):
        config = ToolConfig(
            name="web_search",
            description="Search the web for information. Returns relevant search results.",
            category=ToolCategory.DATA,
            timeout=30.0,
            retry_attempts=2
        )
        super().__init__(config)
        self.api_key = api_key
        self.provider = provider
        self.session: Optional[aiohttp.ClientSession] = None

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                param_type="string",
                description="The search query",
                required=True
            ),
            ToolParameter(
                name="num_results",
                param_type="number",
                description="Number of results to return",
                required=False,
                default=5,
                min_value=1,
                max_value=20
            ),
            ToolParameter(
                name="search_type",
                param_type="string",
                description="Type of search",
                required=False,
                default="web",
                enum_values=["web", "news", "images"]
            ),
            ToolParameter(
                name="time_range",
                param_type="string",
                description="Time range for results",
                required=False,
                default="any",
                enum_values=["any", "day", "week", "month", "year"]
            )
        ]

    async def _execute(self, query: str, num_results: int = 5,
                      search_type: str = "web", time_range: str = "any") -> List[Dict]:
        """Execute web search."""
        if self.provider == "google" and self.api_key:
            return await self._google_search(query, num_results, time_range)
        elif self.provider == "bing" and self.api_key:
            return await self._bing_search(query, num_results, time_range)
        else:
            # Fallback to simulated results
            return self._simulate_search(query, num_results)

    async def _google_search(self, query: str, num_results: int,
                            time_range: str) -> List[Dict]:
        """Search using Google Custom Search API."""
        if not self.session:
            self.session = aiohttp.ClientSession()

        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.api_key,
            "cx": "your_search_engine_id",  # Configure this
            "q": query,
            "num": num_results
        }

        if time_range != "any":
            date_restrict = {"day": "d1", "week": "w1", "month": "m1", "year": "y1"}
            params["dateRestrict"] = date_restrict.get(time_range, "")

        try:
            async with self.session.get(url, params=params) as response:
                data = await response.json()

                results = []
                for item in data.get("items", []):
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                        "source": "google"
                    })
                return results
        except Exception as e:
            raise Exception(f"Google search failed: {str(e)}")

    async def _bing_search(self, query: str, num_results: int,
                          time_range: str) -> List[Dict]:
        """Search using Bing Search API."""
        if not self.session:
            self.session = aiohttp.ClientSession()

        url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        params = {
            "q": query,
            "count": num_results
        }

        if time_range != "any":
            freshness = {"day": "Day", "week": "Week", "month": "Month"}
            params["freshness"] = freshness.get(time_range, "")

        try:
            async with self.session.get(url, headers=headers, params=params) as response:
                data = await response.json()

                results = []
                for item in data.get("webPages", {}).get("value", []):
                    results.append({
                        "title": item.get("name", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("snippet", ""),
                        "source": "bing"
                    })
                return results
        except Exception as e:
            raise Exception(f"Bing search failed: {str(e)}")

    def _simulate_search(self, query: str, num_results: int) -> List[Dict]:
        """Simulate search results for testing."""
        results = []
        for i in range(num_results):
            results.append({
                "title": f"Result {i+1}: {query}",
                "url": f"https://example.com/result{i+1}?q={query.replace(' ', '+')}",
                "snippet": f"This is a simulated search result about {query}. "
                          f"It contains relevant information for the query.",
                "source": "simulated",
                "relevance_score": 1.0 - (i * 0.1)
            })
        return results

    async def close(self):
        """Close the HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None


class NewsSearchTool(BaseTool):
    """
    Tool for searching news articles.
    """

    def __init__(self, api_key: Optional[str] = None):
        config = ToolConfig(
            name="news_search",
            description="Search for recent news articles on a topic.",
            category=ToolCategory.DATA,
            timeout=30.0
        )
        super().__init__(config)
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                param_type="string",
                description="The news search query",
                required=True
            ),
            ToolParameter(
                name="num_results",
                param_type="number",
                description="Number of articles to return",
                required=False,
                default=5,
                min_value=1,
                max_value=20
            ),
            ToolParameter(
                name="language",
                param_type="string",
                description="Language code for articles",
                required=False,
                default="en",
                enum_values=["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]
            ),
            ToolParameter(
                name="sort_by",
                param_type="string",
                description="How to sort results",
                required=False,
                default="relevancy",
                enum_values=["relevancy", "popularity", "publishedAt"]
            )
        ]

    async def _execute(self, query: str, num_results: int = 5,
                      language: str = "en", sort_by: str = "relevancy") -> List[Dict]:
        """Execute news search."""
        if self.api_key:
            return await self._newsapi_search(query, num_results, language, sort_by)
        else:
            return self._simulate_news(query, num_results)

    async def _newsapi_search(self, query: str, num_results: int,
                             language: str, sort_by: str) -> List[Dict]:
        """Search using NewsAPI."""
        if not self.session:
            self.session = aiohttp.ClientSession()

        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "pageSize": num_results,
            "language": language,
            "sortBy": sort_by,
            "apiKey": self.api_key
        }

        try:
            async with self.session.get(url, params=params) as response:
                data = await response.json()

                results = []
                for article in data.get("articles", []):
                    results.append({
                        "title": article.get("title", ""),
                        "url": article.get("url", ""),
                        "snippet": article.get("description", ""),
                        "source": article.get("source", {}).get("name", ""),
                        "published_at": article.get("publishedAt", ""),
                        "author": article.get("author", "")
                    })
                return results
        except Exception as e:
            raise Exception(f"News search failed: {str(e)}")

    def _simulate_news(self, query: str, num_results: int) -> List[Dict]:
        """Simulate news results for testing."""
        results = []
        for i in range(num_results):
            results.append({
                "title": f"Breaking: News about {query} - Story {i+1}",
                "url": f"https://news.example.com/article{i+1}",
                "snippet": f"Latest developments regarding {query}. "
                          f"Experts weigh in on the implications.",
                "source": f"News Source {i+1}",
                "published_at": datetime.now().isoformat(),
                "author": f"Reporter {i+1}"
            })
        return results

    async def close(self):
        """Close the HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None

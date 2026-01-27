"""
Real Web Search Tool using DuckDuckGo
"""

from typing import List, Dict, Any, Optional
import httpx
import asyncio
import re
from urllib.parse import quote_plus


class WebSearchTool:
    """Real web search using DuckDuckGo."""

    def __init__(self):
        self.client: Optional[httpx.AsyncClient] = None
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    async def _get_client(self) -> httpx.AsyncClient:
        if not self.client:
            self.client = httpx.AsyncClient(timeout=30.0, headers=self.headers, follow_redirects=True)
        return self.client

    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Search the web using DuckDuckGo.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of search results with title, url, and snippet
        """
        client = await self._get_client()
        results = []

        try:
            # DuckDuckGo HTML search
            url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            response = await client.get(url)

            if response.status_code == 200:
                html = response.text
                results = self._parse_ddg_results(html, max_results)
        except Exception as e:
            print(f"Search error: {e}")

        return results

    def _parse_ddg_results(self, html: str, max_results: int) -> List[Dict[str, str]]:
        """Parse DuckDuckGo HTML results."""
        results = []

        # Find result blocks
        result_pattern = r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>([^<]+)</a>'
        snippet_pattern = r'<a class="result__snippet"[^>]*>([^<]+(?:<[^>]+>[^<]*</[^>]+>)*[^<]*)</a>'

        links = re.findall(result_pattern, html)
        snippets = re.findall(snippet_pattern, html)

        for i, (url, title) in enumerate(links[:max_results]):
            snippet = snippets[i] if i < len(snippets) else ""
            # Clean HTML from snippet
            snippet = re.sub(r'<[^>]+>', '', snippet).strip()

            # Decode URL
            if url.startswith("//duckduckgo.com/l/?uddg="):
                url = url.split("uddg=")[1].split("&")[0]
                from urllib.parse import unquote
                url = unquote(url)

            results.append({
                "title": title.strip(),
                "url": url,
                "snippet": snippet[:300]
            })

        return results

    async def fetch_page(self, url: str) -> str:
        """Fetch and extract text content from a webpage."""
        client = await self._get_client()

        try:
            response = await client.get(url)
            if response.status_code == 200:
                html = response.text
                # Extract text content
                text = self._extract_text(html)
                return text[:5000]  # Limit content
        except Exception as e:
            return f"Error fetching page: {e}"

        return ""

    def _extract_text(self, html: str) -> str:
        """Extract readable text from HTML."""
        # Remove scripts and styles
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<head[^>]*>.*?</head>', '', html, flags=re.DOTALL | re.IGNORECASE)

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', html)

        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    async def close(self):
        if self.client:
            await self.client.aclose()


async def web_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Convenience function for web search."""
    tool = WebSearchTool()
    try:
        return await tool.search(query, max_results)
    finally:
        await tool.close()


async def fetch_url(url: str) -> str:
    """Convenience function to fetch a URL."""
    tool = WebSearchTool()
    try:
        return await tool.fetch_page(url)
    finally:
        await tool.close()

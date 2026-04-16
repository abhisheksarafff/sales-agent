import os
from dotenv import load_dotenv
from langchain.tools import tool
from tavily import TavilyClient

load_dotenv()


@tool
def search_web(query: str) -> str:
    """Search the web for competitor info, recent news, or anything
    NOT found in the internal knowledge base. Use ONLY as a fallback
    or to enrich internal KB results with current data."""

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Web search unavailable: TAVILY_API_KEY not configured."

    client = TavilyClient(api_key=api_key)
    results = client.search(query=query, search_depth="basic", max_results=5)

    output = []
    for r in results.get("results", []):
        output.append(f"**{r['title']}**\n{r['content']}\nSource: {r['url']}")

    return "\n\n---\n\n".join(output) if output else "No web results found."
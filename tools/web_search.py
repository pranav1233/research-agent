"""
Phase 2: Web Search Tool
========================
Wraps DuckDuckGo search into a plain Python function that the agent
loop can call. No API key required — ddgs talks to DuckDuckGo directly.

What this file does:
  1. Defines `web_search(inputs)` — takes a query, returns results as text.
  2. Registers it in TOOL_REGISTRY so the loop can dispatch to it.
  3. Adds its schema to TOOL_SCHEMAS so the model knows it exists.

After importing this file, the model can call web_search just like echo.
"""

from ddgs import DDGS
from agent.loop import register_tool, TOOL_SCHEMAS

MAX_RESULTS = 5     # how many DuckDuckGo results to return to the model


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------

@register_tool("web_search")
def web_search(inputs: dict) -> str:
    """
    Search DuckDuckGo and return the top results as a formatted string.

    Args (from model):
        query  : the search query string
        max_results (optional): how many results to return (default 5)

    Returns:
        A plain text block with numbered results — title, url, snippet.
        The model reads this text to formulate its answer.
    """
    query = inputs.get("query", "").strip()
    if not query:
        return "[ERROR] web_search requires a non-empty 'query' argument."

    max_results = int(inputs.get("max_results", MAX_RESULTS))

    try:
        ddgs = DDGS()
        raw_results = ddgs.text(query, max_results=max_results)
    except Exception as e:
        return f"[ERROR] DuckDuckGo search failed: {e}"

    if not raw_results:
        return f"[NO RESULTS] DuckDuckGo returned nothing for: '{query}'"

    # Format results as readable text for the model
    # Keeping it plain text (not JSON) so the model can reason over it easily
    lines = [f"Search results for: '{query}'\n"]
    for i, r in enumerate(raw_results, 1):
        lines.append(f"[{i}] {r.get('title', 'No title')}")
        lines.append(f"    URL  : {r.get('href', 'No URL')}")
        lines.append(f"    Snippet: {r.get('body', 'No snippet')[:300]}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Register schema so the model knows this tool exists
# ---------------------------------------------------------------------------
# This appends to the TOOL_SCHEMAS list imported from loop.py.
# When loop.py sends TOOL_SCHEMAS to the model, web_search is included.

TOOL_SCHEMAS.append({
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web using DuckDuckGo. Use this when you need current "
            "information, recent events, facts you are unsure about, or anything "
            "outside your training data. Returns titles, URLs, and snippets."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Be specific for better results."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 5, max 10).",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
})

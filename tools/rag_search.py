"""
Phase 3c: RAG Search Tool
==========================
Wraps the retriever into a tool the agent can call — exactly like
web_search.py did for DuckDuckGo.

Importing this file:
  1. Registers rag_search() in TOOL_REGISTRY  →  loop can dispatch it
  2. Appends its schema to TOOL_SCHEMAS        →  model knows it exists

The agent now has TWO retrieval tools:
  web_search  — live internet results (broad, current)
  rag_search  — curated arXiv passages (deep, domain-specific)

The model decides which one to call based on the question.
"""

from agent.loop import register_tool, TOOL_SCHEMAS
from rag.retriever import retriever


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------

@register_tool("rag_search")
def rag_search(inputs: dict) -> str:
    """
    Search the local arXiv knowledge base using semantic similarity.

    Args (from model):
        query    : the search string
        top_k    : how many passages to return (default 4)
    """
    query = inputs.get("query", "").strip()
    if not query:
        return "[ERROR] rag_search requires a non-empty 'query' argument."

    top_k = int(inputs.get("top_k", 4))
    return retriever.search(query, top_k=top_k)


# ---------------------------------------------------------------------------
# Register schema
# ---------------------------------------------------------------------------

TOOL_SCHEMAS.append({
    "type": "function",
    "function": {
        "name": "rag_search",
        "description": (
            "Search a curated knowledge base of arXiv ML papers using semantic search. "
            "Use this for deep technical questions about: attention mechanisms, transformers, "
            "fine-tuning, LoRA, RAG, chain-of-thought, LLMs, and related ML research. "
            "Prefer this over web_search for questions grounded in ML research papers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The search query. Use natural language — "
                        "e.g. 'how does multi-head attention work' rather than keywords."
                    )
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of passages to return (default 4).",
                    "default": 4
                }
            },
            "required": ["query"]
        }
    }
})

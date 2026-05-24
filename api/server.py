"""
Phase 4: FastAPI Server
========================
Wraps the research agent in an HTTP API so it can be called
from anywhere — a frontend, another service, or curl.

Endpoints:
  GET  /health   — confirms the server + index are loaded
  POST /ask      — accepts a query, runs the agent, returns the answer

Run with:
  uvicorn api.server:app --reload --port 8000

Then test with:
  curl -X POST http://localhost:8000/ask \
       -H "Content-Type: application/json" \
       -d '{"query": "what is LoRA?"}'
"""

import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from agent.loop import run_agent, load_tools
from rag.retriever import retriever


# ---------------------------------------------------------------------------
# Lifespan — runs once at startup before any requests are handled
# ---------------------------------------------------------------------------
# We load tools and the RAG index here so every request is fast.
# Without this, the first request would take 10+ seconds to load the model.

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy resources once at startup."""
    print("\n[startup] Loading tools...")
    load_tools()

    print("[startup] Loading RAG index...")
    retriever.load()

    print("[startup] Server ready.\n")
    yield
    # anything after yield runs on shutdown (nothing needed here)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Research Agent API",
    description=(
        "Agentic research assistant powered by Llama 3.1 + "
        "DuckDuckGo web search + FAISS RAG over arXiv papers."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response schemas  (Pydantic validates these automatically)
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    query: str = Field(
        ...,                            # required field
        min_length=3,
        description="The research question to answer.",
        examples=["What is the attention mechanism in transformers?"]
    )
    verbose: bool = Field(
        default=False,
        description="If true, prints agent iterations to server logs."
    )


class AskResponse(BaseModel):
    query: str              # echo the original query back
    answer: str             # the agent's final answer
    duration_seconds: float # how long the agent took


class HealthResponse(BaseModel):
    status: str
    rag_vectors: int        # confirms how many vectors are in the index
    tools: list[str]        # confirms which tools are registered


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    """
    Quick check that the server is up and the index is loaded.
    Hit this first to confirm everything initialized correctly.
    """
    from agent.loop import TOOL_REGISTRY

    return HealthResponse(
        status="ok",
        rag_vectors=retriever.index.ntotal if retriever.index else 0,
        tools=list(TOOL_REGISTRY.keys()),
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    Run the research agent on a query and return the answer.

    The agent will decide whether to call web_search, rag_search,
    or both — depending on the question.
    """
    try:
        start = time.time()
        answer = run_agent(request.query, verbose=request.verbose)
        duration = round(time.time() - start, 2)

        return AskResponse(
            query=request.query,
            answer=answer,
            duration_seconds=duration,
        )

    except Exception as e:
        # Return a proper HTTP error instead of crashing the server
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Run directly (alternative to uvicorn CLI)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)

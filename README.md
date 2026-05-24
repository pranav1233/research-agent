# 🔍 Multi-Tool Research Agent

An agentic research assistant that answers complex questions by reasoning over two retrieval tools — live web search and a semantic knowledge base of arXiv ML papers. Built with a local LLM, zero cloud dependencies, and a reproducible evaluation benchmark.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│              Agent Loop                 │
│         (llama3.1:8b via Ollama)        │
│                                         │
│  1. Receive query + tool schemas        │
│  2. Decide: answer directly or use      │
│     a tool?                             │
│  3. If tool → dispatch → get result     │
│  4. Feed result back, repeat            │
│  5. When done → return final answer     │
└──────────────┬──────────────────────────┘
               │ dispatches to one or both
       ┌───────┴────────┐
       │                │
       ▼                ▼
┌─────────────┐  ┌──────────────────────┐
│  web_search │  │      rag_search      │
│ (DuckDuckGo)│  │  (FAISS + arXiv)     │
│             │  │                      │
│ Live results│  │ Semantic search over │
│ No API key  │  │ curated ML papers    │
└─────────────┘  └──────────┬───────────┘
                            │
                   ┌────────┴────────┐
                   │  all-MiniLM-L6  │
                   │  384-dim vectors│
                   │  FAISS IndexL2  │
                   └─────────────────┘

Served via FastAPI  →  POST /ask
Measured via        →  50-question benchmark
```

---

## Results

| Category | Correct | Total | Accuracy |
|----------|---------|-------|----------|
| RAG only | 15 | 20 | **75.0%** |
| Web only | 11 | 15 | **73.3%** |
| Multi-hop | 8 | 15 | **53.3%** |
| **Overall** | **34** | **50** | **68.0%** |

**Key finding:** Multi-hop performance (53%) is bottlenecked by the 8B local model's tool-chaining reliability — it occasionally calls only one tool when both are needed. RAG and web-only categories both exceed 73%, confirming the retrieval tools themselves work well.

---

## Project Structure

```
research-agent/
├── agent/
│   └── loop.py           # Core agentic loop — the tool-use engine
├── tools/
│   ├── web_search.py     # DuckDuckGo tool (free, no API key)
│   └── rag_search.py     # FAISS semantic search tool
├── rag/
│   ├── ingest.py         # Fetch arXiv papers → chunk → embed → FAISS
│   └── retriever.py      # Load index, embed query, return top-k chunks
├── api/
│   └── server.py         # FastAPI server — GET /health, POST /ask
├── eval/
│   ├── questions.json    # 50-question benchmark (20 RAG, 15 web, 15 multi-hop)
│   └── harness.py        # Keyword-recall scoring harness → results.csv
├── run.py                # Interactive CLI runner
└── requirements.txt
```

---

## Quickstart

### 1. Install dependencies

```bash
# Install Ollama from https://ollama.com
ollama pull llama3.1:8b

# Python setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Build the knowledge base

```bash
# Fetches arXiv papers, chunks, embeds, saves FAISS index to data/
python -m rag.ingest
```

### 3. Run the interactive CLI

```bash
python run.py
```

### 4. Run the API server

```bash
uvicorn api.server:app --reload --port 8000
```

```bash
# Health check
curl http://localhost:8000/health

# Ask a question
curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"query": "What is LoRA fine-tuning?"}'
```

### 5. Run the evaluation benchmark

```bash
python -m eval.harness            # full 50 questions
python -m eval.harness --limit 5  # quick smoke test
python -m eval.harness --category rag_only
```

---

## How the Agent Loop Works

The agent operates as a reasoning loop driven by the model's `finish_reason`:

```python
while True:
    response = llm(messages, tools=TOOL_SCHEMAS)

    if response.finish_reason == "stop":
        return response.text              # done

    if response.finish_reason == "tool_calls":
        for tool_call in response.tool_calls:
            result = dispatch(tool_call)  # run the tool locally
            messages.append(tool_result) # feed result back
        # loop again — model now has the tool output
```

The model never has internet access directly. It emits structured tool calls; your Python code executes them and returns plain text. The model reasons over that text to decide whether to call another tool or answer.

---

## How RAG Works

```
BUILD (once)                        QUERY (per request)
─────────────────────               ─────────────────────
arXiv API                           User query string
    ↓                                   ↓
Paper abstracts                     Embed with MiniLM
    ↓                                   ↓  (384-dim vector)
Chunk (500 chars,                   FAISS.search(query_vec, k=4)
 100 char overlap)                      ↓
    ↓                               Top-k chunk indices
Embed with MiniLM                       ↓
    ↓  (384-dim vectors)            Lookup text in chunks.json
FAISS IndexFlatL2                       ↓
    ↓                               Return formatted passages
Save to disk                            ↓
                                    Model reads and answers
```

---

## Evaluation Methodology

Questions are scored using **keyword recall**: an answer is correct if it contains at least `min_facts_required` of the expected `key_facts` (case-insensitive substring match).

```json
{
  "question": "What is LoRA?",
  "key_facts": ["low-rank", "parameter", "fine-tuning", "frozen", "matrices"],
  "min_facts_required": 3
}
```

This approach is transparent, reproducible, and consistent with factoid QA benchmarks in the literature (e.g. TriviaQA, Natural Questions). The 50 questions span three categories to measure different capabilities:

- **rag_only** — tests knowledge base depth and retrieval quality
- **web_only** — tests live search grounding
- **multi_hop** — tests the model's ability to chain multiple tool calls

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | `llama3.1:8b` via Ollama (local, free) |
| Tool-use | OpenAI-compatible function calling API |
| Web search | DuckDuckGo (`ddgs`) — no API key required |
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers |
| Vector store | FAISS `IndexFlatL2` |
| Document source | arXiv API (ML papers) |
| API server | FastAPI + Uvicorn |
| Evaluation | Custom keyword-recall harness |

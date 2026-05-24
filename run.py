"""
Research Agent runner — Phase 3+
Run from project root: python run.py
"""

from agent.loop import run_agent, load_tools
from rag.retriever import retriever

# Load all tools (registers web_search + rag_search into the loop)
load_tools()

# Preload the RAG index at startup — so the first query isn't slow
retriever.load()

if __name__ == "__main__":
    print("\nResearch Agent ready. Two tools available:")
    print("  web_search  — live DuckDuckGo results")
    print("  rag_search  — local arXiv knowledge base")
    print("\nType your question. Ctrl+C to quit.\n")

    while True:
        try:
            query = input("You: ").strip()
            if not query:
                continue
            answer = run_agent(query, verbose=True)
            print(f"\nAgent: {answer}\n")
        except KeyboardInterrupt:
            print("\nBye!")
            break

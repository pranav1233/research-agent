"""
Phase 2 runner — tests the agent with real web search.
Run from the project root: python run.py
"""

from agent.loop import run_agent, load_tools

# Load all tools BEFORE calling run_agent
# This registers web_search into TOOL_REGISTRY and TOOL_SCHEMAS
load_tools()

if __name__ == "__main__":
    print("Research Agent — Phase 2 (Web Search)\n")
    print("Type your question. Ctrl+C to quit.\n")

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

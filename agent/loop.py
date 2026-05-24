"""
Phase 1: Core Agent Loop
========================
Implements the tool-use agentic loop using Ollama (local LLM).

Ollama exposes an OpenAI-compatible API, so we use the `openai` Python
package pointed at localhost:11434. The tool-use format follows the
OpenAI function-calling spec:

  finish_reason == "tool_calls"  →  model wants to call tools
  finish_reason == "stop"        →  model is done, extract text

Compared to Anthropic's format:
  Anthropic                          OpenAI / Ollama
  ─────────────────────────────────────────────────
  stop_reason = "tool_use"           finish_reason = "tool_calls"
  stop_reason = "end_turn"           finish_reason = "stop"
  block.type == "tool_use"           tool_call.function.name
  block.input  (dict)                json.loads(tool_call.function.arguments)
  role: "user", type: "tool_result"  role: "tool", tool_call_id: ...

Everything else — the loop structure, tool registry, dispatcher — is
identical. Swapping back to Claude later = changing ~20 lines.
"""

import json
from typing import Any
from openai import OpenAI

# ---------------------------------------------------------------------------
# Client  — points to Ollama's OpenAI-compatible local server
# ---------------------------------------------------------------------------

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",           # Ollama doesn't need a real key; string required
)

MODEL = "llama3.1:8b"
MAX_ITERATIONS = 10             # safety cap — prevents infinite loops

# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------
# Maps tool_name → callable(input_dict) -> str
# Phase 2 adds web_search; Phase 3 adds rag_search.

TOOL_REGISTRY: dict[str, Any] = {}


def register_tool(name: str):
    """Decorator to register a Python function as an agent tool."""
    def decorator(fn):
        TOOL_REGISTRY[name] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Tool schemas  (OpenAI function-calling format)
# ---------------------------------------------------------------------------
# The model reads these to know what tools exist, when to call them,
# and what arguments to pass.

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "echo",
            "description": (
                "Repeats back the provided text. "
                "Use this ONLY for testing the agent loop."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to echo back."
                    }
                },
                "required": ["text"]
            }
        }
    }
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

@register_tool("echo")
def echo(inputs: dict) -> str:
    return f"[ECHO] {inputs['text']}"


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

def dispatch_tool(tool_name: str, tool_input: dict) -> str:
    """
    Look up and run a tool by name.
    Returns a plain string — fed back to the model as a tool message.
    """
    if tool_name not in TOOL_REGISTRY:
        return f"[ERROR] Unknown tool: '{tool_name}'"
    try:
        return str(TOOL_REGISTRY[tool_name](tool_input))
    except Exception as e:
        return f"[ERROR] Tool '{tool_name}' raised: {e}"


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_agent(query: str, verbose: bool = True) -> str:
    """
    Run the agentic loop for a given user query.

    Args:
        query:   The user's question or instruction.
        verbose: Print each step so you can watch the loop in real time.

    Returns:
        The final text answer from the model.
    """
    messages = [{"role": "user", "content": query}]

    for iteration in range(1, MAX_ITERATIONS + 1):
        if verbose:
            print(f"\n{'='*60}")
            print(f"  ITERATION {iteration}")
            print(f"{'='*60}")
            print(f"  Sending {len(messages)} message(s) to model...")

        # ── Step 1: call the model ──────────────────────────────────────
        response = client.chat.completions.create(
            model=MODEL,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        choice = response.choices[0]
        finish_reason = choice.finish_reason

        if verbose:
            print(f"  finish_reason : {finish_reason}")

        # ── Step 2: decide what to do ───────────────────────────────────

        if finish_reason == "stop":
            # Model is done — return the text answer
            answer = choice.message.content or "[Agent produced no text output]"
            if verbose:
                print(f"\n  FINAL ANSWER:\n  {answer}")
            return answer

        if finish_reason == "tool_calls":
            # Append the assistant's message (contains tool_call objects)
            messages.append(choice.message)

            # Execute every tool the model requested
            for tool_call in choice.message.tool_calls:
                name = tool_call.function.name
                # arguments arrive as a JSON string — parse to dict
                try:
                    inputs = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    inputs = {}

                if verbose:
                    print(f"\n  TOOL CALL  : {name}")
                    print(f"  TOOL INPUT : {json.dumps(inputs, indent=2)}")

                result = dispatch_tool(name, inputs)

                if verbose:
                    print(f"  TOOL RESULT: {result}")

                # Feed result back — role must be "tool" in OpenAI format
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

            continue  # go back to top of loop

        raise RuntimeError(f"Unexpected finish_reason: {finish_reason}")

    raise RuntimeError(f"Agent exceeded MAX_ITERATIONS ({MAX_ITERATIONS})")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Phase 1 smoke test — echo tool only\n")
    answer = run_agent(
        "Please echo the phrase 'Hello from the agent loop!' "
        "and then tell me what you just echoed.",
        verbose=True,
    )
    print(f"\n{'='*60}")
    print("Smoke test passed ✓")


# ---------------------------------------------------------------------------
# Phase 2: load tools (import triggers registration via decorators)
# ---------------------------------------------------------------------------
# Importing tools/web_search.py runs the module which:
#   1. Decorates web_search() with @register_tool → adds to TOOL_REGISTRY
#   2. Appends its schema to TOOL_SCHEMAS → model sees it on next call
# This pattern means adding a new tool = just adding one import line here.

def load_tools():
    """Import all tool modules. Must be called before run_agent()."""
    import tools.web_search  # noqa: F401  Phase 2
    # import tools.rag_search  # noqa: F401  Phase 3 (uncomment later)

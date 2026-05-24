"""
Phase 5: Evaluation Harness
============================
Runs all 50 benchmark questions through the agent and measures accuracy.

Evaluation method: keyword recall
  Each question has a list of key_facts and a min_facts_required threshold.
  An answer is CORRECT if it contains at least min_facts_required of the
  key_facts as substrings (case-insensitive).

  This is simple, transparent, and reproducible — the same approach used
  in many LLM evaluation papers (e.g. TriviaQA, Natural Questions).

Output:
  - Live progress printed to console
  - results.csv saved to eval/ with per-question details
  - Final accuracy breakdown by category

Run with:
  python -m eval.harness
  python -m eval.harness --limit 10      # test on first 10 questions only
  python -m eval.harness --category rag_only
"""

import json
import time
import argparse
import csv
from pathlib import Path
from tqdm import tqdm

from agent.loop import run_agent, load_tools
from rag.retriever import retriever

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

QUESTIONS_PATH  = "eval/questions.json"
RESULTS_PATH    = "eval/results.csv"
VERBOSE_AGENT   = False     # set True to see full agent loop per question


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_answer(answer: str, key_facts: list[str], min_required: int) -> tuple[bool, int]:
    """
    Check if the answer contains enough key facts to be considered correct.

    Args:
        answer      : the agent's raw answer string
        key_facts   : list of expected facts/keywords
        min_required: how many facts must be present to count as correct

    Returns:
        (is_correct, facts_found_count)
    """
    answer_lower = answer.lower()
    found = sum(1 for fact in key_facts if fact.lower() in answer_lower)
    return found >= min_required, found


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def run_eval(questions: list[dict]) -> list[dict]:
    """
    Run each question through the agent and score the answer.
    Returns a list of result dicts (one per question).
    """
    results = []

    for q in tqdm(questions, desc="Evaluating"):
        start = time.time()

        try:
            answer = run_agent(q["question"], verbose=VERBOSE_AGENT)
            error = None
        except Exception as e:
            answer = ""
            error = str(e)

        duration = round(time.time() - start, 2)

        is_correct, facts_found = score_answer(
            answer,
            q["key_facts"],
            q["min_facts_required"],
        )

        result = {
            "id"              : q["id"],
            "category"        : q["category"],
            "question"        : q["question"],
            "answer"          : answer[:300],       # truncate for CSV readability
            "key_facts"       : ", ".join(q["key_facts"]),
            "min_required"    : q["min_facts_required"],
            "facts_found"     : facts_found,
            "correct"         : is_correct,
            "duration_seconds": duration,
            "error"           : error or "",
        }
        results.append(result)

        # Live feedback so you can watch progress
        status = "✓" if is_correct else "✗"
        print(f"  [{status}] Q{q['id']:02d} ({q['category']}) — {facts_found}/{len(q['key_facts'])} facts — {duration}s")

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(results: list[dict]) -> None:
    """Print accuracy breakdown by category."""
    total   = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total * 100

    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Overall accuracy : {correct}/{total} = {accuracy:.1f}%")

    # Breakdown by category
    categories = sorted(set(r["category"] for r in results))
    print(f"\n  By category:")
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        cat_correct = sum(1 for r in cat_results if r["correct"])
        cat_total   = len(cat_results)
        cat_pct     = cat_correct / cat_total * 100
        print(f"    {cat:<12} {cat_correct}/{cat_total} = {cat_pct:.1f}%")

    # Slowest questions (useful to spot timeouts)
    print(f"\n  Slowest 3 questions:")
    slowest = sorted(results, key=lambda r: r["duration_seconds"], reverse=True)[:3]
    for r in slowest:
        print(f"    Q{r['id']:02d}: {r['duration_seconds']}s — {r['question'][:60]}")

    print(f"\n  Full results saved → {RESULTS_PATH}")


def save_csv(results: list[dict]) -> None:
    """Save all results to a CSV for further analysis."""
    Path(RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)
    fields = results[0].keys()
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run the research agent eval harness")
    parser.add_argument("--limit",    type=int,   default=None, help="Only run first N questions")
    parser.add_argument("--category", type=str,   default=None, help="Filter by category (rag_only, web_only, multi_hop)")
    args = parser.parse_args()

    # Load tools + RAG index
    print("Loading tools and RAG index...")
    load_tools()
    retriever.load()

    # Load questions
    with open(QUESTIONS_PATH) as f:
        questions = json.load(f)

    if args.category:
        questions = [q for q in questions if q["category"] == args.category]
        print(f"Filtered to category '{args.category}': {len(questions)} questions")

    if args.limit:
        questions = questions[:args.limit]
        print(f"Limited to first {args.limit} questions")

    print(f"\nRunning {len(questions)} questions...\n")

    results  = run_eval(questions)
    save_csv(results)
    print_report(results)


if __name__ == "__main__":
    main()

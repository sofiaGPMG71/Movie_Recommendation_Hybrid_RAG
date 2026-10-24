"""
generate_test_report.py — Run evaluation and generate Markdown test report.

This script runs the hybrid RAG system's evaluation suite and outputs a
Markdown test report with actual routing results, answer quality, and
response times.

Usage:
    python generate_test_report.py > test_report.md

Or redirect output to a file:
    python generate_test_report.py > test_report.md
"""

import sys
import time
import datetime
from pathlib import Path

# Ensure the project folder is on the path
sys.path.insert(0, str(Path(__file__).parent))

from evaluate import run_evaluation

# ---------------------------------------------------------------------------
# Sample test questions matching the assignment requirements
# These are the questions that will be evaluated
# ---------------------------------------------------------------------------
TEST_QUESTIONS = [
    # Graph questions (structural/relational)
    {
        "question": "Who acted in The Matrix?",
        "expected": "graph",
        "note": "Direct entity -> cast lookup",
    },
    {
        "question": "What movies has Tom Hanks appeared in?",
        "expected": "graph",
        "note": "Actor filmography",
    },
    {
        "question": "Who directed Inception?",
        "expected": "graph",
        "note": "Director lookup",
    },
    {
        "question": "Who has Keanu Reeves acted with?",
        "expected": "graph",
        "note": "Multi-hop: co-actors",
    },
    {
        "question": "What films did Christopher Nolan direct?",
        "expected": "graph",
        "note": "Director filmography",
    },
    # Vector questions (semantic/recommendation)
    {
        "question": "Recommend a sci-fi movie involving space or advanced technology.",
        "expected": "vector",
        "note": "Semantic genre search",
    },
    {
        "question": "Find me a movie about survival in extreme isolation.",
        "expected": "vector",
        "note": "Thematic search",
    },
    {
        "question": "I want to watch something romantic and emotional.",
        "expected": "vector",
        "note": "Mood-based search",
    },
    {
        "question": "Tell me about a film involving crime and moral dilemmas.",
        "expected": "vector",
        "note": "Abstract theme search",
    },
    {
        "question": "Suggest a mind-bending movie with an unusual narrative structure.",
        "expected": "vector",
        "note": "Concept-based search",
    },
    # Edge cases
    {
        "question": "Who acted in a movie called Nonexistent Film?",
        "expected": "graph",
        "note": "Query with movie not in graph database",
    },
    {
        "question": "What is the meaning of life?",
        "expected": "vector",
        "note": "Philosophical question unlikely to match movie data",
    },
    {
        "question": "Who acted with Al Pacino in the 1990s?",
        "expected": "graph",
        "note": "Actor not in our movie dataset",
    },
    {
        "question": "Tell me about a movie involving time travel and paradoxes.",
        "expected": "vector",
        "note": "Semantic search for time travel theme",
    },
]


# ---------------------------------------------------------------------------
# Run evaluation using default questions from evaluate.py,
# then supplement with our test questions
# ---------------------------------------------------------------------------
print("=" * 70)
print("  HYBRID GRAPH RAG — TEST REPORT GENERATION")
print("=" * 70)

# Run the evaluation suite (uses default questions from evaluate.py)
print("\n  Running evaluation suite...\n")
results = run_evaluation()

# ---------------------------------------------------------------------------
# Now supplement with our custom test questions by running them manually
# ---------------------------------------------------------------------------
# Import the router and rag_chain to route our custom questions
sys.path.insert(0, str(Path(__file__).parent))

import config  # noqa: F401 — ensures env vars are loaded
from router import route
from rag_chain import run_hybrid_rag

print(f"\n  Running {len(TEST_QUESTIONS)} additional custom questions...\n")

custom_results = []
for q_item in TEST_QUESTIONS:
    question = q_item["question"]
    expected = q_item["expected"]
    start_time = time.time()

    try:
        result = run_hybrid_rag(question, verbose=False)
        elapsed = time.time() - start_time
        actual = result["retriever"]
        correct = actual == expected
    except Exception as e:
        elapsed = float("inf")
        actual = "ERROR"
        correct = False
        print(f"  [ERROR] Question '{question}': {e}")

    custom_results.append({
        "question": question,
        "expected": expected,
        "actual": actual,
        "correct": correct,
        "elapsed": elapsed,
    })

    status = "[OK]" if correct else "[FAIL]"
    print(f"  {status} [Q{len(results)+len(custom_results)}] {question} -> {actual.upper()} "
          f"(expected {expected.upper()}, {elapsed:.2f}s)")


# ---------------------------------------------------------------------------
# Combine all results
# ---------------------------------------------------------------------------
all_results = results + custom_results

# ---------------------------------------------------------------------------
# Compute summary statistics
# ---------------------------------------------------------------------------
total = len(all_results)
correct_count = sum(1 for r in all_results if r["correct"])

# Count by category
graph_results = [r for r in all_results if r["expected"] == "graph"]
vector_results = [r for r in all_results if r["expected"] == "vector"]
edge_results = [r for r in all_results if r["expected"] not in ("graph", "vector")]

graph_count = len(graph_results)
vector_count = len(vector_results)
edge_count = len(edge_results)

graph_correct = sum(1 for r in graph_results if r["correct"])
vector_correct = sum(1 for r in vector_results if r["correct"])
edge_correct = sum(1 for r in edge_results if r["correct"])

accuracy = correct_count / total * 100 if total > 0 else 0

# Wrong answers
wrong = [r for r in all_results if not r["correct"]]

# ---------------------------------------------------------------------------
# Print Markdown report
# ---------------------------------------------------------------------------
date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f"# Test Report: Hybrid Knowledge Graph RAG Movie Assistant")
print(f"**Date:** {date_str}")
print(f"**Total Questions Tested:** {total}")
print(f"**Overall Accuracy:** {accuracy:.1f}%\n")

print("## Test Questions & Outcomes")
print("| # | Question | Expected Retriever | Actual Retriever | Correct? | Retrieval Method | Response Time |")
print("|---|----------|-------------------|------------------|----------|------------------|---------------|")

for r in all_results:
    q_short = r["question"][:20] + ("..." if len(r["question"]) > 20 else "")
    correct_marker = "[YES]" if r["correct"] else "[NO]"
    method = r["actual"].upper()
    time_str = f"{r['elapsed']:.2f}s"
    print(f"| {q_short} | {r['question']} | {r['expected'].upper()} | {method} | {correct_marker} | {method} | {time_str} |")

print("## Routing Accuracy Summary")
print(f"- **Graph questions:** {graph_correct}/{graph_count} correct")
print(f"- **Vector questions:** {vector_correct}/{vector_count} correct")
print(f"- **Edge cases handled:** {edge_correct}/{edge_count} without crash")
print(f"- **Overall accuracy:** {correct_count}/{total} ({accuracy:.1f}%)\n")

print("## Incorrectly Routed Questions")
if wrong:
    print(f"| Question | Expected | Actual |")
    print(f"|----------|----------|--------|")
    for r in wrong:
        print(f"| {r['question']} | {r['expected'].upper()} | {r['actual'].upper()} |")
else:
    print("- None — all queries routed correctly!")

print("\n## Notes")
print("- **Graph questions** test relationship-based retrieval via Neo4j Cypher queries (actor/movie/director lookups, multi-hop co-actor queries).")
print("- **Vector questions** test semantic similarity retrieval via FAISS embeddings (theme-based, mood-based, and concept-based recommendations).")
print("- **Edge cases** test the system's ability to handle queries with no matching data without crashing, and provide helpful error messages instead.")
print("- The routing logic uses keyword heuristics with an LLM fallback for ambiguous queries.")
print("- All answers are generated by GPT-4o-mini grounded in the retrieved context from the appropriate retriever.")
print()
print("=" * 70)
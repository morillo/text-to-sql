"""Evaluation framework for text-to-SQL quality.

Metrics
───────
We use two complementary metrics, not just one:

  Execution Accuracy (EX)
      Fraction of questions where the generated SQL runs without a
      sqlite3.Error.  A necessary but not sufficient condition for correctness.

  Result Accuracy (RA)
      Fraction of questions where the result set of the generated SQL exactly
      matches the gold-standard result set.  This is the meaningful measure of
      correctness.  We compare result *sets* (row order– and column order–
      independent), not SQL strings, because many correct queries look different
      in text but produce identical data.

Baseline comparison
───────────────────
The simple baseline prompt:

    "Convert this question to SQL: {question}"

…provides no schema context, no dialect hints, and no output-format guidance.
We run this same (minimal) prompt against the same model and same questions so
we can report a concrete before/after comparison.

Usage
─────
  python -m src.evals                        # full run with baseline comparison
  python -m src.evals --no-baseline          # agent only, skips baseline calls
  python -m src.evals --output results.json  # save detailed results to file
"""

import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from src.agent import TextToSQLAgent
from src.utils import load_db, query_db


# ─── Baseline implementation ──────────────────────────────────────────────────

# We use the same model for baseline vs. agent so the only variable is the
# prompt engineering — this isolates the quality improvement clearly.

def _run_baseline_query(question: str, client: OpenAI, model: str) -> Dict[str, Any]:
    """
    Execute the original baseline prompt with no schema context.

    This reproduces the baseline setup:
        "Convert this question to SQL: {question}"

    Args:
        question: Natural language question.
        client:   OpenAI-compatible client.
        model:    The model identifier.

    Returns:
        dict with keys: sql, usage
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": f"Convert this question to SQL:\n{question}"}
        ],
        temperature=0.1,
        max_tokens=512,
    )
    raw = response.choices[0].message.content or ""

    # Try to extract a bare SQL statement from whatever the model returned
    sql = _extract_sql_heuristic(raw)
    return {
        "sql": sql,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        } if response.usage else {},
    }


def _extract_sql_heuristic(text: str) -> str:
    """Extract SQL from free-form baseline model output."""
    text = text.strip()
    m = re.search(r"```sql\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"((?:SELECT|WITH)\s+.+?)(?:;|\Z)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


# ─── Result comparison ────────────────────────────────────────────────────────

def _normalize_value(v: Any) -> Any:
    """
    Normalize a cell value for comparison.

    - Floats are rounded to 2 decimal places to absorb floating-point drift.
    - Strings are stripped of surrounding whitespace.
    - Everything else is returned as-is.
    """
    if isinstance(v, float):
        return round(v, 2)
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        return v.strip()
    return v


def _row_to_value_tuple(row: Dict) -> Tuple:
    """
    Convert a row dict to a sorted tuple of normalized VALUES only.

    We intentionally exclude column names from the comparison.  This matches
    the standard "Execution Accuracy" (EX) metric used in text-to-SQL research
    benchmarks such as Spider and BIRD.  Two queries that return the same data
    with different column aliases are both considered correct — the column name
    is a presentational choice, not a semantic one.

    Example: {"TotalSales": 826.65} and {"Revenue": 826.65} are treated equal.

    Values are sorted by their string representation to handle mixed types
    (e.g. a row that contains both a str and a float column).
    """
    normalized = [_normalize_value(v) for v in row.values()]
    return tuple(sorted(normalized, key=str))


def compare_results(
    predicted: Optional[List[Dict]],
    expected: List[Dict],
) -> bool:
    """
    Compare two result sets for semantic equality (values only, standard EX metric).

    Row order and column order are both ignored.  Column names are also ignored
    — only the data values matter.  Float values are rounded to 2 decimal places
    to absorb floating-point drift from SQLite arithmetic.

    Args:
        predicted: Rows returned by the agent's generated SQL (or None on error).
        expected:  Gold-standard expected rows from dev_questions_with_answers.json.

    Returns:
        True if the result sets are semantically identical.
    """
    if predicted is None:
        return False
    if len(predicted) != len(expected):
        return False

    pred_sorted = sorted(_row_to_value_tuple(r) for r in predicted)
    exp_sorted  = sorted(_row_to_value_tuple(r) for r in expected)
    return pred_sorted == exp_sorted


# ─── Per-question evaluation ──────────────────────────────────────────────────

def _eval_one(
    question: Dict,
    agent: TextToSQLAgent,
    baseline_client: Optional[OpenAI],
    conn: sqlite3.Connection,
) -> Dict:
    """
    Evaluate a single question against the agent (and optionally the baseline).

    Args:
        question:         One entry from dev_questions_with_answers.json.
        agent:            Initialised TextToSQLAgent.
        baseline_client:  OpenAI-compatible client for baseline, or None to skip.
        conn:             Live database connection (for executing baseline SQL).

    Returns:
        dict with per-question results for both agent and baseline.
    """
    q_id      = question["id"]
    text      = question["question"]
    expected  = question["expected_result"]
    tier      = question["tier"]
    gold_sql  = question["gold_sql"]

    # ── Agent ─────────────────────────────────────────────────────────────────
    agent.reset_conversation()
    pred = agent.query(text)

    agent_exec_ok   = pred["error"] is None
    agent_result_ok = compare_results(pred["results"], expected) if agent_exec_ok else False

    agent_entry = {
        "id":            q_id,
        "tier":          tier,
        "question":      text,
        "gold_sql":      gold_sql,
        "predicted_sql": pred["sql"],
        "exec_success":  agent_exec_ok,
        "result_match":  agent_result_ok,
        "error":         pred.get("error"),
        "attempts":      pred.get("attempts", 1),
    }

    # ── Baseline ──────────────────────────────────────────────────────────────
    baseline_entry = None
    if baseline_client is not None:
        base = _run_baseline_query(text, baseline_client, agent.model)

        base_exec_ok   = False
        base_result_ok = False
        base_error     = None

        if base["sql"]:
            try:
                base_rows      = query_db(conn, base["sql"], return_as_df=False)
                base_exec_ok   = True
                base_result_ok = compare_results(base_rows, expected)
            except sqlite3.Error as e:
                base_error = str(e)
        else:
            base_error = "No SQL extracted from baseline response"

        baseline_entry = {
            "id":            q_id,
            "tier":          tier,
            "predicted_sql": base["sql"],
            "exec_success":  base_exec_ok,
            "result_match":  base_result_ok,
            "error":         base_error,
        }

    return {"agent": agent_entry, "baseline": baseline_entry}


# ─── Reporting ────────────────────────────────────────────────────────────────

def _print_summary(agent_rows: List[Dict], baseline_rows: Optional[List[Dict]]) -> None:
    """Print a formatted evaluation summary table."""
    n = len(agent_rows)
    a_exec  = sum(r["exec_success"]  for r in agent_rows) / n
    a_res   = sum(r["result_match"]  for r in agent_rows) / n

    has_baseline = baseline_rows is not None and len(baseline_rows) > 0
    if has_baseline:
        b_exec = sum(r["exec_success"]  for r in baseline_rows) / n
        b_res  = sum(r["result_match"]  for r in baseline_rows) / n

    width = 66 if has_baseline else 46
    print()
    print("=" * width)
    print("EVALUATION SUMMARY")
    print("=" * width)

    hdr = f"  {'Metric':<28}  {'Agent':>8}"
    if has_baseline:
        hdr += f"  {'Baseline':>8}  {'Δ':>6}"
    print(hdr)
    print("  " + "-" * (width - 2))

    def row(label, a_val, b_val=None):
        line = f"  {label:<28}  {a_val:>7.1%}"
        if has_baseline and b_val is not None:
            delta = a_val - b_val
            line += f"  {b_val:>7.1%}  {delta:>+6.1%}"
        print(line)

    row("Execution Accuracy",  a_exec,  b_exec if has_baseline else None)
    row("Result Accuracy",     a_res,   b_res  if has_baseline else None)

    # Per-tier breakdown (agent only)
    print()
    print("  Per-Tier Result Accuracy (Agent):")
    for tier in (1, 2, 3):
        tier_rows = [r for r in agent_rows if r["tier"] == tier]
        if tier_rows:
            acc = sum(r["result_match"] for r in tier_rows) / len(tier_rows)
            bar = "█" * int(acc * 10) + "░" * (10 - int(acc * 10))
            print(f"    Tier {tier}  {bar}  {acc:.0%}  ({len(tier_rows)} questions)")

    # Failure details
    failures = [r for r in agent_rows if not r["result_match"]]
    if failures:
        print()
        print("  Failure analysis (Agent):")
        for f in failures:
            if not f["exec_success"]:
                reason = f"EXEC ERROR — {f['error']}"
            else:
                reason = "WRONG RESULT — SQL executed but data didn't match"
            print(f"    [{f['id']}] Tier {f['tier']}: {reason[:70]}")
    else:
        print()
        print("  No failures — all questions answered correctly.")

    print("=" * width)
    print()


# ─── Public API ───────────────────────────────────────────────────────────────

def run_eval(
    questions_path: str = "data/dev_questions_with_answers.json",
    run_baseline: bool = True,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    """
    Run the full evaluation suite.

    Args:
        questions_path: Path to the dev_questions_with_answers.json file.
        run_baseline:   If True, also evaluate the baseline prompt.
        output_path:    If set, write detailed JSON results to this file.
        verbose:        If True, print per-question progress and summary.

    Returns:
        dict with keys:
          agent     – {execution_accuracy, result_accuracy, details}
          baseline  – same structure, or None if run_baseline=False
    """
    conn  = load_db()
    agent = TextToSQLAgent(conn)

    baseline_client = None
    if run_baseline:
        baseline_client = agent.client

    with open(questions_path) as fh:
        questions = json.load(fh)

    if verbose:
        print(f"\nRunning evaluation on {len(questions)} questions")
        print(f"  Agent model : {agent.model}")
        print(f"  Baseline    : {'enabled (same model, no schema)' if run_baseline else 'disabled'}")
        print()

    agent_rows: List[Dict]    = []
    baseline_rows: List[Dict] = []

    for q in questions:
        if verbose:
            tier_label = f"[Tier {q['tier']}]"
            print(f"  {q['id']} {tier_label:<8} {q['question'][:55]}...")

        one = _eval_one(q, agent, baseline_client, conn)

        agent_rows.append(one["agent"])

        a = one["agent"]
        if verbose:
            a_sym = "✓" if a["result_match"] else ("⚡" if not a["exec_success"] else "✗")
            print(f"           Agent: {a_sym}", end="")

        if one["baseline"] is not None:
            baseline_rows.append(one["baseline"])
            b = one["baseline"]
            if verbose:
                b_sym = "✓" if b["result_match"] else ("⚡" if not b["exec_success"] else "✗")
                print(f"  Baseline: {b_sym}", end="")

        if verbose:
            print()

    n = len(questions)
    metrics = {
        "agent": {
            "execution_accuracy": sum(r["exec_success"] for r in agent_rows) / n,
            "result_accuracy":    sum(r["result_match"] for r in agent_rows) / n,
            "details": agent_rows,
        },
        "baseline": {
            "execution_accuracy": sum(r["exec_success"] for r in baseline_rows) / n,
            "result_accuracy":    sum(r["result_match"] for r in baseline_rows) / n,
            "details": baseline_rows,
        } if baseline_rows else None,
    }

    if verbose:
        _print_summary(agent_rows, baseline_rows if baseline_rows else None)

    if output_path:
        with open(output_path, "w") as fh:
            json.dump(metrics, fh, indent=2)
        if verbose:
            print(f"Detailed results written to {output_path}\n")

    return metrics


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run text-to-SQL evaluation")
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip the baseline prompt evaluation (saves API calls)",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help="Save detailed results JSON to FILE",
    )
    args = parser.parse_args()

    run_eval(
        run_baseline=not args.no_baseline,
        output_path=args.output,
    )

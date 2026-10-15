"""Evaluation framework for text-to-SQL quality.

Metrics
───────
  Execution Accuracy (EX)
      Fraction of questions where the generated SQL runs without sqlite3.Error.

  Result Accuracy (RA)
      Fraction of questions where the result set of the generated SQL exactly
      matches the gold-standard expected result set.

Datasets
────────
  - Dev set: data/dev_questions_with_answers.json (10 questions)
  - Extended set: data/extended_questions_with_answers.json (10 additional edge-case questions)

Usage
─────
  python -m src.evals
  python -m src.evals --dataset extended
  python -m src.evals --dataset all --output eval_results.json
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

from src.agent import TextToSQLAgent, DEFAULT_MODEL
from src.utils import load_db, query_db


def _run_baseline_query(question: str, client: OpenAI, model: str) -> Dict[str, Any]:
    """Execute simple baseline prompt with no schema context."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": f"Convert this question to SQL:\n{question}"}
        ],
        temperature=0.0,
        max_tokens=512,
    )
    raw = response.choices[0].message.content or ""
    sql = _extract_sql_heuristic(raw)
    return {
        "sql": sql,
        "usage": {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
            "completion_tokens": getattr(response.usage, "completion_tokens", 0),
            "total_tokens": getattr(response.usage, "total_tokens", 0),
        },
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


def _normalize_value(v: Any) -> Any:
    """Normalize cell values for robust floating point and string comparison."""
    if isinstance(v, float):
        return round(v, 2)
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        return v.strip()
    return v


def _row_to_value_tuple(row: Dict) -> Tuple:
    """Convert row dict to a sorted tuple of normalized values."""
    normalized = [_normalize_value(v) for v in row.values()]
    return tuple(sorted(normalized, key=str))


def compare_results(
    predicted: Optional[List[Dict]],
    expected: List[Dict],
) -> bool:
    """Compare two result sets for semantic data equality."""
    if predicted is None:
        return False
    if len(predicted) != len(expected):
        return False

    pred_sorted = sorted(_row_to_value_tuple(r) for r in predicted)
    exp_sorted = sorted(_row_to_value_tuple(r) for r in expected)
    return pred_sorted == exp_sorted


def _eval_one(
    question: Dict,
    agent: TextToSQLAgent,
    baseline_client: Optional[OpenAI],
    conn: sqlite3.Connection,
) -> Dict:
    q_id = question["id"]
    text = question["question"]
    expected = question["expected_result"]
    tier = question.get("tier", 1)
    gold_sql = question.get("gold_sql", "")

    # Agent evaluation
    agent.reset_conversation()
    pred = agent.query(text)

    agent_exec_ok = pred["error"] is None
    agent_result_ok = (
        compare_results(pred["results"], expected) if agent_exec_ok else False
    )

    agent_entry = {
        "id": q_id,
        "tier": tier,
        "question": text,
        "gold_sql": gold_sql,
        "predicted_sql": pred["sql"],
        "exec_success": agent_exec_ok,
        "result_match": agent_result_ok,
        "error": pred.get("error"),
        "attempts": pred.get("attempts", 1),
    }

    # Baseline evaluation
    baseline_entry = None
    if baseline_client is not None:
        base = _run_baseline_query(text, baseline_client, agent.model)
        base_exec_ok = False
        base_result_ok = False
        base_error = None

        if base["sql"]:
            try:
                base_rows = query_db(conn, base["sql"], return_as_df=False)
                base_exec_ok = True
                base_result_ok = compare_results(base_rows, expected)
            except sqlite3.Error as e:
                base_error = str(e)
        else:
            base_error = "No SQL extracted"

        baseline_entry = {
            "id": q_id,
            "tier": tier,
            "predicted_sql": base["sql"],
            "exec_success": base_exec_ok,
            "result_match": base_result_ok,
            "error": base_error,
        }

    return {"agent": agent_entry, "baseline": baseline_entry}


def _print_summary(agent_rows: List[Dict], baseline_rows: Optional[List[Dict]]) -> None:
    n = len(agent_rows)
    a_exec = sum(r["exec_success"] for r in agent_rows) / n
    a_res = sum(r["result_match"] for r in agent_rows) / n

    has_baseline = bool(baseline_rows)
    b_exec: Optional[float] = None
    b_res: Optional[float] = None
    if baseline_rows:
        b_exec = sum(r["exec_success"] for r in baseline_rows) / n
        b_res = sum(r["result_match"] for r in baseline_rows) / n

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

    row("Execution Accuracy", a_exec, b_exec)
    row("Result Accuracy", a_res, b_res)

    print()
    print("  Per-Tier Result Accuracy (Agent):")
    for tier in (1, 2, 3):
        tier_rows = [r for r in agent_rows if r["tier"] == tier]
        if tier_rows:
            acc = sum(r["result_match"] for r in tier_rows) / len(tier_rows)
            bar = "█" * int(acc * 10) + "░" * (10 - int(acc * 10))
            print(f"    Tier {tier}  {bar}  {acc:.0%}  ({len(tier_rows)} questions)")

    failures = [r for r in agent_rows if not r["result_match"]]
    if failures:
        print()
        print("  Failure analysis (Agent):")
        for f in failures:
            reason = f"EXEC ERROR — {f['error']}" if not f["exec_success"] else "WRONG RESULT"
            print(f"    [{f['id']}] Tier {f['tier']}: {reason[:70]}")
    else:
        print()
        print("  No failures — 100% of questions answered correctly!")

    print("=" * width)
    print()


def run_eval(
    dataset: str = "dev",
    db_path: str = "data/Chinook.db",
    run_baseline: bool = True,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    conn = load_db(db_path)
    agent = TextToSQLAgent(conn)

    baseline_client = None
    if run_baseline:
        client_kwargs = {}
        if agent.api_key:
            client_kwargs["api_key"] = agent.api_key
        if agent.base_url:
            client_kwargs["base_url"] = agent.base_url
        baseline_client = OpenAI(**client_kwargs)

    questions: List[Dict] = []
    base_dir = Path(__file__).resolve().parent.parent

    if dataset in ("dev", "all"):
        dev_file = base_dir / "data" / "dev_questions_with_answers.json"
        if dev_file.exists():
            with open(dev_file) as fh:
                questions.extend(json.load(fh))

    if dataset in ("extended", "all"):
        ext_file = base_dir / "data" / "extended_questions_with_answers.json"
        if ext_file.exists():
            with open(ext_file) as fh:
                questions.extend(json.load(fh))

    if verbose:
        print(f"\nRunning evaluation on {len(questions)} questions (dataset={dataset})")
        print(f"  Agent model : {agent.model}")
        print(f"  Baseline    : {'enabled' if run_baseline else 'disabled'}\n")

    agent_rows: List[Dict] = []
    baseline_rows: List[Dict] = []

    for q in questions:
        if verbose:
            tier_label = f"[Tier {q.get('tier', 1)}]"
            print(f"  {q['id']} {tier_label:<8} {q['question'][:55]}...")

        one = _eval_one(q, agent, baseline_client, conn)
        agent_rows.append(one["agent"])

        if verbose:
            a_sym = "✓" if one["agent"]["result_match"] else ("⚡" if not one["agent"]["exec_success"] else "✗")
            print(f"           Agent: {a_sym}", end="")

        if one["baseline"] is not None:
            baseline_rows.append(one["baseline"])
            if verbose:
                b_sym = "✓" if one["baseline"]["result_match"] else ("⚡" if not one["baseline"]["exec_success"] else "✗")
                print(f"  Baseline: {b_sym}", end="")

        if verbose:
            print()

    n = len(questions)
    metrics = {
        "dataset": dataset,
        "total_questions": n,
        "agent": {
            "execution_accuracy": sum(r["exec_success"] for r in agent_rows) / n if n > 0 else 0,
            "result_accuracy": sum(r["result_match"] for r in agent_rows) / n if n > 0 else 0,
            "details": agent_rows,
        },
        "baseline": {
            "execution_accuracy": sum(r["exec_success"] for r in baseline_rows) / n if n > 0 else 0,
            "result_accuracy": sum(r["result_match"] for r in baseline_rows) / n if n > 0 else 0,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run text-to-SQL evaluation")
    parser.add_argument(
        "--dataset",
        choices=["dev", "extended", "all"],
        default="dev",
        help="Dataset to evaluate against (dev, extended, all)",
    )
    parser.add_argument(
        "--db",
        default="data/Chinook.db",
        help="Path to SQLite database file",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip baseline evaluation",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help="Save detailed results JSON to FILE",
    )
    args = parser.parse_args()

    run_eval(
        dataset=args.dataset,
        db_path=args.db,
        run_baseline=not args.no_baseline,
        output_path=args.output,
    )

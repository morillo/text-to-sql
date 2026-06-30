"""Performance measurement: latency benchmarking and cost analysis.

What this module measures
─────────────────────────
  Latency   End-to-end wall-clock time from the user hitting Enter to the
            agent returning a result (includes network round-trip + execution).
            We report P50 (median), P95, and mean across multiple runs.

  Cost      Estimated daily and monthly API spend at a hypothetical scale:
            1,000 users × 30 queries/day = 30,000 queries/day.
            Compared against a baseline model at standard market pricing.

Pricing assumptions
───────────────────
  Current Model (configured via env):
    Input:  Based on LLM_INPUT_PRICE_PER_1M
    Output: Based on LLM_OUTPUT_PRICE_PER_1M

  Baseline Model (configured via env):
    Input:  Based on BASELINE_INPUT_PRICE_PER_1M
    Output: Based on BASELINE_OUTPUT_PRICE_PER_1M

Usage
─────
  python -m src.perf                     # benchmark with default settings
  python -m src.perf --runs 5            # 5 runs per question (more stable)
  python -m src.perf --output perf.json  # save results to file
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import TextToSQLAgent
from src.utils import load_db


# ─── Pricing constants ────────────────────────────────────────────────────────
# All prices are in USD per 1,000,000 tokens.
# Default values can be overridden via environment variables.

# Current model pricing (default to $0.90/1M, like DeepSeek V4)
INPUT_PRICE_PER_1M  = float(os.environ.get("LLM_INPUT_PRICE_PER_1M", "0.90"))
OUTPUT_PRICE_PER_1M = float(os.environ.get("LLM_OUTPUT_PRICE_PER_1M", "0.90"))

# Baseline model pricing for comparison (default to $2.00/$8.00 / 1M, like GPT-4)
BASELINE_INPUT_PRICE_PER_1M  = float(os.environ.get("BASELINE_INPUT_PRICE_PER_1M", "2.00"))
BASELINE_OUTPUT_PRICE_PER_1M = float(os.environ.get("BASELINE_OUTPUT_PRICE_PER_1M", "8.00"))

# Stated scale
DAILY_QUERIES = 30_000   # 1,000 users × 30 queries/day


# ─── Benchmark runner ─────────────────────────────────────────────────────────

def benchmark(
    n_runs: int = 3,
    questions_path: str = "data/dev_questions.json",
    verbose: bool = True,
) -> Dict:
    """
    Benchmark the agent on latency and token usage.

    For each of the dev questions we run the agent n_runs times and record:
      - wall-clock time (start of agent.query() → result returned)
      - input/output token counts from the API response

    We reset conversation history between each run so we're measuring
    cold-start (fresh-question) latency, which is the common case.

    Args:
        n_runs:         Number of repetitions per question.
        questions_path: Path to dev_questions.json.
        verbose:        If True, print per-run progress.

    Returns:
        A dict with "latency", "tokens", and "cost" sub-dicts.
    """
    import dotenv
    dotenv.load_dotenv()

    conn  = load_db()
    agent = TextToSQLAgent(conn)

    with open(questions_path) as fh:
        questions = json.load(fh)

    if verbose:
        total = n_runs * len(questions)
        print(f"\nLatency benchmark: {n_runs} run(s) × {len(questions)} questions = {total} API calls")
        print(f"  Model: {agent.model}\n")

    latencies:     List[float] = []
    input_tokens:  List[int]   = []
    output_tokens: List[int]   = []

    for run_idx in range(n_runs):
        for q in questions:
            agent.reset_conversation()

            t0      = time.perf_counter()
            result  = agent.query(q["question"])
            elapsed = time.perf_counter() - t0

            latencies.append(elapsed)

            usage = result.get("usage", {})
            if usage:
                input_tokens.append(usage.get("prompt_tokens", 0))
                output_tokens.append(usage.get("completion_tokens", 0))

            if verbose:
                status = "✓" if not result["error"] else "✗"
                attempts_note = f" (retry×{result['attempts']-1})" if result["attempts"] > 1 else ""
                print(f"  run {run_idx+1}  {q['id']}  {elapsed:.2f}s  {status}{attempts_note}")

    # ── Latency statistics ────────────────────────────────────────────────────
    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)
    p50  = statistics.median(latencies)
    p95  = latencies_sorted[min(int(n * 0.95), n - 1)]
    mean = statistics.mean(latencies)

    # ── Token averages ────────────────────────────────────────────────────────
    avg_in  = statistics.mean(input_tokens)  if input_tokens  else 0.0
    avg_out = statistics.mean(output_tokens) if output_tokens else 0.0
    avg_tot = avg_in + avg_out

    # ── Cost per query ────────────────────────────────────────────────────────
    curr_cost_per_query = (
        (avg_in  / 1_000_000) * INPUT_PRICE_PER_1M
        + (avg_out / 1_000_000) * OUTPUT_PRICE_PER_1M
    )
    base_cost_per_query = (
        (avg_in  / 1_000_000) * BASELINE_INPUT_PRICE_PER_1M
        + (avg_out / 1_000_000) * BASELINE_OUTPUT_PRICE_PER_1M
    )

    # ── Daily / monthly projections ──────────────────────────────────────────
    curr_daily    = curr_cost_per_query    * DAILY_QUERIES
    base_daily = base_cost_per_query * DAILY_QUERIES

    curr_monthly    = curr_daily    * 30
    base_monthly = base_daily * 30

    savings_daily   = base_daily   - curr_daily
    savings_monthly = base_monthly - curr_monthly
    cost_ratio      = (base_cost_per_query / curr_cost_per_query) if curr_cost_per_query > 0 else 0

    current_model_dict = {
        "model":                  agent.model,
        "input_price_per_1m":     INPUT_PRICE_PER_1M,
        "output_price_per_1m":    OUTPUT_PRICE_PER_1M,
        "cost_per_query_usd":     round(curr_cost_per_query,    6),
        "daily_cost_usd":         round(curr_daily,             2),
        "monthly_cost_usd":       round(curr_monthly,           2),
    }

    baseline_model_dict = {
        "model":                  os.environ.get("BASELINE_MODEL_NAME", "Baseline Model (proxy)"),
        "input_price_per_1m":     BASELINE_INPUT_PRICE_PER_1M,
        "output_price_per_1m":    BASELINE_OUTPUT_PRICE_PER_1M,
        "cost_per_query_usd":     round(base_cost_per_query, 6),
        "daily_cost_usd":         round(base_daily,          2),
        "monthly_cost_usd":       round(base_monthly,        2),
    }

    report = {
        "latency": {
            "p50_s":           round(p50,  3),
            "p95_s":           round(p95,  3),
            "mean_s":          round(mean, 3),
            "meets_3s_target": p50 < 3.0,
            "samples":         n,
        },
        "tokens": {
            "avg_input_tokens":  round(avg_in),
            "avg_output_tokens": round(avg_out),
            "avg_total_tokens":  round(avg_tot),
        },
        "cost": {
            "queries_per_day":    DAILY_QUERIES,
            "current_model":       current_model_dict,
            "baseline_model":      baseline_model_dict,
            "cost_reduction_ratio_x":  round(cost_ratio,        1),
            "daily_savings_usd":       round(savings_daily,      2),
            "monthly_savings_usd":     round(savings_monthly,    2),
        },
    }

    if verbose:
        _print_report(report)

    return report


# ─── Reporting ────────────────────────────────────────────────────────────────

def _print_report(report: Dict) -> None:
    lat  = report["latency"]
    tok  = report["tokens"]
    cost = report["cost"]
    curr = cost["current_model"]
    base = cost["baseline_model"]

    target_flag = "✓ MEETS <3 s target" if lat["meets_3s_target"] else "✗ MISSES <3 s target"

    print()
    print("=" * 62)
    print("PERFORMANCE REPORT")
    print("=" * 62)

    print(f"\nLatency  ({lat['samples']} samples across all runs)")
    print(f"  P50 (median) : {lat['p50_s']:.3f} s   {target_flag}")
    print(f"  P95          : {lat['p95_s']:.3f} s")
    print(f"  Mean         : {lat['mean_s']:.3f} s")
    print(f"  Hypothetical baseline (reported) : ~7.0 s")

    print(f"\nToken usage  (avg per query)")
    print(f"  Input  : {tok['avg_input_tokens']:,} tokens")
    print(f"  Output : {tok['avg_output_tokens']:,} tokens")
    print(f"  Total  : {tok['avg_total_tokens']:,} tokens")

    print(f"\nCost analysis  ({cost['queries_per_day']:,} queries/day = 1 k users × 30 q/day)")
    
    # Format labels to fit formatting columns nicely
    curr_model_short = curr['model'][:16]
    base_model_short = base['model'][:16]
    
    print(f"  {'':32}  {curr_model_short:>15}  {base_model_short:>18}")
    print(f"  {'─'*68}")
    print(f"  {'Model':32}  {curr_model_short:>15}  {base_model_short:>18}")
    print(f"  {'Input $/1M tokens':32}  ${curr['input_price_per_1m']:>14.2f}  ${base['input_price_per_1m']:>17.2f}")
    print(f"  {'Output $/1M tokens':32}  ${curr['output_price_per_1m']:>14.2f}  ${base['output_price_per_1m']:>17.2f}")
    print(f"  {'Cost per query':32}  ${curr['cost_per_query_usd']:>14.4f}  ${base['cost_per_query_usd']:>17.4f}")
    print(f"  {'Daily cost':32}  ${curr['daily_cost_usd']:>14.2f}  ${base['daily_cost_usd']:>17.2f}")
    print(f"  {'Monthly cost (30 days)':32}  ${curr['monthly_cost_usd']:>14.2f}  ${base['monthly_cost_usd']:>17.2f}")
    print()
    print(f"  Cost reduction : {cost['cost_reduction_ratio_x']:.1f}× cheaper with Current Model")
    print(f"  Daily savings  : ${cost['daily_savings_usd']:,.2f}")
    print(f"  Monthly savings: ${cost['monthly_savings_usd']:,.2f}")
    print("=" * 62)
    print()


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark agent latency and cost")
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        metavar="N",
        help="Number of repetitions per question (default: 3)",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help="Write JSON report to FILE",
    )
    args = parser.parse_args()

    results = benchmark(n_runs=args.runs)

    if args.output:
        with open(args.output, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"Report saved to {args.output}\n")

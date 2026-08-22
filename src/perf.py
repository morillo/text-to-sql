"""Performance measurement: latency benchmarking and token consumption analysis.

What this module measures
─────────────────────────
  Latency   End-to-end wall-clock time from user prompt to agent result.
            Reports P50 (median), P95, and mean across repetitions.

  Tokens    Average input, output, and total token usage per query.

Usage
─────
  python -m src.perf
  python -m src.perf --runs 5
  python -m src.perf --output perf_report.json
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

from src.agent import TextToSQLAgent, DEFAULT_MODEL
from src.utils import load_db


def benchmark(
    n_runs: int = 3,
    questions_path: str = "data/dev_questions.json",
    verbose: bool = True,
) -> Dict:
    conn = load_db()
    agent = TextToSQLAgent(conn)

    base_dir = Path(__file__).resolve().parent.parent
    full_path = base_dir / questions_path if not Path(questions_path).is_absolute() else Path(questions_path)

    with open(full_path) as fh:
        questions = json.load(fh)

    if verbose:
        total = n_runs * len(questions)
        print(f"\nLatency Benchmark: {n_runs} run(s) × {len(questions)} questions = {total} API calls")
        print(f"  Model: {agent.model}\n")

    latencies: List[float] = []
    input_tokens: List[int] = []
    output_tokens: List[int] = []

    for run_idx in range(n_runs):
        for q in questions:
            agent.reset_conversation()

            t0 = time.perf_counter()
            result = agent.query(q["question"])
            elapsed = time.perf_counter() - t0

            latencies.append(elapsed)

            usage = result.get("usage", {})
            if usage:
                input_tokens.append(usage.get("prompt_tokens", 0))
                output_tokens.append(usage.get("completion_tokens", 0))

            if verbose:
                status = "✓" if not result["error"] else "✗"
                retry_note = f" (retry×{result['attempts']-1})" if result["attempts"] > 1 else ""
                print(f"  run {run_idx+1}  {q['id']}  {elapsed:.2f}s  {status}{retry_note}")

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)
    p50 = statistics.median(latencies)
    p95 = latencies_sorted[min(int(n * 0.95), n - 1)]
    mean = statistics.mean(latencies)

    avg_in = statistics.mean(input_tokens) if input_tokens else 0.0
    avg_out = statistics.mean(output_tokens) if output_tokens else 0.0
    avg_tot = avg_in + avg_out

    report = {
        "latency": {
            "p50_s": round(p50, 3),
            "p95_s": round(p95, 3),
            "mean_s": round(mean, 3),
            "samples": n,
        },
        "tokens": {
            "avg_input_tokens": round(avg_in),
            "avg_output_tokens": round(avg_out),
            "avg_total_tokens": round(avg_tot),
        },
        "model": agent.model,
    }

    if verbose:
        _print_report(report)

    return report


def _print_report(report: Dict) -> None:
    lat = report["latency"]
    tok = report["tokens"]

    print()
    print("=" * 62)
    print("PERFORMANCE BENCHMARK REPORT")
    print("=" * 62)
    print(f"  Model        : {report['model']}")

    print(f"\nLatency ({lat['samples']} samples)")
    print(f"  P50 (median) : {lat['p50_s']:.3f} s")
    print(f"  P95          : {lat['p95_s']:.3f} s")
    print(f"  Mean         : {lat['mean_s']:.3f} s")

    print(f"\nToken Usage (avg per query)")
    print(f"  Input        : {tok['avg_input_tokens']:,} tokens")
    print(f"  Output       : {tok['avg_output_tokens']:,} tokens")
    print(f"  Total        : {tok['avg_total_tokens']:,} tokens")
    print("=" * 62)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark agent latency and token usage")
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
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

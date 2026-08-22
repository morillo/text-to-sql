"""Interactive Terminal CLI for Model-Agnostic Text-to-SQL Agent.

Run:
    python -m src.cli
    python -m src.cli --db custom.db
    python -m src.cli --model gpt-4o-mini
    python -m src.cli --no-stream

Features:
  - Query any SQLite database using natural language from the terminal.
  - Works with OpenAI, Google Gemini, DeepSeek, local Ollama, or custom endpoints.
  - Real-time token streaming for perceived latency < 1.0 second.
  - Formatted ASCII table result output.
  - Export query results to CSV (`export <filename.csv>`).
  - View database schema (`schema`).
  - Multi-turn conversation memory (`reset` to clear).
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import TextToSQLAgent, DEFAULT_MODEL
from src.utils import load_db, print_table_schema


def _format_results_table(results: List[Dict[str, Any]]) -> str:
    """Render row dicts as a clean ASCII table."""
    if not results:
        return "  (no rows returned)"

    cols = list(results[0].keys())
    widths = {col: len(col) for col in cols}
    for row in results:
        for col in cols:
            widths[col] = max(widths[col], len(str(row.get(col, ""))))

    sep = "  " + "  ".join("-" * widths[c] for c in cols)
    hdr = "  " + "  ".join(col.ljust(widths[col]) for col in cols)
    lines = [sep, hdr, sep]

    for row in results:
        line = "  " + "  ".join(str(row.get(col, "")).ljust(widths[col]) for col in cols)
        lines.append(line)

    lines.append(sep)
    n = len(results)
    lines.append(f"  ({n} row{'s' if n != 1 else ''})")
    return "\n".join(lines)


def _print_banner(db_path: str, model: str, streaming: bool) -> None:
    print()
    print("=" * 66)
    print("  Model-Agnostic  ·  Text-to-SQL CLI Agent")
    print(f"  Model    : {model}")
    print(f"  Database : {db_path}")
    print(f"  Mode     : {'Real-Time Token Streaming' if streaming else 'Standard Response'}")
    print("=" * 66)
    print("  Commands:")
    print("    <question>        Ask natural language questions")
    print("    schema            View database schema")
    print("    export <file.csv> Export last query results to CSV")
    print("    reset             Clear conversation history")
    print("    exit / quit       End session")
    print("=" * 66)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Model-Agnostic Text-to-SQL CLI Agent")
    parser.add_argument(
        "--db",
        default="data/Chinook.db",
        help="Path to SQLite database file (default: data/Chinook.db)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"LLM model identifier string (default: from .env or {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable real-time token streaming",
    )
    args = parser.parse_args()

    try:
        conn = load_db(args.db)
    except Exception as exc:
        print(f"\nError initializing database connection: {exc}\n")
        sys.exit(1)

    print("\nLoading database schema and initializing agent...", end="", flush=True)
    try:
        agent = TextToSQLAgent(conn, model=args.model)
        print(" ready.\n")
    except Exception as exc:
        print(f"\nError initializing agent: {exc}\n")
        sys.exit(1)

    use_stream = not args.no_stream
    _print_banner(args.db, agent.model, use_stream)

    last_results: List[Dict[str, Any]] = []

    while True:
        try:
            user_input = input("Ask a question > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ("exit", "quit"):
            print("Goodbye!")
            break

        if cmd == "reset":
            agent.reset_conversation()
            last_results = []
            print("\nConversation history cleared. Starting fresh.\n")
            continue

        if cmd == "schema" or cmd.startswith("schema "):
            parts = user_input.split(maxsplit=1)
            tbl = parts[1] if len(parts) > 1 else None
            print_table_schema(conn, table_name=tbl)
            continue

        if cmd.startswith("export"):
            parts = user_input.split(maxsplit=1)
            filename = parts[1] if len(parts) > 1 else "query_results.csv"
            if not last_results:
                print("  No active query results to export.\n")
                continue
            try:
                with open(filename, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(last_results[0].keys()))
                    writer.writeheader()
                    writer.writerows(last_results)
                print(f"  Successfully exported {len(last_results)} rows to {filename}\n")
            except Exception as e:
                print(f"  Error exporting to CSV: {e}\n")
            continue

        print("\nThinking...", end="", flush=True)
        t0 = time.perf_counter()

        if use_stream:
            print("\n")
            result = agent.query_stream(user_input)
            elapsed = time.perf_counter() - t0
        else:
            result = agent.query(user_input)
            elapsed = time.perf_counter() - t0
            print(" done.")

        if result["error"]:
            print(
                f"\n  Could not generate valid SQL after {result['attempts']} attempts.\n"
                f"  Error: {result['error']}\n"
                "  Please try rephrasing your question.\n"
            )
            continue

        if not use_stream:
            print("\nSQL Query:")
            print("  " + "-" * 58)
            for line in result["sql"].splitlines():
                print(f"  {line}")
            print("  " + "-" * 58)

        print("\nResults:")
        last_results = result["results"]
        print(_format_results_table(last_results))
        print()

        if result["explanation"]:
            print(f"  Explanation: {result['explanation']}")

        print(f"  (Latency: {elapsed:.2f}s | Attempts: {result['attempts']})\n")


if __name__ == "__main__":
    main()

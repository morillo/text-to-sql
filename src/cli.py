"""CLI entry point.

Run with:
    python -m src.cli

The CLI launches an interactive terminal session where you can:
  - Type a natural language question and get back the generated SQL + results
  - Ask follow-up questions to refine or explore further (context is preserved)
  - Type 'reset'  to clear conversation history and start fresh
  - Type 'schema' to print the full database schema
  - Type 'exit' or 'quit' (or press Ctrl-C) to end the session

Environment variables required:
    LLM_API_KEY         Your LLM provider API key
"""

import os
import sys
from pathlib import Path

# Allow running as  python -m src.cli  from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import TextToSQLAgent
from src.utils import load_db, print_table_schema


# ─── Display helpers ──────────────────────────────────────────────────────────

def _format_results_table(results: list) -> str:
    """
    Render a list-of-dicts as a fixed-width ASCII table.

    We avoid third-party libraries (tabulate, rich) so the CLI works
    with only the dependencies already in pyproject.toml.

    Args:
        results: List of row dicts returned by query_db().

    Returns:
        A multi-line string ready for print().
    """
    if not results:
        return "  (no rows returned)"

    cols = list(results[0].keys())

    # Compute the width needed for each column (header or widest value)
    widths = {col: len(col) for col in cols}
    for row in results:
        for col in cols:
            widths[col] = max(widths[col], len(str(row.get(col, ""))))

    sep   = "  " + "  ".join("-" * widths[c] for c in cols)
    hdr   = "  " + "  ".join(col.ljust(widths[col]) for col in cols)
    lines = [sep, hdr, sep]

    for row in results:
        line = "  " + "  ".join(str(row.get(col, "")).ljust(widths[col]) for col in cols)
        lines.append(line)

    lines.append(sep)
    n = len(results)
    lines.append(f"  ({n} row{'s' if n != 1 else ''})")
    return "\n".join(lines)


def _print_banner(agent: TextToSQLAgent) -> None:
    provider_name = "Gemini" if "generativelanguage" in getattr(agent, "base_url", "") or "gemini" in getattr(agent, "provider", "") else ("OpenAI-Compatible" if getattr(agent, "base_url", "") else "OpenAI")
    print()
    print("=" * 62)
    print("  Text-to-SQL CLI")
    print(f"  Model: {agent.model} ({provider_name})")
    print("  Database: Chinook Music Store (SQLite)")
    print("=" * 62)
    print("  Commands:")
    print("    <question>  Ask anything about the database")
    print("    reset       Clear conversation history")
    print("    schema      Show full database schema")
    print("    exit/quit   End the session")
    print("=" * 62)
    print()


# ─── Main loop ────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point — called by  python -m src.cli."""
    import dotenv
    dotenv.load_dotenv()

    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(
            "\nError: No API key set.\n"
            "Please create a .env file or set one of the following environment variables:\n"
            "  - LLM_API_KEY\n"
            "  - OPENAI_API_KEY\n"
            "  - GEMINI_API_KEY\n"
        )
        sys.exit(1)

    # ── Load the database ─────────────────────────────────────────────────────
    try:
        conn = load_db()
    except FileNotFoundError as exc:
        print(f"\nError: {exc}\n")
        sys.exit(1)

    # ── Initialise the agent (schema is loaded once here) ────────────────────
    print("\nLoading schema and initialising agent...", end="", flush=True)
    agent = TextToSQLAgent(conn)
    print(" ready.\n")

    _print_banner(agent)

    # ── Interactive loop ──────────────────────────────────────────────────────
    while True:
        try:
            user_input = input("Ask a question > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        # ── Built-in commands ────────────────────────────────────────────────
        if cmd in ("exit", "quit"):
            print("Goodbye!")
            break

        if cmd == "reset":
            agent.reset_conversation()
            print("\nConversation history cleared. Starting fresh.\n")
            continue

        if cmd == "schema":
            print_table_schema(conn)
            continue

        # ── Generate SQL and execute ─────────────────────────────────────────
        print("\nThinking...", end="", flush=True)
        result = agent.query(user_input)
        print(" done.\n")

        # ── Error path ────────────────────────────────────────────────────────
        if result["error"]:
            attempts = result["attempts"]
            print(
                f"  Could not produce valid SQL after {attempts} attempt(s).\n"
                f"  Reason: {result['error']}\n"
                "  Please try rephrasing your question.\n"
            )
            continue

        # ── Happy path ────────────────────────────────────────────────────────
        # Print the generated SQL
        print("SQL Query:")
        print("  " + "-" * 58)
        # Indent each line of the SQL for readability
        for line in result["sql"].splitlines():
            print(f"  {line}")
        print("  " + "-" * 58)
        print()

        # Print results as a table
        print("Results:")
        print(_format_results_table(result["results"]))
        print()

        # Print the one-sentence explanation
        if result["explanation"]:
            print(f"  {result['explanation']}")
            print()

        # Transparency: note if the agent had to self-correct
        if result["attempts"] > 1:
            print(
                f"  (Note: required {result['attempts']} attempts — "
                "first SQL failed, agent self-corrected.)\n"
            )


if __name__ == "__main__":
    main()

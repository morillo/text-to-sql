"""
Generate dev_answers.json by running the agent against all 10 dev questions.

Usage:
    python scripts/generate_answers.py

Writes dev_answers.json to the project root.
The format matches dev_answers_example.json exactly.
"""

import json
import os
import sys
from pathlib import Path

# Allow running from scripts/ or from project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.agent import TextToSQLAgent
from src.utils import load_db


def _fmt(v) -> str:
    """Format a cell value for human-readable output.

    Floats are rounded to 2 decimal places so the answer string matches the
    gold_answer format (e.g. 6.66 rather than 6.659999999999999).
    Integers and strings are returned as-is.
    """
    if isinstance(v, float):
        return f"{round(v, 2)}"
    return str(v)


def format_answer_string(results: list) -> str:
    """
    Convert a list-of-dicts result set into a concise human-readable string.

    For single-row, single-column results this returns just the value.
    For multi-row results it builds a comma-separated summary of the key values.
    For multi-column results it formats each row as  Col1 (Col2)  pairs.
    """
    if not results:
        return "No results"

    cols = list(results[0].keys())

    # Single value
    if len(cols) == 1 and len(results) == 1:
        return _fmt(list(results[0].values())[0])

    # Multiple rows, multiple columns: build "Val1 (Val2), Val1 (Val2)" style
    row_strings = []
    for row in results:
        vals = [_fmt(v) for v in row.values()]
        if len(vals) == 1:
            row_strings.append(vals[0])
        elif len(vals) == 2:
            row_strings.append(f"{vals[0]} ({vals[1]})")
        else:
            # First value as label, rest in parentheses
            row_strings.append(f"{vals[0]} ({', '.join(vals[1:])})")

    return ", ".join(row_strings)


def main():
    import dotenv
    dotenv.load_dotenv()

    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(
            "Error: No API key set.\n"
            "Please create a .env file or set one of the following environment variables:\n"
            "  - LLM_API_KEY\n"
            "  - OPENAI_API_KEY\n"
            "  - GEMINI_API_KEY\n"
        )
        sys.exit(1)

    questions_path = project_root / "data" / "dev_questions.json"
    output_path    = project_root / "dev_answers.json"

    with open(questions_path) as fh:
        questions = json.load(fh)

    conn  = load_db(str(project_root / "data" / "Chinook.db"))
    agent = TextToSQLAgent(conn)

    answers = {}
    print(f"\nGenerating answers for {len(questions)} questions...\n")

    for q in questions:
        q_id     = q["id"]
        question = q["question"]

        print(f"  {q_id}: {question}")
        agent.reset_conversation()
        result = agent.query(question)

        if result["error"]:
            print(f"    ✗ Error: {result['error']}")
            answers[q_id] = {
                "sql":    result["sql"] or "ERROR: could not generate SQL",
                "answer": f"Error: {result['error']}",
            }
        else:
            answer_str = format_answer_string(result["results"])
            print(f"    ✓ {len(result['results'])} row(s) — {answer_str[:80]}")
            answers[q_id] = {
                "sql":    result["sql"],
                "answer": answer_str,
            }

    with open(output_path, "w") as fh:
        json.dump(answers, fh, indent=2, ensure_ascii=False)

    print(f"\nWrote {output_path}\n")

    # Quick sanity check against gold answers
    gold_path = project_root / "data" / "dev_questions_with_answers.json"
    if gold_path.exists():
        with open(gold_path) as fh:
            gold = {q["id"]: q for q in json.load(fh)}

        print("Quick sanity check against gold answers:")
        for q_id, entry in answers.items():
            gold_entry = gold.get(q_id, {})
            gold_answer = gold_entry.get("gold_answer", "")
            status = "✓" if entry["answer"] != "No results" and "Error" not in entry["answer"] else "?"
            print(f"  {q_id} {status}  gold: {gold_answer[:60]}")
        print()


if __name__ == "__main__":
    main()

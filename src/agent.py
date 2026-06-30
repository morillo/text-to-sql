"""Agent logic for text-to-SQL conversion.

Architecture overview
─────────────────────
TextToSQLAgent converts natural language questions into SQLite SQL queries
and executes them against the database.

Key design decisions
────────────────────
1. **Full-schema injection** – The entire DDL (CREATE TABLE statements) plus
   two sample rows per table are injected into the system prompt at init time.
   This eliminates hallucinated table/column names, which is the #1 failure
   mode of a schema-less prompt.

2. **Structured JSON output** – The model is instructed to return a JSON
   object with keys "sql" and "explanation".  This makes parsing reliable and
   eliminates fragile regex-based SQL extraction from free-form prose.

3. **Three few-shot examples** – The system prompt includes one example per
   difficulty tier: a multi-table JOIN/aggregate (Tier 1), a date-filter with
   SQLite's strftime() (Tier 2), and a window-function ranking query (Tier 3).
   Few-shot examples are the single highest-leverage prompt improvement.

4. **Automatic retry loop** – If the generated SQL fails to execute, the
   agent re-prompts with the error message (up to MAX_RETRIES attempts).
   This handles transient hallucinations without surfacing them to the user.

5. **Multi-turn conversation history** – The messages list is preserved across
   calls so users can ask follow-up questions ("show only the top 3", "break
   it down by country") without re-explaining context.
"""

import json
import os
import re
import sqlite3
from typing import Any, Dict, List, Optional

from openai import OpenAI

from src.utils import get_schema_string, query_db

# ─── Model ────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "gpt-4o"
MAX_RETRIES = 2  # number of self-correction attempts on SQL execution error

# ─── System prompt template ───────────────────────────────────────────────────
# {schema} is replaced at init time with the live DDL + sample rows.

_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert SQLite SQL query generator for a digital music store database.

## CRITICAL RULES
1. Use ONLY table and column names that appear in the schema below.
2. Write valid SQLite SQL only (not MySQL, PostgreSQL, or any other dialect).
3. Use strftime('%Y', <date_column>) to extract a year in SQLite — never YEAR().
4. Use window functions (e.g. RANK() OVER (ORDER BY ...)) when the question asks for rankings.
5. Respond with EXACTLY one JSON object and nothing else — no markdown fences, no extra text:
   {{"sql": "<your SQL here>", "explanation": "<one sentence plain-English summary>"}}

## DATABASE SCHEMA
{schema}

## EXAMPLES

### Example 1 — multi-table JOIN with aggregation and LIMIT
Question: What are the top 3 genres by total revenue?
{{"sql": "SELECT g.Name, SUM(il.UnitPrice * il.Quantity) AS TotalRevenue FROM Genre g JOIN Track t ON g.GenreId = t.GenreId JOIN InvoiceLine il ON t.TrackId = il.TrackId GROUP BY g.GenreId, g.Name ORDER BY TotalRevenue DESC LIMIT 3", "explanation": "Joins Genre → Track → InvoiceLine to compute revenue per genre, then returns the top 3 by revenue."}}

### Example 2 — date filtering with SQLite's strftime
Question: What is the total revenue for the year 2020?
{{"sql": "SELECT SUM(Total) AS TotalRevenue FROM Invoice WHERE strftime('%Y', InvoiceDate) = '2020'", "explanation": "Filters invoices to 2020 using strftime and sums the Total column."}}

### Example 3 — window function (RANK) with GROUP BY
Question: Show the top 5 customers by total spending with their rank.
{{"sql": "SELECT c.FirstName || ' ' || c.LastName AS CustomerName, SUM(i.Total) AS TotalSpent, RANK() OVER (ORDER BY SUM(i.Total) DESC) AS Rank FROM Customer c JOIN Invoice i ON c.CustomerId = i.CustomerId GROUP BY c.CustomerId ORDER BY TotalSpent DESC LIMIT 5", "explanation": "Aggregates invoice totals per customer and uses RANK() to rank them by spending."}}
"""


# ─── Response parser ──────────────────────────────────────────────────────────

def parse_agent_response(text: str) -> Dict[str, str]:
    """
    Parse the model's raw response into {"sql": ..., "explanation": ...}.

    Attempts in order:
      1. Direct JSON.loads() — the happy path when the model follows instructions.
      2. Regex extraction of the first {...} block — handles stray leading text.
      3. SQL extraction from plain text — last-resort fallback.

    Args:
        text: Raw string from the model's completion.

    Returns:
        dict with keys "sql" and "explanation".

    Raises:
        ValueError: If none of the extraction strategies succeed.
    """
    text = text.strip()

    # Strip any accidental markdown fences wrapping the JSON
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Strategy 1 — direct JSON parse
    try:
        parsed = json.loads(text)
        return {
            "sql": parsed.get("sql", "").strip(),
            "explanation": parsed.get("explanation", "").strip(),
        }
    except (json.JSONDecodeError, AttributeError):
        pass

    # Strategy 2 — extract the first {...} block (handles stray preamble text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group())
            return {
                "sql": parsed.get("sql", "").strip(),
                "explanation": parsed.get("explanation", "").strip(),
            }
        except json.JSONDecodeError:
            pass

    # Strategy 3 — pull a bare SELECT/WITH statement out of free-form text
    m = re.search(r"((?:SELECT|WITH)\s+.+?)(?:;|\Z)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return {"sql": m.group(1).strip(), "explanation": ""}

    raise ValueError(f"Could not parse model response (first 300 chars): {text[:300]}")


# ─── Agent class ──────────────────────────────────────────────────────────────

class TextToSQLAgent:
    """
    Converts natural language questions to SQL and executes them.

    Usage
    ─────
        conn = load_db()
        agent = TextToSQLAgent(conn)
        result = agent.query("What are the top 5 genres by revenue?")
        print(result["sql"])
        print(result["results"])

    The agent keeps a running conversation_history so follow-up questions
    ("show only the top 3", "filter to year 2021") work naturally.
    Call agent.reset_conversation() to start a fresh session.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        model: Optional[str] = None,
    ) -> None:
        """
        Initialise the agent.

        Args:
            conn:  Open SQLite connection to the database.
            model: Model identifier string. If None, loads from env.
        """
        import dotenv
        dotenv.load_dotenv()

        self.conn = conn

        # Load environment variables
        self.provider = os.environ.get("LLM_PROVIDER", "").lower()
        self.api_key = os.environ.get("LLM_API_KEY")
        self.base_url = os.environ.get("LLM_BASE_URL")
        self.model = model or os.environ.get("LLM_MODEL")

        # Fallback to legacy environment variables
        if not self.api_key:
            self.api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        
        if not self.model:
            self.model = os.environ.get("LLM_MODEL", DEFAULT_MODEL)

        # Auto-configure base URL for Gemini if key is provided and base URL is empty
        if self.api_key and self.api_key.startswith("AIzaSy") and not self.base_url:
            self.base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
            if not model and not os.environ.get("LLM_MODEL"):
                self.model = "gemini-2.5-flash"
            self.provider = "gemini"

        # Support provider specific auto-configurations
        if self.provider == "gemini" and not self.base_url:
            self.base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

        # Setup standard OpenAI-compatible client
        client_kwargs = {}
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self.client = OpenAI(**client_kwargs)

        # Build schema string once at startup — injected into every request.
        self.schema_string = get_schema_string(conn)

        # Build the full system prompt (schema is baked in at init time).
        self.system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            schema=self.schema_string
        )

        # Multi-turn conversation history (OpenAI message format).
        # The system prompt is NOT stored here — it is prepended on every call.
        self.conversation_history: List[Dict[str, str]] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def reset_conversation(self) -> None:
        """Clear conversation history to start a fresh session."""
        self.conversation_history = []

    def query(
        self,
        user_question: str,
        max_retries: int = MAX_RETRIES,
    ) -> Dict[str, Any]:
        """
        Convert a natural language question to SQL and execute it.

        The method appends the user question to the running conversation
        history, calls the LLM API, extracts and validates the SQL,
        and automatically retries up to max_retries times if execution fails.

        Args:
            user_question: The user's natural language question.
            max_retries:   How many self-correction attempts to allow.

        Returns:
            dict with keys:
              sql         – the generated SQL string
              results     – list of row dicts (empty list on error)
              explanation – one-sentence plain-English summary
              error       – None on success, error string on failure
              usage       – token counts dict from the API response
              attempts    – number of API calls made (1 = first try succeeded)
        """
        # Append the user message to conversation history
        self.conversation_history.append(
            {"role": "user", "content": user_question}
        )

        # Build the messages list: system prompt + full conversation history.
        # Keeping the full history enables coherent multi-turn follow-ups.
        messages = [
            {"role": "system", "content": self.system_prompt}
        ] + self.conversation_history

        last_error: Optional[str] = None
        sql: Optional[str] = None
        usage: Dict = {}

        for attempt in range(max_retries + 1):

            # On retries, extend the messages with the previous (bad) SQL and
            # the execution error so the model can self-correct.
            if attempt > 0 and last_error and sql:
                retry_messages = messages + [
                    {
                        "role": "assistant",
                        "content": json.dumps({"sql": sql, "explanation": ""}),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"The SQL you just generated failed with this error:\n\n"
                            f"  {last_error}\n\n"
                            "Please correct the SQL and return a valid JSON response."
                        ),
                    },
                ]
            else:
                retry_messages = messages

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=retry_messages,
                    temperature=0.1,   # Low temperature for deterministic SQL
                    max_tokens=1024,
                )

                raw_response = response.choices[0].message.content or ""
                
                # Safe parsing of token usage which might be missing or None in some endpoints
                usage = {}
                if hasattr(response, "usage") and response.usage:
                    usage = {
                        "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                        "total_tokens": getattr(response.usage, "total_tokens", 0),
                    }

                # Parse the JSON response
                parsed = parse_agent_response(raw_response)
                sql = parsed["sql"]
                explanation = parsed["explanation"]

                if not sql:
                    last_error = "Model returned an empty SQL string."
                    continue

                # Try to execute the SQL — this validates correctness
                results = query_db(self.conn, sql, return_as_df=False)

                # ── Success ──────────────────────────────────────────────────
                # Record the assistant's final response in conversation history
                # so subsequent follow-up questions have full context.
                self.conversation_history.append(
                    {"role": "assistant", "content": raw_response}
                )

                return {
                    "sql": sql,
                    "results": results,
                    "explanation": explanation,
                    "error": None,
                    "usage": usage,
                    "attempts": attempt + 1,
                }

            except sqlite3.Error as e:
                last_error = str(e)
            except ValueError as e:
                # parse_agent_response raised — model returned unparseable output
                last_error = str(e)

        # ── All retries exhausted ─────────────────────────────────────────────
        # Still record something in history so the conversation is coherent.
        self.conversation_history.append(
            {
                "role": "assistant",
                "content": f"Failed to generate valid SQL after {max_retries + 1} attempts. Last error: {last_error}",
            }
        )

        return {
            "sql": sql or "",
            "results": [],
            "explanation": "",
            "error": last_error,
            "usage": usage,
            "attempts": max_retries + 1,
        }

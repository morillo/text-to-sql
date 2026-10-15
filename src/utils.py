import re
import sqlite3
from typing import List, Dict, Any, Literal, Optional, Tuple, overload
from pathlib import Path
import pandas as pd


def load_db(db_path: str = "data/Chinook.db", read_only: bool = True) -> sqlite3.Connection:
    """
    Load the SQLite database and return an active connection.

    Args:
        db_path: Path to the SQLite database file. Defaults to "data/Chinook.db".
        read_only: If True, enforces PRAGMA query_only = ON to guard against destructive mutations.

    Returns:
        sqlite3.Connection: Active database connection.
    """
    db_file = Path(db_path)
    if not db_file.exists():
        raise FileNotFoundError(
            f"Database file not found: {db_path}\n"
            "Please run setup.sh first to create the database."
        )

    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        if read_only:
            try:
                conn.execute("PRAGMA query_only = ON")
            except sqlite3.Error:
                pass  # Fall back gracefully if unsupported by older SQLite build
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Error connecting to database: {e}")


def validate_sql_safety(sql: str) -> Tuple[bool, Optional[str]]:
    """
    Guardrail: Validate that generated SQL is strictly a read-only query.

    Blocks destructive statements (DROP, DELETE, UPDATE, INSERT, ALTER, TRUNCATE, ATTACH).

    Args:
        sql: Generated SQL query string.

    Returns:
        Tuple[bool, Optional[str]]: (is_safe, error_message)
    """
    cleaned = sql.strip().upper()

    # Block statements that do not start with SELECT or WITH
    if not (cleaned.startswith("SELECT") or cleaned.startswith("WITH")):
        return False, "Security Violation: Only read-only SELECT or WITH statements are allowed."

    # Check for forbidden keywords as standalone tokens
    forbidden_tokens = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "ATTACH", "DETACH", "PRAGMA"]
    for token in forbidden_tokens:
        if re.search(r"\b" + token + r"\b", cleaned):
            return False, f"Security Violation: Prohibited statement keyword '{token}' detected in query."

    return True, None


@overload
def query_db(
    conn: sqlite3.Connection,
    query: str,
    params: Optional[tuple] = None,
    return_as_df: Literal[True] = True,
    enforce_safety: bool = True,
) -> pd.DataFrame: ...
@overload
def query_db(
    conn: sqlite3.Connection,
    query: str,
    params: Optional[tuple] = None,
    *,
    return_as_df: Literal[False],
    enforce_safety: bool = True,
) -> List[Dict[str, Any]]: ...
@overload
def query_db(
    conn: sqlite3.Connection,
    query: str,
    params: Optional[tuple] = None,
    return_as_df: bool = True,
    enforce_safety: bool = True,
) -> List[Dict[str, Any]] | pd.DataFrame: ...
def query_db(
    conn: sqlite3.Connection,
    query: str,
    params: Optional[tuple] = None,
    return_as_df: bool = True,
    enforce_safety: bool = True,
) -> List[Dict[str, Any]] | pd.DataFrame:
    """
    Execute a SQL query and return results.

    Args:
        conn: Active SQLite database connection.
        query: SQL query string to execute.
        params: Optional tuple of parameters.
        return_as_df: If True, return pandas DataFrame; if False, return list of dicts.
        enforce_safety: If True, validates SQL safety before execution.

    Returns:
        pd.DataFrame or List[Dict[str, Any]]: Query results.
    """
    if enforce_safety:
        is_safe, err_msg = validate_sql_safety(query)
        if not is_safe:
            raise sqlite3.Error(err_msg)

    try:
        if return_as_df:
            return pd.read_sql_query(query, conn, params=list(params) if params else None)
        else:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            columns = [description[0] for description in cursor.description]
            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            return results
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Error executing query: {e}")


def get_schema(conn: sqlite3.Connection) -> Dict[str, List[Dict[str, Any]]]:
    """Get the database schema including all tables and column information."""
    schema = {}
    tables = query_db(
        conn,
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
        return_as_df=False,
    )
    for table in tables:
        table_name = table["name"]
        columns = query_db(conn, f"PRAGMA table_info({table_name})", return_as_df=False)
        schema[table_name] = columns
    return schema


def print_table_schema(
    conn: sqlite3.Connection, table_name: Optional[str] = None
) -> None:
    """Print formatted database schema."""
    schema = get_schema(conn)

    if table_name:
        if table_name not in schema:
            print(f"Error: Table '{table_name}' not found in database.")
            print(f"Available tables: {', '.join(schema.keys())}")
            return
        tables_to_print = {table_name: schema[table_name]}
    else:
        tables_to_print = schema

    print("\n" + "=" * 100)
    print(f"DATABASE SCHEMA - {len(schema)} tables")
    print("=" * 100)

    if not table_name:
        print("\nTables:")
        for i, tbl in enumerate(schema.keys(), 1):
            print(f"  {i}. {tbl}")
        print()

    for tbl_name, columns in tables_to_print.items():
        print("\n" + "-" * 100)
        print(f"Table: {tbl_name} ({len(columns)} columns)")
        print("-" * 100)
        print(f"{'Column':<30} {'Type':<20} {'Nullable':<12} {'PK':<5} {'Default':<15}")
        print("-" * 100)

        for col in columns:
            nullable = "NULL" if col["notnull"] == 0 else "NOT NULL"
            pk = "Y" if col["pk"] > 0 else ""
            default = str(col["dflt_value"]) if col["dflt_value"] is not None else ""
            print(
                f"{col['name']:<30} {col['type']:<20} {nullable:<12} {pk:<5} {default:<15}"
            )

    print("=" * 100 + "\n")


def get_schema_string(conn: sqlite3.Connection) -> str:
    """Build schema DDL + sample rows string for LLM prompt injection."""
    parts = []
    tables = query_db(
        conn,
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name",
        return_as_df=False,
    )

    for table in tables:
        table_name = table["name"]
        ddl_rows = query_db(
            conn,
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            params=(table_name,),
            return_as_df=False,
        )
        if ddl_rows and ddl_rows[0]["sql"]:
            parts.append(ddl_rows[0]["sql"].strip() + ";")

        try:
            rows = query_db(conn, f"SELECT * FROM [{table_name}] LIMIT 2", return_as_df=False)
            if rows:
                cols = list(rows[0].keys())
                parts.append(f"-- Sample rows from {table_name}:")
                parts.append(f"--   columns: {', '.join(cols)}")
                for row in rows:
                    values = ", ".join(str(v) for v in row.values())
                    parts.append(f"--   {values}")
        except sqlite3.Error:
            pass

        parts.append("")

    return "\n".join(parts)


def extract_sql(text: str) -> str:
    """Extract SQL from free-form model output string."""
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

"""Unit and integration tests for TextToSQLAgent and utilities (built-in unittest)."""

import unittest
import sqlite3
import tempfile
from pathlib import Path

from src.utils import load_db, query_db, validate_sql_safety, get_schema_string
from src.agent import parse_agent_response
from src.evals import compare_results


class TestTextToSQL(unittest.TestCase):

    def test_validate_sql_safety(self):
        """Test SQL safety guardrail blocks non-SELECT/destructive queries."""
        safe, err = validate_sql_safety("SELECT * FROM Customer")
        self.assertTrue(safe)
        self.assertIsNone(err)

        safe, err = validate_sql_safety("WITH cte AS (SELECT 1) SELECT * FROM cte")
        self.assertTrue(safe)

        safe, err = validate_sql_safety("DROP TABLE Customer;")
        self.assertFalse(safe)
        self.assertIn("Security Violation", err or "")

        safe, err = validate_sql_safety("DELETE FROM Track WHERE TrackId = 1")
        self.assertFalse(safe)
        self.assertIn("Security Violation", err or "")

        safe, err = validate_sql_safety("UPDATE Employee SET Salary = 100000")
        self.assertFalse(safe)
        self.assertIn("Security Violation", err or "")

    def test_parse_agent_response(self):
        """Test response parser handling JSON, fenced blocks, and fallback."""
        res = parse_agent_response('{"sql": "SELECT 1", "explanation": "test"}')
        self.assertEqual(res["sql"], "SELECT 1")
        self.assertEqual(res["explanation"], "test")

        res = parse_agent_response('```json\n{"sql": "SELECT 2", "explanation": "fenced"}\n```')
        self.assertEqual(res["sql"], "SELECT 2")

        res = parse_agent_response('Here is your query: SELECT * FROM Track;')
        self.assertEqual(res["sql"], "SELECT * FROM Track")

    def test_compare_results(self):
        """Test normalized result set comparison logic."""
        expected = [{"Name": "Rock", "Total": 100.555}]
        predicted = [{"Genre": "Rock", "Total": 100.56}]
        self.assertTrue(compare_results(predicted, expected))

        different = [{"Name": "Pop", "Total": 100.56}]
        self.assertFalse(compare_results(different, expected))

    def test_schema_string(self):
        """Test database schema string extraction."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_file = Path(tmp_dir) / "test.db"
            conn = sqlite3.connect(db_file)
            conn.execute("CREATE TABLE Test (id INT PRIMARY KEY, name TEXT)")
            conn.execute("INSERT INTO Test VALUES (1, 'Alice')")
            conn.commit()

            schema_str = get_schema_string(conn)
            self.assertIn("CREATE TABLE Test", schema_str)
            self.assertIn("Alice", schema_str)
            conn.close()


if __name__ == "__main__":
    unittest.main()

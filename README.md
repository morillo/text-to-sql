# Generic Model-Agnostic Text-to-SQL Agent

A production-ready, model-agnostic Text-to-SQL agent CLI. This tool allows users to query SQLite databases using natural language directly from their terminal, formatting results in clean ASCII tables, providing real-time token streaming, and explaining the queries in plain English.

---

## Key Features

* **100% Provider Agnostic**: Out-of-the-box support for OpenAI (`gpt-4o`, `gpt-4o-mini`), Google Gemini, DeepSeek, local Ollama models, or any OpenAI-compatible API endpoint via standard environment variables.
* **Real-Time Token Streaming (`query_stream`)**: Renders generated SQL tokens chunk-by-chunk in real-time, bringing perceived Time-To-First-Token (TTFT) latency under **1.0 second**.
* **Database Agnostic (`--db`)**: Point the CLI at any arbitrary SQLite database file (`uv run python -m src.cli --db custom.db`).
* **Multi-Turn Conversation Memory**: Preserves context across queries. You can ask follow-up questions (e.g., *"show only the top 3"* or *"filter to 2021"*) without re-specifying the entire request.
* **Full-Schema Injection**: Auto-injects database DDL structures and realistic sample rows for all tables into the system prompt, reducing table/column hallucinations to near zero.
* **Self-Correcting Execution Retry Loop**: If generated SQL fails SQLite execution, the agent feeds the error message back to the LLM for automatic self-correction (up to 2 retries).
* **Security & Read-Only Guardrails**: Enforces `PRAGMA query_only = ON;` and application-level SQL parsing to block non-SELECT / destructive statements (`DROP`, `DELETE`, `UPDATE`, `INSERT`).
* **CSV Export (`export <file.csv>`)**: Export active query results to CSV directly from the terminal.
* **Evaluation Framework**: Measures Execution Accuracy (EX) and Result Accuracy (RA) against gold-standard SQL and answers (`--dataset dev|extended|all`).
* **Automated Unit Tests**: Built-in test suite (`uv run python -m unittest tests/test_agent.py`).

---

## Architecture Overview

```
                  ┌──────────────────────┐
                  │     python -m        │
                  │      src.cli         │
                  └──────────┬───────────┘
                             │ user question
                             ▼
                  ┌──────────────────────┐
                  │    TextToSQLAgent    │
                  │   (src/agent.py)     │
                  └──────────┬───────────┘
                             │
     ┌───────────────────────┼───────────────────────┐
     ▼                       ▼                       ▼
┌──────────────┐      ┌──────────────┐      ┌────────────────┐
│  Injects DDL │      │ Execs query  │      │ Queries LLM    │
│  & Samples   │      │ & Guardrails │      │ Client (via    │
│ (src/utils)  │      │  (SQLite3)   │      │ OpenAI SDK)    │
└──────────────┘      └──────────────┘      └────────────────┘
```

---

## Setup & Installation

### 1. Run Setup Script
Setup: `uv sync` (uv installs the pinned Python from `.python-version` and the locked dependencies from `uv.lock`). The setup script runs it and downloads the sample SQLite database (`Chinook.db`):
```bash
./setup.sh
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Fill in your API credentials:

#### Option A: OpenAI Setup (Default)
```env
OPENAI_API_KEY=your-openai-api-key-here
LLM_MODEL=gpt-4o  # or gpt-4o-mini
```

#### Option B: Google Gemini Setup (via OpenAI Compatibility Endpoint)
```env
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash  # or gemini-1.5-pro, gemini-2.5-pro
LLM_API_KEY=your-gemini-api-key
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
```

#### Option C: Generic OpenAI-Compatible Endpoint (e.g., DeepSeek, Local Ollama, Together AI)
```env
LLM_PROVIDER=openai
LLM_MODEL=deepseek-chat
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.deepseek.com/v1  # or http://localhost:11434/v1 for Ollama
```

---

## How to Run

### Interactive Terminal CLI
Launch the terminal query assistant:
```bash
# 1. Run CLI against sample Chinook database
uv run python -m src.cli

# 2. Point CLI at a custom SQLite database file
uv run python -m src.cli --db path/to/your_database.db
```

#### In-Session CLI Commands:
| Command | Description |
|---|---|
| `<your question>` | Convert natural language to SQL, stream response, execute query, and format ASCII results. |
| `schema` or `schema <table>` | Print database schema structure and column definitions. |
| `export <file.csv>` | Export active query results to a CSV file. |
| `reset` | Clear multi-turn conversation history and start a fresh session. |
| `exit` / `quit` | End terminal session. |

---

## Evaluations & Benchmarking

### Running Quality Evaluations
Evaluate the agent's Execution Accuracy (EX) and Result Accuracy (RA) against benchmark question sets:
```bash
# Evaluate on public dev questions (10 questions)
uv run python -m src.evals --dataset dev

# Evaluate on extended edge-case questions (10 questions)
uv run python -m src.evals --dataset extended

# Run full evaluation suite
uv run python -m src.evals --dataset all --output eval_results.json

# Skip baseline comparison
uv run python -m src.evals --no-baseline
```

### Running Latency & Performance Benchmarks
Measure system latency (P50/P95/Mean) and token consumption metrics:
```bash
uv run python -m src.perf --runs 3 --output perf_report.json
```

### Running Automated Unit Tests
Run the built-in test suite:
```bash
python -m unittest tests/test_agent.py
```

### Re-Generating Answer Set
Re-run the agent over all dev questions to update `dev_answers.json`:
```bash
python scripts/generate_answers.py
```

---

## Architectural Decision: Framework Analysis (LangChain / LangGraph vs. Vanilla Python)

An architecture analysis was conducted to determine whether to rewrite this codebase using **LangChain** or **LangGraph** versus maintaining the current **Vanilla Python + OpenAI SDK** structure.

### Comparison Summary

| Criteria | Vanilla Python (Current) | LangChain | LangGraph |
|---|---|---|---|
| **Architecture** | Direct, imperative Python | Linear DAG chain abstractions | State machine graph loops |
| **Dependency Bloat** | Low (only `openai` & `python-dotenv`) | High (large nested packages) | High (requires graph runtimes) |
| **Self-Correction Retry Loop** | Simple `try-except` retry loops | Clumsy (cycles hard to model in LCEL) | Native (cycles are first-class edges) |
| **Debuggability** | Trivial (normal stack traces) | Hard (complex nested trace stacks) | Medium (visual traces via LangSmith) |
| **Latency / Execution Overhead**| Near-zero | Medium (abstraction wrappers) | Medium-High (state graph overhead) |

### Recommendation
**We recommend keeping the current Vanilla Python architecture.** For single/multi-database querying with structured JSON parsing, real-time streaming, and a self-correcting retry loop, the simplicity, minimal latency, and zero dependency bloat of vanilla Python vastly outweigh the overhead of LangChain or LangGraph.
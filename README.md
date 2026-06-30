# Model-Agnostic Text-to-SQL Agent

A clean, production-ready, and model-provider agnostic Text-to-SQL agent CLI. This tool allows users to query SQLite databases using natural language directly from their terminal, formatting results in clean ASCII tables and explaining the queries in plain English.

---

## Features

* **Multi-Turn Conversation Support**: The agent retains context across queries. You can ask follow-up questions (e.g., *"show only the top 3"* or *"filter to 2021"*) without re-specifying the entire request.
* **Full-Schema Injection**: Auto-injects database DDL structures and realistic sample rows for all tables into the prompt, reducing table and column hallucinations to near zero.
* **Automatic Self-Correction**: Implements a self-correction retry loop. If a generated SQL query fails SQLite execution, the agent feeds the error message back to the LLM and asks it to correct itself (up to 2 retries).
* **100% Provider Agnostic**: Can be run with OpenAI, Google Gemini, DeepSeek, local Ollama models, or any OpenAI-compatible API endpoint via simple environment variables.
* **Evaluation Framework**: Measures Execution Accuracy (EX) and Result Accuracy (RA) against gold-standard SQL and answers.
* **Benchmarking Tool**: Gauges model performance, latency (P50/P95), token consumption, and projected run costs.

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
│  & Samples   │      │   & Retries  │      │ Client (via    │
│ (src/utils)  │      │  (SQLite3)   │      │ OpenAI SDK)    │
└──────────────┘      └──────────────┘      └────────────────┘
```

---

## Setup & Installation

### 1. Run Setup Script
Execute the setup script to create a python virtual environment, install dependencies, and download the sample SQLite database (`Chinook.db`):
```bash
./setup.sh
```

### 2. Configure Environment Variables
Copy the `.env.example` file to `.env`:
```bash
cp .env.example .env
```
Open `.env` and fill in your configuration:

#### Option A: OpenAI Setup
```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o  # or gpt-4o-mini
LLM_API_KEY=your-openai-api-key
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
LLM_BASE_URL=https://api.deepseek.com/v1  # or local/other base URL
```

---

## How to Run

### Interactive CLI
Launch the terminal query assistant:
```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Run CLI
python -m src.cli
```

#### Built-in CLI Commands:
| Command | Description |
|---|---|
| `<your question>` | Convert question to SQL, execute it, display results, and explain. |
| `schema` | Print the full schema structure of all tables in the database. |
| `reset` | Clear conversation history and start a fresh session. |
| `exit` / `quit` | End the session. |

---

## Evaluations & Benchmarking

### Running Quality Evaluations
Evaluate the agent's SQL execution and result accuracy against the 10 golden development questions:
```bash
# Run agent vs schema-less baseline comparison (requires configured .env)
python -m src.evals

# Run agent evaluation only (skips baseline API calls)
python -m src.evals --no-baseline

# Save detailed results log to a JSON file
python -m src.evals --output eval_results.json
```

### Running Latency & Cost Benchmarks
Measure system latency and API pricing metrics:
```bash
# Run performance test (default: 3 iterations over 10 questions)
python -m src.perf

# Specify number of benchmark iterations
python -m src.perf --runs 5

# Save performance benchmark report to a JSON file
python -m src.perf --output perf_report.json
```

### Re-Generating Answer Set
Re-run the agent over all development questions and update the public answer key:
```bash
python scripts/generate_answers.py
```
This writes `dev_answers.json` containing SQL statements and formatted data outputs to the project root.

---

## Architectural Decision: Framework Redesign Analysis

An analysis was conducted to determine whether to redesign this codebase using **LangChain** or **LangGraph** versus maintaining the current **Vanilla Python + OpenAI SDK** architecture.

### Redesign Comparison Summary

| Criteria | Vanilla Python (Current) | LangChain | LangGraph |
|---|---|---|---|
| **Architecture** | Direct, imperative Python | Linear DAG chain abstractions | State machine graph loops |
| **Dependency Bloat** | Low (only `openai` & `dotenv`) | High (large nested packages) | High (requires graph runtimes) |
| **Self-Correction Retry Loop** | Simple `try-except` retry loops | Clumsy (cycles are hard to model in LCEL) | Native (cycles are first-class edges) |
| **Debuggability** | Trivial (normal stack traces) | Hard (complex nested trace stacks) | Medium (visual traces via LangSmith) |
| **Latency / Execution Overhead**| Near-zero | Medium (abstraction wrappers) | Medium-High (state graph overhead) |

### Pros & Cons Analysis

#### 1. Current Vanilla Architecture
* **Pros:** Highly performant with near-zero latency overhead, trivial to trace and debug using standard tools (`pdb`, prints), and has a clean, readable implementation (under 350 lines).
* **Cons:** Memory management, token pruning, or complex routing decisions must be coded manually.

#### 2. LangChain
* **Pros:** Standardized models/vectorstores integration, pre-built SQL helper classes.
* **Cons:** Rigid linear pipelines that make recursive error-retries clumsy. Heavy dependencies can cause package conflicts.

#### 3. LangGraph
* **Pros:** Native, clean representation of the cyclic generation-verification-correction loop. Centralized type-safe state tracking. Supports human-in-the-loop validation (pausing/resuming graphs).
* **Cons:** Significant boilerplate code (declaring graphs, nodes, edges, state schemas) which is overkill for a simple single-database agent.

### Current Verdict & recommendation

> [!IMPORTANT]
> **We recommend keeping the current Vanilla Python architecture.**
> For the current scope (a single-database agent with basic query generation, a simple multi-turn context, and a 2-attempt execution retry loop), the simplicity, minimal latency, and zero dependency overhead of the custom vanilla Python agent vastly outweigh the benefits of LangChain or LangGraph.

### When should you migrate to LangGraph?
You should consider migrating the agent to **LangGraph** only if:
1. **Multi-DB Scale**: You need to dynamically route queries across dozens of database connections based on semantic schema lookup (RAG).
2. **Human-in-the-loop**: You require developers/admins to review and manually edit or approve generated SQL queries before executing them.
3. **Multi-Agent Collaboration**: The workflow expands to multiple cooperative agents (e.g. an SQL generator, a security auditor, and an execution supervisor agent).
# SQL Assistant - LangGraph Implementation Plan

## Overview
Building a modular AI assistant using LangGraph and LangChain for SQL query generation and data extraction. This implementation leverages LangGraph's built-in agent architectures, tool calling, memory, and planning capabilities.

## Folder Structure
```
sql_assistant/
├─ workflows/
│  ├─ __init__.py
│  └─ main_graph.py          # LangGraph wiring: nodes, conditional edges, interrupts, checkpointer hookup
│
├─ core/
│  ├─ __init__.py
│  ├─ state.py               # Typed state keys + merge rules (user_query, schema_context, plan, sql, validation_ok, loop_count, etc.)
│  ├─ config.py              # Policy knobs (gen_k, max_loops, row/time caps, blocklist, defaults per env)
│  ├─ safety.py              # DDL/DML blocklist checks, LIMIT injection, read-only helpers
│  ├─ tools.py               # Common tool shims: DB connectors, EXPLAIN/dry-run, schema introspection
│  └─ memory/
│     ├─ __init__.py
│     ├─ checkpointer.py     # Choice of file/SQLite/Redis checkpointer wrapper
│     └─ knowledge.py        # (Optional) synonyms, prior corrections, few-shot index handles
│
├─ agents/
│  ├─ schema_context/        # (Toolish) subgraph that loads & merges schema contexts
│  │  ├─ __init__.py
│  │  └─ subgraph.py         # wraps get_db_schema + SemanticSchemaManager + combine
│  │
│  ├─ interpret_plan/        # Router + Worker subgraph (Intent & Plan)
│  │  ├─ __init__.py
│  │  ├─ subgraph.py         # parse intent, build context, create plan, detect ambiguity, plan validation/auto-fix
│  │  └─ prompts.py          # planner/intent prompts (+ schema adaptation templates)
│  │
│  ├─ sql_generate/          # Worker subgraph (k-best generation)
│  │  ├─ __init__.py
│  │  ├─ subgraph.py         # few-shot retrieval, candidate generation, ranking, safety pre-checks
│  │  └─ prompts.py          # generation templates w/ dialect guards
│  │
│  ├─ validate_diagnose/     # Gate + Worker subgraph (Reflection)
│  │  ├─ __init__.py
│  │  ├─ subgraph.py         # syntax/AST checks, semantic probes, patch/swap candidate, bounded loop control
│  │  ├─ probes.py           # cheap COUNTs, value existence, join selectivity helpers
│  │  └─ prompts.py          # critic/repair prompts (optional)
│  │
│  ├─ execute/               # Tool node (safe DB runner)
│  │  ├─ __init__.py
│  │  └─ runner.py           # EXPLAIN, read-only/session config, LIMIT + timeout, transient retries
│  │
│  └─ present/               # Tool/Worker (format + optional viz)
│     ├─ __init__.py
│     └─ renderer.py         # tabular formatting; optional chart spec proposer
│
├─ prompts/                  # Shared/system-level prompt fragments (kept separate from agent-specific)
│  ├─ system.txt
│  └─ style.txt
│
├─ ui/
│  └─ gradio_app.py          # simple front-end to call the compiled graph and stream results
│
├─ tests/
│  ├─ unit/
│  │  ├─ test_router.py
│  │  ├─ test_sqlgen.py
│  │  ├─ test_validator.py
│  │  └─ test_safety.py
│  └─ e2e/
│     ├─ test_graph_routing.py   # never execute unless validation_ok==True, loop bound respected
│     └─ test_golden_queries.py  # pass@1 / pass@k on your golden set
│
├─ data/
│  ├─ semantic/              # optional semantic schemas
│  └─ examples/              # few-shot examples / golden queries
│
├─ docs/
│  ├─ Architecture.md        # your original + LangGraph mapping
│  └─ AgentCards.md          # inputs/outputs/tools/policy/stop-conditions per subgraph
│
├─ .env.example              # OPENAI_API_KEY=..., DATABASE_URL=..., SEMANTIC_SCHEMA_PATH=...
├─ requirements.txt
├─ dev-requirements.txt
└─ README.md

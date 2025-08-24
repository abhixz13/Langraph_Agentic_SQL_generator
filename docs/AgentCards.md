"""
# Purpose: 
A short, human-friendly spec for each agent/subgraph that your team (and future you) can read. It explains what the node does, what it consumes and produces, and the rules it must follow. It’s where you lock the contract before writing code.
"""

## Global State Contract (keys & merge rules)
- **Inputs / Control** *(replace / increment counters)*
  - `user_query`, `dialect`, `gen_k`, `max_loops`, `loop_count`
- **Schema** *(replace)*
  - `db_schema`, `semantic_schema?`, `schema_context`
- **Interpret / Plan** *(replace)*
  - `intent_json`, `plan`, `plan_ok`, `plan_confidence`, `ambiguity?`, `clarifying_question?`, `clarifying_answer?`
- **Generate** *(replace for singletons, append for lists, increment counters)*
  - `few_shots[]`, `sql_candidates[]`, `sql`, `gen_attempts`
- **Validate** *(replace/append/increment as noted)*
  - `validation_ok`, `val_reasons[]`, `val_signals{}`, `val_attempts`
- **Execute** *(replace)*
  - `exec_preview`, `result_sample[]`, `exec_rows`, `error?`
- **Observability** *(append/aggregate)*
  - `metrics{latency_ms, tokens, model}`

Merge rules summary:
- **replace:** singletons & dicts (e.g., `sql`, `plan`, `validation_ok`, `schema_context`)
- **append:** lists (e.g., `val_reasons`, `sql_candidates`)
- **increment:** counters (`loop_count`, `gen_attempts`, `val_attempts`)

---

## Module 1 — Schema Context (Tool)

**Purpose:** Load DB schema, optionally load semantic schema, and produce a **combined `schema_context`** used downstream.

- **Reads:** DB connection (internal), `SEMANTIC_SCHEMA_PATH?`
- **Writes:** `db_schema`, `schema_context`
- **Tools reused:** `get_db_schema()`, `SemanticSchemaManager.load_schema(path)`, `get_combined_schema_context(semantic, schema)`
- **Policy knobs:** none (pure tool)
- **Stop conditions:** always completes or raises a hard error (invalid DB/semantics)
- **Failure modes:** connection error, malformed semantic schema
- **Metrics:** latency

---

## Module 2 — Interpret & Plan (Router + Worker)

**Purpose:** Convert NL query to `intent_json`, draft `plan`, validate the plan; set `ambiguity` and a `clarifying_question` if needed.

- **Reads:** `user_query`, `schema_context`, memory (synonyms/prior fixes), `dialect?`
- **Writes:** `intent_json`, `plan`, `plan_ok`, `plan_confidence`, `ambiguity?`, `clarifying_question?`
- **Tools reused:** `IntentParserAgent.parse(q, schema_info)`, `adapt_schema_for_llm()`, `_build_context_generic()`, `SQLPlannerAgent.create_plan()`, `SQLValidatorAgent.validate_plan()`
- **Policy knobs:** ambiguity threshold
- **Stop conditions:** 
  - If ambiguous → set `ambiguity=true` + `clarifying_question`
  - Else `plan_ok=true` and proceed
- **Failure modes & auto-fixes:** missing join path, type mismatch → deterministic plan tweaks (join key, group-by completion)
- **Metrics:** latency, tokens, plan confidence

---

## Module 3 — SQL Generate (k-best) (Worker)

**Purpose:** Produce **k** dialect-aware SQL candidates and pick a top-1 that parses and honors safety pre-checks.

- **Reads:** `plan`, `schema_context`, `dialect`, `gen_k`
- **Writes:** `few_shots[]`, `sql_candidates[]`, `sql`, `gen_attempts += 1`
- **Tools reused:** “Few-shot examples” loader, `SQLGeneratorAgent.generate_sql(plan)`
- **Policy knobs:** `gen_k`
- **Stop conditions:** at least one parseable candidate with `LIMIT` and no blocked keywords
- **Failure modes & auto-fixes:** inject `LIMIT`, drop blocked keywords, re-rank by AST quality
- **Metrics:** latency, tokens, k produced

---

## Module 4 — Validate & Diagnose (Reflection) (Gate + Worker)

**Purpose:** Verify SQL safety & semantics; run **cheap probes**; if failing, patch `plan/sql` or choose next candidate; **bounded loop**.

- **Reads:** `sql`, `sql_candidates[]`, `plan`, `schema_context`, `dialect`, `loop_count`, `max_loops`
- **Writes:** `validation_ok`, `val_reasons[]`, `val_signals{}`, `val_attempts += 1`, (`plan` or `sql` if patched), `loop_count ±`
- **Tools reused:** “Schema Validation” (parse/AST, table/column check), probes (`COUNT(*)` with filters, join selectivity, value existence)
- **Policy knobs:** `max_loops`, probe budget
- **Stop conditions:** 
  - `validation_ok=true` → advance
  - else if `loop_count < max_loops` → retry via next candidate/patch
  - else → end with helpful error + suggestions
- **Failure modes & auto-fixes:** zero rows due to casing → switch to `ILIKE`; wrong join key → patch plan; missing column → pick alt candidate
- **Metrics:** latency, tokens, attempts, probe count

---

## Module 5 — Safe Execute (Tool)

**Purpose:** Dry-run / EXPLAIN then execute **read-only** with `LIMIT` & timeout; small transient retry policy.

- **Reads:** `sql` (must be validated), `dialect`, caps (`ROW_CAP`, `EXECUTE_TIMEOUT_MS`)
- **Writes:** `exec_preview`, `result_sample[]`, `exec_rows`, `error?`
- **Tools reused:** “Query Execution”, “Retry Logic”
- **Policy knobs:** `ROW_CAP`, `EXECUTE_TIMEOUT_MS`, `DB_READONLY=true`
- **Stop conditions:** success or transient failure with bounded retries
- **Failure modes:** timeouts, connection drops (retry); permission errors (fail)
- **Metrics:** latency, rows returned

---

## Module 6 — Present Results (Tool/Worker)

**Purpose:** Format rows and optionally produce a visualization spec when aggregates/categories exist; render to UI.

- **Reads:** `result_sample[]`, `plan/intent`
- **Writes:** UI payload (and optionally `viz_spec`)
- **Tools reused:** “Results Processing”, “Visualization Agent”, “Display Results”
- **Policy knobs:** none
- **Stop conditions:** always completes
- **Failure modes:** very wide tables (paginate)
- **Metrics:** latency

---

## Interrupt — Human Clarification

**Trigger:** `ambiguity == true` from Interpret & Plan (or targeted validator questions).

- **Writes after user reply:** `clarifying_answer` (used to update `plan`)
- **Coordinator behavior:** pause & resume from checkpoint

---

## Invariants & Guardrails (for tests)

- **Never** reach Execute unless `validation_ok == true`.
- Reflection loop halts at `loop_count >= max_loops`.
- All generated SQL includes `LIMIT` ≤ `ROW_CAP`.
- DDL/DML keywords are blocked everywhere.
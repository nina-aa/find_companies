# Agentic Company-Search System

Natural-language company mandate → ranked, evidence-backed shortlist over a
50,000-row synthetic dataset. Built for the Comparables.ai technical assessment.

> **Status:** M1–M4 complete — data foundation, retrieval tools, LLM layer, all 7
> workflow nodes, the bounded runner (custom driver **and** LangGraph), CLI
> `run` / `eval`, the FastAPI endpoint, and structured logging. The eval harness
> reports 7/7 green on the deterministic checks. M5 (deploy + README polish) next.
> See `PLAN.md` (not committed) for the full roadmap.

---

## Quick start

```bash
python -m pip install -r requirements.txt

# Build the retrieval index (SQLite + FTS5) from the raw dataset.
python -m app.cli ingest

# Sanity-check it with raw SQL.
python -m app.cli db "SELECT industry, COUNT(*) FROM companies GROUP BY industry"

# Full workflow on one mandate (needs an OpenAI key; ~$0.002, ~10s).
python -m app.cli run "German drug-discovery companies, preferably founded after 2018" \
  --provider openai --verbose --explain

# The evaluation set -> report table + eval/RESULTS.md
python -m app.cli eval --provider openai

# HTTP API
uvicorn app.api:app --port 8000
curl -s -X POST localhost:8000/agent/search -H 'X-API-Key: <key>' \
  -H 'content-type: application/json' -d '{"query":"UK fintech doing payments or lending"}'

# Run the test suite (offline, no API tokens).
python -m pytest

# Opt-in live smoke test (one real gpt-4o-mini call, < $0.001):
RUN_LIVE=1 python -m pytest tests/test_live_smoke.py -q -s
```

The index is written to `data/index/companies.db` (+ `manifest.json`) and is not
committed — it is fully reproducible from `data/companies.json` via `app.ingest`.

---

## Architecture (planned)

A **deterministic workflow**, not an autonomous agent: 7 nodes, of which exactly
2 make a single schema-constrained LLM call each; every routing decision is made
by Python, not a model.

```
interpret_mandate ─► build_search_plan ─► check_feasibility ─► retrieve ─► validate_and_rank ─► compose_response
   (LLM #1)            (deterministic)      (deterministic)     (tools)      (LLM #2)             (deterministic)
                                                 │                              │
                              0 mandatory matches ┘         not enough matches ─┘─► relax_preferences ─► retrieve …
```

The LLM interprets the mandate and judges candidate↔mandate match + evidence.
Everything else — filtering, ranking, revision policy, budget enforcement,
grounding checks — is deterministic and testable without the API.

---

## Repository layout

| Path | What |
|---|---|
| `app/ingest.py` | Build `companies.db`: typed columns, region lookup, FTS5 (bm25), manifest. |
| `app/db.py` | Read-only SQLite connection + manifest loader. |
| `app/schemas.py` | Pydantic tool contracts: `StructuredFilters`, `Exclusions`, `Candidate`, `Company`, `SearchResult`. |
| `app/tools.py` | `search_companies` / `count_matching` / `get_by_ids` — deterministic, 0 tokens. |
| `app/llm.py` | `LLMClient` + `OpenAIProvider` / `FakeProvider`, response cache, cost table. |
| `app/state.py` | Mandate / plan / validation schemas, `RunState`, `RunTrace`, `AgentResponse`. |
| `app/prompts.py` | Prompt builders for the two LLM calls (pure functions). |
| `app/nodes.py` | The 7 workflow nodes as `RunState → RunState` functions. |
| `app/budget.py` | `BudgetGuard` — the hard bound on LLM calls / iterations / revisions / deadline. |
| `app/workflow.py` | `run_workflow()` + the custom bounded driver. |
| `app/graph.py` | The LangGraph engine (same nodes, same guard). |
| `app/api.py` | FastAPI: `POST /agent/search`, `X-API-Key` guard. |
| `app/logging.py` | Stdlib JSON structured logging, one record per stage. |
| `app/evaluate.py` | Assessment eval harness → `eval/RESULTS.md`. |
| `app/revenue.py` | `revenue_range` bucket ↔ euro-interval parsing. |
| `app/config.py` | Loads `schema_map.yaml` / `regions.yaml` / `lexicon.yaml`; region + industry resolution. |
| `app/cli.py` | Command-line entry point (the primary test surface). |
| `app/profile_dataset.py` | Reproducible dataset profiling → `data/PROFILE.md`. |
| `eval/queries.yaml` | Held-out evaluation set (5 required + S1/S3) with hand-authored `expected` blocks. |
| `eval/dev_queries.yaml` | 10 self-authored dev queries used for all iteration. |
| `tests/` | Unit tests — run against a fixture index, no API. |

---

## Dataset ingestion & indexing

`python -m app.cli ingest` (alias for `python -m app.ingest`) reads
`data/companies.json`, normalises each record through `app/schema_map.yaml`, and
produces:

- **`companies`** — one typed row per company plus derived `revenue_min_eur` /
  `revenue_max_eur` columns parsed from the bucket string.
- **`company_regions`** — `(company_id, region)` pairs so region filters
  (`nordic`, `europe`, …) are a plain indexed JOIN.
- **`companies_fts`** — an FTS5 (bm25, porter-stemmed) index over `name` +
  `description`, the primary keyword/topic signal.
- **`data/index/manifest.json`** — row count, source-file SHA-256, build
  timestamp, embedding model (`none`). The runtime can refuse to start if this
  disagrees with its config.

No embeddings are built: the descriptions are templated (`"{prefix} {noun} for
{topics}."`), so FTS5 keyword matching over the structurally-filtered set
captures nearly all of the semantic signal. The embedding path is described under
*What I'd do next*.

---

## Decision Log

Appended as decisions are made. Each entry: **what**, **why**, **the alternative**.

### D1 — Deterministic workflow, not autonomous agents
The brief explicitly rewards this (*"the number of agents is not an evaluation
criterion"*, *"must not be able to enter an uncontrolled reasoning or tool-use
loop"*). The task is well-defined enough that the retrieval strategy is derivable
from the parsed mandate, so an LLM-driven planner would add failure modes without
adding capability. **Alternative (README-only):** an agent with a large tool
space, warranted when the right strategy genuinely cannot be determined up front.

### D2 — SQLite + FTS5 for retrieval
50k rows fit in RAM, but SQLite makes the structured filters *be* SQL, gives
bm25 keyword ranking for free, and makes `count_matching` (the feasibility gate)
a one-liner — the honest engineering choice at this scale, using only the stdlib.
**Alternative (README-only):** Elasticsearch + a vector DB with ANN, hybrid-ranked
and sharded, for 350M+ companies.

### D3 — FTS5 only, no embeddings (yet)
Dataset profiling (`data/PROFILE.md`): 1,308 distinct descriptions over 50,000
rows, each a template collapsing to a handful of topic keywords. Keyword ≈
semantic here. **Alternative (README-only / if time):** a `fastembed` local-ONNX
cosine re-rank over the filtered subset — no FAISS at this scale.

### D4 — Region and revenue resolution are deterministic config, never LLM
`regions.yaml` maps `nordic` → `{Finland, Norway, Sweden}` (the data has only 8
countries; Denmark/Iceland absent). `revenue.py` maps `"below EUR 10M"` →
buckets `{0-1M, 1M-10M}`. Relative terms are resolved to concrete values before
any filtering, and unknown region terms are recorded as ambiguities rather than
guessed. **Alternative:** let the LLM enumerate countries — rejected as
non-traceable and error-prone.

### D5 — `schema_map.yaml` indirection for ingestion
The ingestion layer reads a field-mapping config instead of hard-wiring the eight
dataset fields, so a differently-shaped source only needs a new mapping.

### D6 — Retrieval tools return a `SearchResult`, not a bare `list[Candidate]`
`search_companies` returns the ranked candidates **plus** exclusion bookkeeping
(`matched_filters`, `excluded`, `fts_query`, `truncated`). Q3 must be able to
report "exclusion criteria matched no candidates" and S3 "removed N" — losing
those counters in the tool signature would push that logic into the caller.
Every tool has an explicit Pydantic in/out contract ([app/schemas.py](app/schemas.py))
and costs **zero tokens** — they are plain SQL.

### D7 — Preferences can never reach a retrieval tool
`search_companies` only accepts `StructuredFilters` (mandatory) + `topic_terms` +
`Exclusions`. There is no parameter through which a *preference* (founded-year,
headcount band) could become a hard filter — preferences are applied later as
ranking boosts in `validate_and_rank`. Enforced by the type, not by convention.

### D8 — One `LLMClient` seam; fake provider is the test default
Every model call goes through `LLMClient.complete(messages, response_model)`
([app/llm.py](app/llm.py)). `OpenAIProvider` uses `chat.completions.parse` with a
Pydantic model as strict `response_format`; `FakeProvider` returns schema-valid
dummies (`fabricate`) so the whole test suite runs offline at $0. One repair
retry on a schema-validation failure, then a structured `LLMError`; a 429 gets
one `Retry-After` backoff then fails gracefully. The response cache mixes the
provider identity (`fake` vs `openai:gpt-4o-mini`) into every key, so a fake
answer is never served for a real call.

### D9 — Few-shot the mandatory/preference split
gpt-4o-mini, given rules alone, repeatedly classified "preferably founded after
2018" as a *mandatory* filter and emitted `0` for unspecified numeric bounds.
Two worked examples (built from the real Pydantic models, so every field + null
is shown) fixed both. Verified with the live smoke test. The alternative —
escalating interpret to `gpt-4o` — was rejected: 16× the cost for a nuance a
demonstration handles.

### D10 — Groundedness is a deterministic check, not a prompt instruction
`validate_and_rank` asks the model for a `quote` per text judgement, then
[app/nodes.py](app/nodes.py) verifies each quote is a literal substring of the
named field of the *real* record ([get_by_ids]). A quote that fails is demoted
from `evidence` to `inference`. A cited quote is an LLM claim until checked.

### D11 — Revision is deterministic and bounded
`relax_preferences` is a fixed ladder (widen founding-year → drop it → drop the
employee band → enlarge the pool), never an LLM call — Q4 specifies the rule
itself. Mandatory criteria and exclusions are never touched; at most one revision
per run. On this dataset revision rarely triggers from a query string (templated
descriptions keep most candidates verifiable), so the path is proven by a unit
test that scripts an empty first validation pass.

### D12a — The narrowing funnel is reported explicitly
`RunTrace.funnel` (in `--verbose`, the `run.funnel` log line, and
`metadata.funnel`) records the count at every narrowing step —
`mandatory_filters → topic_match → after_exclusions → retrieved_pool (≤100) →
sent_to_validation (≤10) → passed_validation → returned`. So "580 companies
matched, why did I get 10?" is answerable from one block: a topic filter, a bm25
sort, and the two hard caps.

### D12 — `BudgetGuard` is the real bound, not the framework
[app/budget.py](app/budget.py) is checked at the entry of every node (both
engines) and caps LLM calls (≤5), retrieval iterations (≤2), revisions (≤1), pool
(≤100), validation batch (≤10), results (≤10) and a wall-clock deadline. A breach
raises `BudgetExceeded`; `run_workflow` catches it, records the reason, and still
returns a composed (partial) `AgentResponse` with `stop_reason` set — never an
exception, never an unbounded loop. LangGraph's `recursion_limit` is only a
secondary backstop.

### D13 — Custom driver first, LangGraph second, same signature
[app/workflow.py](app/workflow.py) ships a ~40-line bounded driver; [app/graph.py](app/graph.py)
is the LangGraph port using the *same* node functions and guard, with `RunState`
carried under one key so there are no partial-dict reducers to reason about.
`cfg.engine` selects (`driver` default); a test asserts both produce identical
results. The JD names LangGraph as a required skill — this demonstrates it while
keeping a framework-free fallback.

### D14 — Deterministic preference ranking + capability gate
The LLM's `relevance_score` proved weak at ordering. Structured preferences
(founding-year, headcount, revenue) are scored deterministically in
`_preference_fit` and drive both the retrieval pool re-rank (so
preference-matching companies reach validation instead of being truncated by
bm25) and the final sort. The AND/OR logic over capabilities also lives in code:
the model reports per-capability support, and `validate_and_rank` applies "at
least one of" / "all of" — it never asks the model to do set logic.

---

## Evaluation philosophy

Development iterates against `eval/dev_queries.yaml` (10 self-authored queries).
The required 5 (+ supplements S1/S3) in `eval/queries.yaml` are **held out** — run
once at the end of M3 as a direction check, then not again until the final M5
eval — so the reported numbers measure generalisation, not fitting to the given
strings.

A **successful result** = all mandatory filters satisfied, ≥1 grounded evidence
item per returned company, no fabricated companies, bounded execution respected,
and a correct *empty* result when the data genuinely has no match (Q5). A correct
abstention scores as PASS.

---

## What was intentionally left out (so far)

Tracked here as milestones close. Nothing cut yet — M1 only.

## What I'd do next

See `PLAN.md` §C10 for the full "how it would evolve" discussion (Elasticsearch +
vector DB, MCP-wrapped tools, self-hosted LLMs, prompt/model versioning,
data lineage & audit).

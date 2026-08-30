# Agentic Company-Search System

Natural-language company mandate → ranked, evidence-backed shortlist over a
50,000-row synthetic dataset. Built for the Comparables.ai technical assessment.

> **Status:** M1 (data foundation) complete. Milestones M2–M5 in progress — see
> `PLAN.md` (not committed) for the full roadmap.

---

## Quick start

```bash
python -m pip install -r requirements.txt

# Build the retrieval index (SQLite + FTS5) from the raw dataset.
python -m app.cli ingest

# Sanity-check it with raw SQL.
python -m app.cli db "SELECT industry, COUNT(*) FROM companies GROUP BY industry"

# Run the test suite (offline, no API tokens).
python -m pytest
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
| `app/db.py` | Read-only DB access, `StructuredFilters` → SQL, `count_matching`. |
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

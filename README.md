---
title: Agentic Company Search
emoji: 🔎
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Agentic Company-Search System

A natural-language company mandate → a ranked, evidence-backed shortlist over a
50,000-row synthetic dataset. Built for the Comparables.ai technical assessment.

The system is a **deterministic workflow**, not an autonomous agent: seven nodes,
of which exactly two make a single schema-constrained LLM call each; every routing
decision is made by Python, never by a model.

```
interpret_mandate ─► build_search_plan ─► check_feasibility ─► retrieve ─► validate_and_rank ─► compose_response
   (LLM #1)            (deterministic)      (deterministic)     (tools)      (LLM #2/#3)          (deterministic)
                                                 │                              │
                              0 mandatory matches ┘         not enough results ─┘─► relax_preferences ─► retrieve …
                                 → empty + reason              (≤1 revision, deterministic ladder)
```

---

## Quick start

```bash
python -m pip install -r requirements.txt
cp .env.example .env            # add your OPENAI_API_KEY

python -m app.cli ingest        # build data/index/companies.db (SQLite + FTS5)
python -m pytest                # 152 passing, offline, no API tokens

python -m app.cli run "Finnish fintech doing fraud detection, prefer <250 employees" \
  --provider openai --verbose --explain

python -m app.cli eval --provider openai       # the assessment table -> eval/RESULTS.md
```

HTTP API:

```bash
uvicorn app.api:app --port 8000
curl -s -X POST localhost:8000/agent/search \
  -H 'X-API-Key: <AGENT_API_KEY>' -H 'content-type: application/json' \
  -d '{"query":"UK fintech doing payments or lending"}'
```

Docker (self-contained — builds the index at image-build time):

```bash
docker build -t compara-search .
docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... -e AGENT_API_KEY=secret compara-search
```

---

## The dataset, and what it forces

`data/companies.json` — 50,000 rows, profiled reproducibly by
`python -m app.profile_dataset` → `data/PROFILE.md`.

| Field | Reality |
|---|---|
| `industry` | **10 labels only** — no "Cybersecurity" |
| `location` | **8 countries only**: Finland, Germany, France, Norway, Sweden, Netherlands, USA, UK |
| `revenue_range` | 6 buckets, skewed large (~1,400 companies below €10M) |
| `employee_count` | 5 – **5000** (5000 is the maximum) |
| `founded_year` | 1995 – 2024 |
| `description` | one templated sentence: `"{AI-powered\|data-driven\|cloud-native} {platform\|engine\|software} for {topics}."` — 1,308 distinct strings over 50k rows |

Consequences that shaped the design:

- **Semantic search is low-value here.** Descriptions collapse to a handful of
  topic keywords, so FTS5 keyword matching over the structurally-filtered set
  captures nearly all of the signal. No embeddings (the path is described below).
- **Domain terms are not industries.** "drug-discovery companies" → `industry =
  Biotech` **plus** a topic filter. The interpret step emits both; the plan
  validates the industry against the 10-label enum.
- Relative terms ("Nordic", "after 2015", "below €10M") are resolved to concrete
  values **deterministically** — `regions.yaml`, a revenue-bucket parser — never
  by the model.
- **Evidence is bounded by the source.** A one-sentence description often can't
  support "serves European banks". Groundedness leans on a deterministic
  span-check; expect responses where `evidence` is one span and the rest is
  clearly-labelled `inference`.

---

## Implemented workflow

| Node | Kind | What it does |
|---|---|---|
| `interpret_mandate` | **LLM #1** | NL mandate → `MandateCriteria` (mandatory / preferences / exclusions, resolved relative terms, `industries` + `capabilities`, ambiguities). Two few-shot examples; strict `response_format`. |
| `build_search_plan` | deterministic | `MandateCriteria` → `SearchPlan`. Validates industries against the enum, resolves geography through `regions.yaml` ("United Kingdom" → "UK"), normalises capability phrases via the lexicon ("fraud detection technology" → "fraud detection"), normalises British spelling ("optimisation" → "optimization"), builds the revision ladder. |
| `check_feasibility` | deterministic | `count_matching(mandatory structured filters)`. **0 → short-circuit** to an empty result + reason, spending zero further LLM calls (this is how Q5 stays cheap and correct). |
| `retrieve` | tools | `search_companies`: mandatory filters as a hard SQL `WHERE`, then FTS5 bm25 over ORed topic phrases, then the exclusion gate. Pool re-ranked by deterministic preference fit so preference-matching companies reach validation. Pool ≤ 100. |
| `validate_and_rank` | **LLM #2 (#3)** | One batched call for ≤10 candidates. The model reports, per candidate, whether each capability / "serves" phrase is **supported + a verbatim quote** — it does *not* decide the verdict. Then deterministic post-processing (below). |
| `relax_preferences` | deterministic | A fixed ladder (widen founding-year → drop it → drop the employee band → enlarge the pool). Mandatory criteria and exclusions are never touched. ≤ 1 revision. |
| `compose_response` | deterministic | Hydrates every returned company's fields **from the DB via `get_by_ids`** (never from LLM output), assembles `AgentResponse` with `interpreted_mandate`, `search_plan`, ranked `results`, `revision`, and `metadata` (including the funnel). |

**LLM-call budget**: infeasible = 1, typical = 2, revision run = 3. Hard cap 5.

### Why a workflow, not agents

The brief invites this — *"the number of agents is not an evaluation criterion"*,
*"must not be able to enter an uncontrolled reasoning or tool-use loop"*, 20% on
bounded recovery, 25% on groundedness. It is still an agent loop in the
meaningful sense (plan → act → observe → adapt); we gave the loop a
**deterministic controller** instead of an LLM one. By Anthropic's "Building
Effective Agents" taxonomy this is a *workflow*, and that guidance says workflows
win for well-defined tasks — this one is. An LLM-driven planner would be
warranted with a large tool space where the strategy can't be determined up
front; here there are three tools and the strategy is derivable from the parsed
mandate.

### Why LangGraph (and a framework-free fallback)

The job description names LangGraph as a required skill, so demonstrating it is
on-target. The system was built and shipped first on a **~40-line custom driver**
([app/workflow.py](app/workflow.py)); the LangGraph engine ([app/graph.py](app/graph.py))
is a port on top, using the *same* node functions and the *same* `BudgetGuard`,
with `RunState` carried under one state key so there are no partial-dict reducers
to reason about. `cfg.engine` selects (`driver` is the default); a test asserts
both engines produce identical results. LangGraph's `recursion_limit` is only a
secondary backstop — the guard is the real bound.

---

## Structured state

One Pydantic model, `RunState` ([app/state.py](app/state.py)), is passed between
nodes. Every LLM output has its own **narrow, purpose-built** schema — no open
`dict` fields, safe for OpenAI strict mode, and never a reused storage model:

- `MandateCriteria` — interpret output. Mandatory / preferences share
  `MandateConstraints`; `capabilities_any` vs `capabilities_all` carries the
  any/all distinction.
- `ValidationBatch` → `CandidateJudgement` — validate output. Carries **no
  company fields**; IDs and record data are joined back in code afterwards.
- `SearchPlan`, `FeasibilityResult`, `RankedCompany`, `AgentResponse`,
  `RunTrace` (with the narrowing `Funnel`) — internal, never sent to the model.

---

## Tool contracts

Three tools ([app/tools.py](app/tools.py)), each a pure function with an explicit
Pydantic input/output schema, **deterministic, zero LLM tokens**:

| Tool | Input | Output | Notes |
|---|---|---|---|
| `search_companies` | `StructuredFilters`, `topic_terms`, `Exclusions`, `limit` | `SearchResult` (candidates + `matched_filters` / `matched_query` / `excluded` / `fts_query` / `truncated`) | 1. mandatory filters → SQL `WHERE` (hard gate). 2. FTS5 bm25 on ORed topic phrases within that set. 3. exclusion gate (industry `NOT IN` + keyword substring). 4. truncate to `limit` (≤ 100). |
| `count_matching` | `StructuredFilters` | `int` | Feasibility gate + revision reasoning. Mandatory filters only — exclusions are **not** applied (feasibility is about whether the mandate itself can be met). |
| `get_by_ids` | `list[int]` | `list[Company]` | Order-preserving hydrate + region lookup. Evidence lookup and final response assembly. |

There is **no parameter** through which a *preference* could become a hard
filter — `search_companies` only accepts `StructuredFilters` (mandatory). Enforced
by the type.

---

## Retrieval & filtering strategy

`companies.json` → **SQLite** (`app.ingest`, committed source, index rebuilt at
image-build time): typed columns, derived `revenue_min/max_eur`, a
`company_regions` lookup so region filters are a plain JOIN, and an **FTS5**
(bm25, porter-stemmed) virtual table over `name + description`.

- **Structured filters** are SQL `WHERE` predicates — the primary recall gate.
- **Keyword / topic** is FTS5 bm25 over the filtered id-set — the primary rank
  signal, given the templated text.
- **Preferences** are never filters. They're scored deterministically
  (`_preference_fit`) and used twice: to re-rank the retrieval pool (so a
  preference-matching company at bm25 rank 60 still reaches validation) and in
  the final sort.

**Why no embeddings.** Profiling showed keyword ≈ semantic on this data, and
FTS5 is one fewer dependency to build and defend. For real data the README's
"how it scales" section describes the local-ONNX re-rank and, at scale,
Elasticsearch + a vector DB.

### The narrowing funnel

Every run reports where the candidates go (`--verbose`, the `run.funnel` log
line, `metadata.funnel`):

```
mandatory_filters → topic_match → after_exclusions → retrieved_pool (≤100)
  → sent_to_validation (≤10) → passed_validation → returned (≤10)
```

So "580 companies matched, why did I get 10?" is one block: a topic filter, a
bm25 sort, and two hard caps.

---

## Validation, ranking & groundedness

`validate_and_rank` hands the model the full record plus **what is already known**
— which mandatory criteria are structurally satisfied, which need a text
judgement, which preferences the record matches — so it spends its effort on the
text asks, not on re-deriving facts we hold. A test asserts these signals are in
the rendered prompt.

The model returns per-capability / per-"serves" `TextFinding`s (`supported` +
`quote`). Then **deterministic** post-processing:

- **Span-grounding check**: every `quote` must be a literal substring of the named
  field of the *real* record (via `get_by_ids`). A quote that fails is demoted
  from `evidence` to `inference`. *A cited quote is an LLM claim until verified.*
- **Capability gate**: `capabilities_any` → satisfied if ≥ 1 supported;
  `capabilities_all` → all supported. This set logic lives in code, never in the
  prompt. (An FTS match rescues a capability the model *omitted*, never one it
  explicitly rejected.)
- **Verdict** is computed: `match` if capability gate passes and all "serves"
  pass; `partial` if capabilities pass but "serves" can't be verified from the
  text; drop otherwise.
- **Sort** by `(verdict, preference_score, relevance_score, mandatory_met)`.

`evidence[]` (verified quotes) and `inferences[]` (model claims + stated basis)
are always split in the response.

---

## Search-revision & failure recovery

- `check_feasibility` → 0 mandatory matches → empty result + which criteria
  failed, **no wasted LLM calls**.
- Otherwise, if fewer than `min_results` usable results (match + partial) after
  iteration 1 and no revision yet → `relax_preferences` (a deterministic ladder,
  preferences only) → one more retrieve + validate. Bounded: ≤ 1 revision, ≤ 2
  iterations.
- `BudgetGuard` ([app/budget.py](app/budget.py)) is checked at every node entry on
  both engines: LLM calls ≤ 5, iterations ≤ 2, revisions ≤ 1, pool ≤ 100, batch
  ≤ 10, results ≤ 10, wall-clock deadline. A breach raises `BudgetExceeded`; the
  runner records `stop_reason` and still returns a composed partial response —
  never an exception, never an unbounded loop.

On this dataset revision rarely triggers from a real query string (templated
descriptions keep most candidates verifiable), so the path is proven by a unit
test that scripts an empty first validation pass
(`test_workflow.test_revision_path_runs_once`).

---

## LLM & model choices

- **`openai:gpt-4o-mini`, single provider.** ~$1–3 for the whole assessment;
  cheap enough that rate-limit management and "$0-equivalent" cost reporting
  aren't worth it. Pricing verified against live docs (Aug 2026): $0.15 / $0.60
  per 1M tokens.
- **`LLMClient.complete(messages, response_model)`** is the only entry point the
  nodes use — a ~200-line wrapper, every line explainable, not a LangChain model
  class. `chat.completions.parse` with a Pydantic model as strict
  `response_format`.
- **Fake provider** backs the entire test suite — schema-valid dummies, zero
  tokens, offline. Only `tests/test_live_smoke.py` (opt-in, `RUN_LIVE=1`) and
  `app.cli eval --provider openai` hit the paid API.
- **Response cache** namespaced by provider identity (`fake` vs
  `openai:gpt-4o-mini`) so a fake answer is never served for a real call.
- **One repair retry** on schema-validation failure, then a structured
  `LLMError`. A 429 gets one `Retry-After` backoff, then fails gracefully.
- The `Provider` protocol is the seam toward multi-provider fallback /
  self-hosted models (see "how it scales").

---

## Latency & cost observations

From `app.cli eval --provider openai --no-cache` (7 queries, real API):

| | typical run | infeasible (Q5) |
|---|---|---|
| LLM calls | 2 | 1 |
| tokens | ~4,700 in / ~1,400 out | ~1,750 in / ~200 out |
| est. cost | **~$0.0015** | **~$0.0004** |
| latency | 6–14 s | ~2 s |

A full 7-query eval pass is **~$0.009**. The dominant cost is the validation
call's prompt (10 candidate records). `--no-cache` gives real numbers; with the
cache, dev re-runs are near-instant and free.

---

## Evaluation

Development iterates against `eval/dev_queries.yaml` (20 self-authored queries).
The required 5 + supplements S1/S3 in `eval/queries.yaml` are the scored set, each
with a **hand-authored `expected` block** — the "correct parse" oracle.

**Honest note on held-out purity:** the intent was to run `queries.yaml` once as a
direction check and not again until this final pass. In practice I exercised it
while debugging M4, so the strict "look once" methodology is partly spent — but
every prompt/code change was driven by *general* correctness (the mandatory /
preference split, capability OR-logic, geography canonicalisation, British
spelling), not by fitting the expected strings. `dev_queries.yaml` carried the
real iteration load.

`app.cli eval` scores the **deterministic** checks and writes
[eval/RESULTS.md](eval/RESULTS.md):

| Check | How |
|---|---|
| parsed criteria correct | `interpreted_mandate` / `search_plan` vs `expected` |
| mandatory filters applied | re-check every returned company against parsed criteria, using the DB |
| all returned companies exist | id set-membership against the dataset |
| evidence traceable | re-run the span-grounding check on every `evidence.quote` |
| revision performed == expected | `metadata.revised_search_performed` |
| correct abstention (Q5) | `results == []` + reason present — **scored as PASS** |
| budget respected | `llm_calls ≤ 5`, `iterations ≤ 2`, `revisions ≤ 1` |
| ranking / exclusion / evidence *sufficiency* | manual judgment column |

**Latest pass: 7/7 green** on the deterministic checks.

Needs human judgment (not auto-scored): ranking quality, faithfulness to nuance,
category-exclusion correctness, and whether the evidence is *sufficient* (not just
present).

---

## Deployment

**Docker, self-contained.** `python -m app.ingest` runs at image-build time, so
the container carries the 50k index and starts instantly. Runs as a non-root user
on port 7860.

```bash
docker build -t compara-search .
docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... -e AGENT_API_KEY=secret compara-search
curl -s localhost:7860/health
```

**Render.com (free tier).** [render.yaml](render.yaml) is a Blueprint for the
same Dockerfile:

1. Render dashboard → *New → Blueprint* → connect this repo (or *New → Web
   Service* → Docker).
2. Add `OPENAI_API_KEY` and `AGENT_API_KEY` under *Environment* (they are
   `sync: false` in the blueprint, i.e. never committed).
3. Deploy. Render builds the image (index baked in) and serves at
   `https://<name>.onrender.com`; `/health` is the health check.

The free instance spins down after ~15 min idle → the next request cold-starts in
~30–60 s. 512 MB RAM is sufficient.

**Alternatives.** *Hugging Face Spaces* — the README front-matter already
declares `sdk: docker`, `app_port: 7860`, so a push to a Docker Space works, but
HF now gates the Docker SDK behind PRO. *Google Cloud Run*
(`gcloud run deploy --source .`, scale-to-zero, generous free tier) — card
required. *Vercel* — analysed and rejected (below).

**Vercel — analysed and rejected:** serverless execution-time limits fight a
bounded-but-slow agent run; no persistent process for run state; ~250 MB bundle
cap. It forces sync-only + a tight timeout + external state for no benefit over a
persistent container.

---

## Repository layout

| Path | What |
|---|---|
| `app/ingest.py` | Build `companies.db`: typed columns, region lookup, FTS5, manifest. |
| `app/tools.py` · `app/schemas.py` | The three retrieval tools + their Pydantic contracts. |
| `app/llm.py` | `LLMClient` + `OpenAIProvider` / `FakeProvider`, response cache, cost table. |
| `app/state.py` · `app/prompts.py` | Workflow schemas / `RunState` · the two prompt builders. |
| `app/nodes.py` | The 7 nodes as `RunState → RunState` functions. |
| `app/budget.py` · `app/workflow.py` · `app/graph.py` | `BudgetGuard` · custom driver · LangGraph engine. |
| `app/api.py` · `app/cli.py` · `app/logging.py` | FastAPI · CLI · structured JSON logging. |
| `app/evaluate.py` | Assessment eval harness → `eval/RESULTS.md`. |
| `app/config.py` · `app/*.yaml` | Taxonomy config: regions, lexicon, schema map, revenue buckets. |
| `eval/queries.yaml` · `eval/dev_queries.yaml` | Scored set (with `expected`) · dev iteration set. |
| `tests/` | 152 tests, offline, against the fake provider + a fixture index. |

CLI: `ingest` · `db "<sql>"` · `search` (tools, no LLM) · `check` (interpret +
plan + feasibility) · `run` · `eval`.

---

## Known limitations

- **`relevance_score` from the model is weak** — often a flat 0.5. Ordering is
  really driven by the deterministic `preference_score` then bm25. Fine here;
  a real system would calibrate or replace it.
- **Recall is bm25-top-100.** Topic-matching companies past bm25 rank 100 never
  reach validation. On this templated data the top 100 almost certainly contains
  every real match; on real data this needs a bigger pool + embeddings.
- **"serves X" is usually unverifiable** from a one-sentence description, so Q3-
  style queries return `partial`, not `match` — correct, but the ceiling is the
  source, not the system.
- **Revision rarely fires naturally** on this dataset (see above) — the path is
  real and tested, just hard to trigger from a query string.
- Single provider, sync API, in-memory (no) run store — all deliberate for the
  time box; the seams to change each are called out below.

## What was intentionally left out for time

- **Embeddings / semantic re-rank** — profiling said it wouldn't move the needle
  here; the path is documented.
- **`GET /agent/runs/{id}` + a `RunStore`** — the API is stateless. A `RunStore`
  protocol (in-memory dict now) is the ~20-line seam to add it.
- **Async API (`202` + status endpoint)** — sync with a 90 s deadline is safe
  behind a persistent container.
- **A UI** — the CLI is the primary surface; the JSON API is the integration
  surface.

## What I'd do next (how it scales)

- **350M+ companies:** retrieval → **Elasticsearch** (structured filters + BM25)
  **+ a vector DB** (Qdrant/Weaviate) with ANN, hybrid-ranked; tiered filtering
  (cheap structured gate → ANN → LLM validation); sharding by region/sector;
  embeddings precomputed in a batch pipeline.
- **Concurrent runs:** stateless workers behind a queue; Postgres run store +
  Redis locks; per-run budget isolation; provider rate-limit pooling.
- **OSS / self-hosted LLMs:** the `LLMClient` seam → vLLM serving Llama/Mistral,
  grammar-constrained decoding for structured-output reliability.
- **Production eval:** labelled golden sets, retrieval recall@k, an LLM-judge
  calibrated to human labels, a per-prompt/model regression gate in CI.
- **Lineage & audit:** immutable run store; every evidence item points to
  `(company_id, source_field, record_version)`; prompts + model versions retained
  per run.
- **MCP:** the tool contracts are already Pydantic in/out — wrap them as an MCP
  server so the same tools serve other agents unchanged.

---

## Decision Log

<details>
<summary>The reasoning on record, in build order (D1–D14).</summary>

**D1 — Deterministic workflow, not autonomous agents.** The brief rewards it; the
strategy is derivable from the parsed mandate, so an LLM planner adds failure
modes without capability.

**D2 — SQLite + FTS5.** Makes structured filters *be* SQL, bm25 for free,
`count_matching` a one-liner; stdlib only. Alternative (README): ES + vector DB.

**D3 — FTS5 only, no embeddings.** 1,308 distinct templated descriptions →
keyword ≈ semantic. Alternative: `fastembed` local re-rank.

**D4 — Region & revenue resolution are deterministic config, never LLM.**
`regions.yaml`, a bucket parser; unknown region terms → recorded as ambiguity,
not guessed.

**D5 — `schema_map.yaml` indirection** so ingestion isn't hard-wired to 8 fields.

**D6 — Tools return `SearchResult`, not a bare list** — exclusion bookkeeping
(Q3 "matched nothing", S3 "removed N") must survive the tool boundary.

**D7 — Preferences can never reach a retrieval tool** — enforced by the type.

**D8 — One `LLMClient` seam; fake provider is the test default.** Repair retry,
`Retry-After` backoff, provider-namespaced cache.

**D9 — Few-shot the mandatory/preference split.** gpt-4o-mini kept classifying
"preferably founded after 2018" as mandatory and emitting `0` for unset bounds;
two worked examples fixed both. Cheaper than escalating to `gpt-4o`.

**D10 — Groundedness is a deterministic substring check, not a prompt
instruction.** Failed quote → demoted to inference.

**D11 — Revision is a deterministic ladder, bounded to 1.** Q4 specifies the rule
itself.

**D12 — `BudgetGuard` is the real bound, not the framework.** Checked at every
node entry, both engines; breach → partial response, never an exception.

**D12a — The narrowing funnel is reported explicitly** (`RunTrace.funnel`).

**D13 — Custom driver first, LangGraph second, same signature.** A test asserts
both engines agree.

**D14 — Deterministic preference ranking + capability gate.** The model reports
per-capability support; the AND/OR logic and the verdict are computed in code.

</details>

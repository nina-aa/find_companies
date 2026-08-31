# Agentic Company-Search System

A natural-language company mandate → a ranked, evidence-backed shortlist over a
50,000-row synthetic company dataset.

It's a **deterministic workflow**, not an autonomous agent: seven nodes, two of
which make one schema-constrained LLM call each. Every routing decision, filter,
score and bound is Python — the model parses the request and judges text, and
that's all it does.

```
interpret_mandate ─► build_search_plan ─► check_feasibility ─► retrieve ─► validate_and_rank ─► compose_response
   (LLM)              (code)               (code)              (tools)     (LLM)                (code)
                                                │                            │
                             0 mandatory matches ┘      too few results ─────┘─► relax_preferences ─► retrieve …
                                → empty + reason         (≤1 revision; rarely triggers on this dense data)
```

---

## Repository layout

| Path | What |
|---|---|
| `app/ingest.py` · `app/profile_dataset.py` | build `companies.db` (typed columns, FTS5, region lookup, manifest) · profile the data → `PROFILE.md` + `dataset_vocab.json` |
| `app/tools.py` · `app/schemas.py` | the three retrieval tools + their Pydantic contracts |
| `app/llm.py` · `app/prompts.py` | `LLMClient` + providers + cache · the two prompt builders |
| `app/state.py` · `app/nodes.py` | all schemas + `RunState` · the 7 nodes as `RunState → RunState` |
| `app/budget.py` · `app/workflow.py` · `app/graph.py` | `BudgetGuard` · the driver engine · the LangGraph engine |
| `app/api.py` · `app/cli.py` · `app/logging.py` | FastAPI · CLI · structured JSON logging |
| `app/evaluate.py` | the evaluation harness → `eval/RESULTS.md` |
| `app/config.py` · `app/*.yaml` | deterministic taxonomy: regions, lexicon, schema map |
| `eval/queries.yaml` · `eval/dev_queries.yaml` | scored set (with `expected`) · dev iteration set |
| `tests/` | 207 offline tests (fake provider + a fixture index) + opt-in live smoke |

CLI: `ingest` · `db "<sql>"` · `search` (tools, no LLM) · `check` (interpret +
plan + feasibility) · `run` · `eval`.

---

## Quick start

```bash
python -m pip install -r requirements.txt
cp .env.example .env            # add your OPENAI_API_KEY

python -m app.cli ingest        # build data/index/companies.db (SQLite + FTS5)
python -m pytest                # 207 passing, offline, no API tokens

python -m app.cli run "Finnish fintech doing fraud detection, prefer <250 employees" \
  --provider openai --verbose --explain

python -m app.cli eval --provider openai       # the evaluation table -> eval/RESULTS.md
```

HTTP API:

```bash
uvicorn app.api:app --port 8000
curl -s -X POST localhost:8000/agent/search \
  -H 'content-type: application/json' \
  -d '{"query":"UK fintech doing payments or lending"}'
```

Docker (self-contained — builds the index at image-build time):

```bash
docker build -t compara-search .
docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... compara-search
```

---
## Deployment

**Docker, self-contained.** `python -m app.ingest` and `python -m app.profile_dataset`
run at image-build time, so the container carries the 50k index and starts
instantly. Non-root, listens on `$PORT` (default 7860).

```bash
docker build -t compara-search .
docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... compara-search
curl -s localhost:7860/health
```

**Render.com (free tier).** Dashboard → **New → Web Service** → connect the repo →
runtime **Docker** (auto-detected) → instance type **Free**. Under *Environment*
add `OPENAI_API_KEY` (and `AGENT_API_KEY` if you want the endpoint gated). The
image bakes the index in, so no persistent disk is needed. `render.yaml` is kept
as a config reference. The free instance spins down after ~15 min idle → the next
request cold-starts in ~30–60 s.

**Alternatives.** *Google Cloud Run* (`gcloud run deploy --source .`) — card
required. *Vercel* was rejected: serverless execution-time limits fight a
bounded-but-slow run, there's no persistent process for run state, and the bundle
cap is tight — sync-only + external state for no gain over a container.

---

## The dataset shapes the design

`data/companies.json` — 50,000 rows. `python -m app.profile_dataset` regenerates
`data/PROFILE.md` and `data/dataset_vocab.json`.

| Field | Reality |
|---|---|
| `industry` | 10 labels only  |
| `location` | 8 countries: Finland, Germany, France, Norway, Sweden, Netherlands, USA, UK |
| `revenue_range` | 6 buckets, skewed large (~1,400 companies below €10M) |
| `employee_count` | 5 – 5000 (5000 is the max) |
| `founded_year` | 1995 – 2024 |
| `description` | one templated sentence: `"{adjective} {noun} for {topic}."` — 8 filler adjectives, 6 filler nouns, and **27 topic phrases** that carry all the meaning |

Three consequences:

- **Keyword search is enough.** Every description reduces to one of 27 topics, so
  FTS5 keyword matching over the structurally-filtered set captures the signal.
  No embeddings — see "Semantic retrieval" under "Limitations & what's next" for
  where they'd earn their place.
- **All 27 topics are mapped.** [app/lexicon.yaml](app/lexicon.yaml) covers every
  one, and a parametrised test fails if it drifts out of sync with the data. The
  interpret prompt also carries the list, so free-text phrasing resolves at parse
  time ("cancer research" → `["drug discovery", "molecular analysis"]`); the
  lexicon is the deterministic backstop.
- **Relative terms are resolved in code, never by the model** — "Nordic" →
  FI/NO/SE via `regions.yaml`, "below €10M" → two revenue buckets via a parser.
  An unknown term becomes a recorded ambiguity, not a guess.

---

## How it works

| Node | Kind | What it does |
|---|---|---|
| `interpret_mandate` | **LLM** | query → `MandateCriteria`: mandatory / preference / exclusion split, phrasing mapped onto the 10 industries + 27 topics, relative terms resolved, `serves` phrases and `ambiguities` extracted. 5 few-shot examples, strict `response_format`. |
| `build_search_plan` | code | `MandateCriteria` → `SearchPlan`: validate industries against the enum, resolve geography, normalise capability phrasing through the lexicon, build the revision ladder. |
| `check_feasibility` | code | `count_matching(mandatory filters)`. **0 → stop here** with an empty result + reason. No further LLM calls. |
| `retrieve` | tools | `search_companies`: mandatory filters as a hard SQL `WHERE`, then FTS5 bm25 on the topic phrases, then the exclusion gate. A second, preference-filtered search is merged to the front so preference-perfect rows survive the pool cap (≤ 100). |
| `validate_and_rank` | **LLM** | one batched call over the top ≤ 10 candidates — per capability / `serves` phrase: "supported? + a verbatim quote". It does **not** score or rank. Then deterministic scoring + sort over the whole pool. **Skipped** (0 LLM calls) when the mandate is purely structural. |
| `relax_preferences` | code | a fixed ladder — widen the founding-year preference, then drop it, then drop the employee band, then enlarge the pool. Mandatory criteria untouched. ≤ 1 revision. |
| `compose_response` | code | hydrate every returned company from the DB (never from LLM output); assemble `AgentResponse`. |

**LLM calls per run:** 1 (infeasible or purely structural), 2 (typical), 3 (with a
revision). Hard cap 5, enforced by `BudgetGuard`.

### Structured state

One Pydantic model, `RunState` ([app/state.py](app/state.py)), passes between
nodes. Each LLM call has its own narrow schema — no open `dict` fields, safe for
strict mode, never a reused storage model:

- `MandateCriteria` — the parse. `capabilities_any` vs `capabilities_all` carries
  the OR/AND distinction.
- `ValidationBatch` → `CandidateJudgement` — the text judgement. Carries **no
  company fields**; ids and records are joined back in code.

Everything else (`SearchPlan`, `RankedCompany`, `AgentResponse`, `RunTrace`) is
internal and never sent to the model.

### The three tools

[app/tools.py](app/tools.py) — pure functions, explicit Pydantic in/out,
deterministic, zero LLM tokens:

| Tool | In → Out | What |
|---|---|---|
| `search_companies` | `StructuredFilters`, `topic_terms`, `Exclusions`, `limit` → `SearchResult` | mandatory filters → SQL `WHERE`; FTS5 bm25 on ORed topic phrases within that set; exclusion gate (`industry NOT IN` + keyword substring); truncate to `limit` (≤ 100). Returns candidates + counters (`matched_filters`, `matched_query`, `excluded`, …). |
| `count_matching` | `StructuredFilters` → `int` | feasibility gate + revision reasoning. Mandatory filters only. |
| `get_by_ids` | `list[int]` → `list[Company]` | order-preserving hydrate + region lookup. Powers span-grounding and the final response. |

`search_companies` only accepts `StructuredFilters` (mandatory) — a *preference*
cannot become a hard filter, enforced by the type.

### A query, end to end

*"Find fintech companies in Finland working on fraud detection or banking
analytics. Prefer companies founded after 2015 with fewer than 250 employees."*

**1. `interpret_mandate` (LLM):**
```
mandatory:   countries=[Finland]  industries=[Fintech]
             capabilities_any=[fraud detection, banking analytics]
preferences: founded_year_gte=2016  employee_count_lte=249
```

**2. `build_search_plan` (code):** `Fintech` is in the enum; `Finland` is a
country; topic terms are the two capabilities; the preferences stay out of the
filter.

**3. `check_feasibility`:**
```sql
SELECT COUNT(*) FROM companies c
WHERE c.industry IN ('Fintech') AND c.location IN ('Finland');   -- 580 > 0, feasible
```

**4. `retrieve` → `search_companies`** (topic terms present, so the FTS join):
```sql
SELECT c.*, bm25(companies_fts) AS bm25_score
FROM companies_fts JOIN companies c ON c.id = companies_fts.rowid
WHERE companies_fts MATCH '"fraud detection" OR "banking analytics"'
  AND (c.industry IN ('Fintech') AND c.location IN ('Finland'))
ORDER BY bm25_score
LIMIT 100;
```
A second query adds `AND c.founded_year >= 2016 AND c.employee_count <= 249`; its
hits go to the front of the pool. (No topic terms → no FTS join, `ORDER BY c.id`.
A revenue / `serves` / exclusion clause adds more `AND …` predicates.)

**5. `validate_and_rank` (LLM):** the model sees the top 10 records and reports,
per company, whether the text supports "fraud detection" and "banking analytics",
with a quote. Code then applies the OR gate (keep if ≥ 1), computes
`score = 0.65·(mandatory fraction) + 0.35·(preference fraction)`, and sorts by
tier.

**6. `compose_response`:** company fields pulled from the DB by id, response
assembled.

---

## Orchestration: a workflow, not an agent

It is still an agent loop in the useful sense — plan (interpret), act (retrieve),
observe (validate), adapt (revise) — but the loop's controller is Python, not a
model. By Anthropic's "Building Effective Agents" taxonomy that makes it a
*workflow*, and the guidance there is that workflows win for well-defined tasks.
This one is well-defined: with 8 countries, 10 industries, 27 topics and 3 tools,
the retrieval strategy is fully derivable from the parsed mandate. An LLM-driven
planner would be warranted with a large, open-ended tool space where the right
next step genuinely can't be decided up front; here it would only add failure
modes — uncontrolled loops, unpredictable cost — for no extra capability. So the
control flow is hand-written and bounded by `BudgetGuard`; it cannot loop
uncontrollably.

**Two interchangeable engines**, selected by `cfg.engine`:

- **`driver`** (default) — a ~25-line loop in [app/workflow.py](app/workflow.py)
- **`graph`** — a LangGraph `StateGraph` in [app/graph.py](app/graph.py): the same
  seven node functions, the same `BudgetGuard`, `RunState` under one state key

`test_both_engines_agree` asserts identical output.

**LangGraph is overkill here.** This flow is almost linear — no
persistence, no human-approval gates, no parallel fan-out — nothing a framework
buys over a `for` loop. The driver is the real engine. I built the port because I
wanted hands-on time with LangGraph, and because it's a concrete check that the
node functions aren't coupled to the control flow: if this ever needed a
checkpointer or interrupt nodes (see "what's next"), the swap is already done and
tested. For now it's a curiosity and a seam, not a necessity.

---

## What the LLM decides — and what it doesn't

**The model does exactly two things:**

1. **Parses the request** (`interpret_mandate`) — natural language → the
   structured `MandateCriteria`.
2. **Judges text** (`validate_and_rank`) — for each candidate, per requirement:
   is it supported by the description, and what is the verbatim quote? Nothing
   more — the prompt says so explicitly ("you do not decide the ranking or a
   score").

**Everything else is deterministic Python:**

| | where |
|---|---|
| which companies are retrieved | SQL `WHERE` + FTS5 |
| "Nordic" → countries, "€10M" → buckets, industry-enum validation | `config.py` + the YAML tables |
| the capability AND/OR logic and the drop decision | `validate_and_rank` |
| `mandatory_met` / `mandatory_total`, the `score`, the tiered sort | `_finalise_ranking` |
| preference fit — company fields vs the preferences | `_preference_fit` |
| the revision trigger + the relaxation ladder | `relax_preferences` |
| span-grounding — a quote not literally in the record is demoted regardless of what the model said; the model's `source_field` label is not trusted | `_resolve_source_field` |
| every company field in the response | `get_by_ids`, never the model |

**So the model's output changes the result in only two ways:** it can **drop a
candidate** (says "doesn't do capability X" and an FTS keyword match doesn't
rescue it), and it can **raise `mandatory_met`** with a grounded `serves` quote
(rare — see below). It also fills `evidence[]` vs `inferences[]`, which is central
to groundedness but doesn't move the ranking.

There is **no LLM-assigned relevance score** — the ranking is the deterministic
tier sort described below.

---

## Retrieval, scoring & groundedness

`companies.json` → SQLite (`app.ingest`): typed columns, derived
`revenue_min/max_eur`, a `companies_fts` FTS5 table (bm25, porter-stemmed) over
name + description, a `company_regions` lookup.

**The funnel** — reported on every run (`--verbose`, `metadata.funnel`):

```
mandatory_filters → topic_match → after_exclusions → pool (≤100)
  → sent_to_validation (≤10) → passed_validation → returned (≤ result_limit, default 10)
```

That is the answer to "580 matched, why did I get 10": a topic filter, a bm25
sort, and two hard caps.

**Filters vs preferences.** Mandatory criteria are the hard SQL gate. Preferences
never filter — they drive the pool re-rank and the final sort. The extra
preference-filtered retrieval exists only so a preference-perfect company isn't
lost to the bm25 `LIMIT`.

**bm25's role.** bm25 is how FTS5 ranks a keyword match — the topic search runs
on it. But the descriptions are templated ("*predictive engine* for fraud
detection" vs "*automated software* for fraud detection"), so two companies
matching the same topic score almost identically and bm25 can't separate them.
So it does two narrow jobs: decide which ≤ 100 rows enter the pool when more
match (`ORDER BY bm25_score LIMIT 100`), and break an exact tie in the final
sort. The ranking you actually see is the deterministic tier sort below, not
bm25. On real, varied prose bm25 would carry real weight.

**Scoring.** Each result reports:

| field | meaning |
|---|---|
| `mandatory_met / total` | every mandatory requirement — location, industry, each numeric bound, each exclusion, the capability group, each `serves` phrase. Structural ones always hold; the total is the mandate's size. |
| `preferences_met / total` | soft preferences met; each miss named in `unmet_preferences` |
| `score` | `0.65·mand + 0.35·pref` — a readout, **not** the sort key |

The **sort is tiered**: `(all mandatory met?, #mandatory met, #preferences met)`,
then bm25 breaks an exact tie. No weights. When every returned row sits in one
tier that is larger than the returned slice, `results_are_top_ranked` is `false`
and `ranking_note` says so ("25 companies meet every requirement and preference;
the 10 returned are an arbitrary slice").

**Groundedness.** The model returns `{supported, quote}` per requirement; then:

- every quote is checked as a literal substring of the real record. Not found →
  demoted from `evidence[]` to `inferences[]`. *A cited quote is a claim until
  verified.*
- `evidence[]` (verified) and `inferences[]` (model claim + basis) are always
  separate in the response.

**`serves` phrases** — "provides X *to European banks*". These describe the
customer, not the company:

- they are a mandatory requirement (counted in `mandatory_total`)
- this dataset has **no customer field**, so they are almost never verifiable →
  the row lands at e.g. `3/4`, `full_match` is `false`, the gap goes to
  `inferences[]`, and `metadata.caveats` explains why
- a region word *inside* the phrase ("**European** banks") becomes a location
  *preference*, never a filter — the mandate never said where the company itself
  is based

---

## Bounded execution & recovery

- **`check_feasibility`** — 0 mandatory matches → an empty result naming the
  failed criteria, zero further LLM calls. This is the real "no results" path
  (Q5 hits it).
- **`relax_preferences`** — if too few candidates clear the capability gate after
  the first pass, relax the softest preference and retrieve once more.
  Implemented and unit-tested, but it **never fires on the evaluation queries**:
  this dataset is dense enough that any feasible mandate returns a full set. On
  sparse real data it would matter.
- **`BudgetGuard`** ([app/budget.py](app/budget.py)) — checked at the entry of
  every node on both engines: LLM calls ≤ 5, iterations ≤ 2, revisions ≤ 1, pool
  ≤ 100, validation batch ≤ 10, results ≤ 50 (default 10), wall-clock deadline. A
  breach returns a composed partial response with `stop_reason` set — never an
  exception, never an unbounded loop.

---

## HTTP API

`POST /agent/search` runs the full workflow synchronously and returns the same
`AgentResponse` the CLI produces. `GET /health` reports index status; `GET /docs`
is the generated OpenAPI UI (the authoritative schema). An `X-API-Key` header is
checked against `AGENT_API_KEY` **only when that env var is set** — wallet
protection for a public URL, not a security boundary.

```jsonc
// request
{ "query": "…", "min_results": 3, "limit": 10 }

// response (abridged — full schema at /docs)
{
  "run_id": "…",
  "query": "…",
  "interpreted_mandate": { "mandatory": {…}, "preferences": {…}, "exclusions": {…},
                           "semantic_focus": "…", "ambiguities": ["…"] },
  "search_plan": { "filters": {…}, "topic_terms": ["…"], "topic_mode": "any",
                   "preferences": {…}, "serves": ["…"], "notes": ["…"] },
  "results": [
    { "rank": 1, "company_id": 123, "name": "…", "industry": "…", "location": "…",
      "founded_year": 2019, "employee_count": 40, "revenue_range": "1M-10M",
      "mandatory_met": 3, "mandatory_total": 3, "preferences_met": 2,
      "preferences_total": 2, "score": 1.0, "llm_validated": true,
      "evidence":   [ { "requirement": "capability: fraud detection",
                        "source_field": "description", "quote": "…" } ],
      "inferences": [ { "claim": "serves: European banks", "basis": "…" } ],
      "unmet_preferences": [] }
  ],
  "revision": { "performed": false, "relaxed": [], "reason": "" },
  "ambiguities": ["…"],
  "empty_reason": "",
  "metadata": {
    "stages_executed": ["…"], "tools_called": [ { "name": "…", "ok": true } ],
    "funnel": {…}, "match_summary": {…}, "validation_outcome": {…},
    "revised_search_performed": false, "ranking_note": "", "caveats": ["…"],
    "llm_calls": 2, "prompt_tokens": 5000, "completion_tokens": 1200,
    "est_cost_usd": 0.0016, "latency_ms": 9000, "model": "gpt-4o-mini",
    "provider": "openai", "timed_out": false, "stop_reason": ""
  }
}
```

`BudgetGuard`'s per-run deadline (`AGENT_DEADLINE_S`, ~90 s) returns a partial
body with `timed_out: true` rather than hanging; there is no server-side HTTP
timeout, so clients should set their own above 90 s. An async `202` + status
endpoint is a documented cut.

---

## LLM client & cost

- **`openai:gpt-4o-mini`, one provider.** ~$1–3 to build and evaluate the whole
  thing. Pricing checked live (Aug 2026): $0.15 / $0.60 per 1M tokens.
- **`LLMClient.complete(messages, response_model)`** — the only entry point the
  nodes use. `chat.completions.parse` with a Pydantic model as strict
  `response_format`; a ~200-line hand-written wrapper, not a LangChain class. One
  repair retry on a schema-validation failure, then a structured `LLMError`; a
  429 gets one `Retry-After` backoff.
- **Fake provider** backs the whole test suite (schema-valid dummies, offline,
  zero tokens). Only `tests/test_live_smoke.py` (opt-in, `RUN_LIVE=1`) and
  `app.cli eval --provider openai` spend money.
- **Response cache** namespaced by provider identity, so a fake answer is never
  served for a real call.
- The `Provider` protocol is the seam for multi-provider / self-hosted models.

| | typical | infeasible (Q5) |
|---|---|---|
| LLM calls | 2 | 1 |
| tokens | ~5,000 in / ~1,200 out | ~1,800 in / ~200 out |
| cost | ~$0.0016 | ~$0.0005 |
| latency (local) | 6–14 s | ~2 s |

A full 7-query eval is ~$0.01. On the free-tier deploy, latency is 2–3× higher
and a cold start adds up to ~60 s.

---

## Evaluation

`eval/queries.yaml` — 5 core queries (qualitative + structured criteria;
mandatory vs preferred; exclusions + evidence; controlled revision; empty /
conflicting handling) + 2 supplements, each with a hand-written `expected` block.
Development iterated on the separate `eval/dev_queries.yaml` (20 queries);
`queries.yaml` was held back and looked at only a few times while debugging,
always for general correctness rather than to overfit the expected strings.

**A result is successful when:** every returned company satisfies the mandatory
structured criteria; each has ≥ 1 grounded evidence span for what the text can
support (gaps labelled, not hidden); no company is fabricated; every budget bound
held; and when the data has no real match the answer is an empty list + a reason,
not a padded one. Ranking quality within the "match" tier is a human call.

`app.cli eval` scores the deterministic checks → [eval/RESULTS.md](eval/RESULTS.md);
the full ranked walkthrough is [eval/eval6.txt](eval/eval6.txt).

| Check | How | |
|---|---|---|
| parsed criteria | `interpreted_mandate` / `search_plan` vs `expected` | deterministic |
| mandatory filters applied | re-check every returned company against the DB | deterministic |
| companies exist | id membership against the dataset | deterministic |
| evidence traceable | re-run span-grounding on every `evidence.quote` | deterministic |
| revision flag / abstention / budget | vs `expected` and the caps | deterministic |
| ranking quality, exclusion correctness, evidence *sufficiency* | manual note | judgment |

**Latest: 7/7 green** on the deterministic checks. Parse accuracy on the 20 dev
queries was 14/20 before prompt tuning (all 5 core queries correct); the misses
were vague colloquial phrasing, addressed by prompt rules.

---

## Limitations & what's next

**Known limitations**

- Ranking within a tier is only as good as the query's signal — no preference +
  templated text means everyone ties, and `ranking_note` says so. Real varied
  text would separate them.
- Recall is bm25-top-100; a topic match past rank 100 never reaches validation.
  Fine on this data, needs a bigger pool + embeddings on real data.
- `serves`-style requirements are unverifiable here (no customer field), so
  Q3-style rows top out at `partial`. The ceiling is the source, not the system.
- Revision is implemented and tested but never triggers on this dense dataset.
- Single provider, sync API, no run store — all deliberate; the seams are below.

**Cut for time:** embeddings / semantic re-rank; `GET /agent/runs/{id}` + a
persistent `RunStore`; async `202` + status endpoint; a UI.

**What I'd build next**

- **An independent parse check.** The interpret step is the highest-leverage
  failure point. Add a *second* LLM call — ideally a *different* model — that gets
  the original query and the assembled plan and answers "does this capture every
  hard requirement, and did it add a constraint the query doesn't state?",
  writing findings into `ambiguities` (never mutating the plan). A different model
  matters: the same one asked to check its own parse tends to defend it. A
  clause-by-clause parse trace in the response is the cheap partial version.
- **Semantic retrieval.** Right now retrieval is a structured SQL gate + FTS5
  keyword match — enough for 27 templated topics, weak the moment descriptions are
  real prose or a query uses wording the lexicon doesn't cover. The step up: embed
  each company's text and the query's `semantic_focus`, and blend a cosine
  re-rank into the pool ordering *after* the structured gate. `search_companies`
  already takes a `semantic_query` argument that is currently ignored — that's the
  hook. Start with a local model (`fastembed`, no torch) over the filtered subset;
  no ANN index needed at this size.
- **A much larger corpus** — once a linear scan over embeddings is too slow
  (millions+ of rows), the vector index moves to a dedicated ANN store
  (Qdrant / Weaviate / `pgvector`) and the structured filters to Elasticsearch,
  hybrid-ranked; tiered filtering (structured gate → ANN → LLM validation);
  embeddings precomputed in a batch pipeline; sharded by region / sector.
- **Concurrent runs** — the workflow is already a pure `run_workflow(query, cfg)`
  with no shared state: stateless workers behind a queue, a Postgres run store
  behind the `RunStore` seam, Redis for locks, provider rate-limit pooling.
- **Self-hosted LLMs** — the `Provider` protocol is the only seam; add a
  `VLLMProvider`. The open question is structured-output reliability without
  strict mode — grammar-constrained decoding (Outlines / GBNF) or a stricter
  repair loop.
- **Persistent workflow state** — LangGraph with a Postgres checkpointer: every
  node boundary a resumable savepoint, human-approval interrupts drop in as
  nodes. `RunState` is one Pydantic model, so it serialises as-is.
- **Production monitoring** — the JSON logs (already keyed by `run_id`) and the
  `RunTrace` counters → OpenTelemetry; dashboard funnel drop-off, p50/p95
  latency, cost per run, abstention rate, `stop_reason`; alert on drift.
- **Model & prompt versioning** — a prompt registry with content-hash version IDs
  stamped into `RunTrace`; any prompt or model change runs `eval/queries.yaml` as
  a CI gate against the previous version.
- **Production eval** — labelled golden sets, retrieval recall@k, an LLM-judge
  calibrated to human labels for the judgment columns, online feedback capture.
- **Data lineage & governance** — an immutable run store; every evidence item
  points to `(company_id, source_field, record_version)`; prompts + model
  versions retained per run; access policy at the tool boundary.
- **MCP** — the three tool contracts are already Pydantic in/out with no hidden
  state; wrap them as an MCP server so the same tools serve other agents
  unchanged.

---


## Design notes

<details>
<summary>The non-obvious choices.</summary>

- **Deterministic workflow, not an agent.** The retrieval strategy is derivable
  from the parsed mandate, so an LLM planner would add failure modes (loops,
  cost) without capability. It is still plan → act → observe → adapt — the
  controller is just Python.
- **SQLite + FTS5, no embeddings.** Structured filters *are* SQL; bm25 comes free;
  `count_matching` is one line; stdlib only. The 27 templated topics make keyword
  ≈ semantic. At scale: Elasticsearch + a vector DB.
- **Preferences are type-locked out of retrieval.** `search_companies` accepts
  only `StructuredFilters`, so a preference cannot become a hard filter even by
  mistake.
- **`serves` counts toward the mandate but is usually unmet** on this data (no
  customer field). Reporting `3/4` + a caveat is more honest than silently
  dropping the requirement or fabricating a match.
- **The validation call is skipped when there is nothing to judge** — a purely
  structural mandate is fully answered by the SQL gate, so the model would only
  rubber-stamp rows. 0 tokens.
- **`result_limit` (default 10, up to 50) is separate from the validation batch
  (10).** The batch bounds LLM cost; rows past rank 10 come back
  `llm_validated: false`, keyword-matched only.
- **The interpret prompt carries the 27-topic vocabulary** as a *soft* target, so
  synonyms map at parse time; a genuinely absent capability is left unmapped
  rather than forced onto a wrong topic.
- **Custom driver first, LangGraph second, same node functions.** A test asserts
  both engines agree. See "Orchestration" for why the framework is optional here.

</details>

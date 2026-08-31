# Agentic Company-Search System

A natural-language company mandate → a ranked, evidence-backed shortlist over a
50,000-row synthetic company dataset.

The system is a **deterministic workflow**, not an autonomous agent: seven nodes,
of which exactly two make a single schema-constrained LLM call each; every routing
decision is made by Python, never by a model.

```
interpret_mandate ─► build_search_plan ─► check_feasibility ─► retrieve ─► validate_and_rank ─► compose_response
   (LLM #1)            (deterministic)      (deterministic)     (tools)      (LLM #2/#3)          (deterministic)
                                                 │                              │
                              0 mandatory matches ┘      too few valid results ─┘─► relax_preferences ─► retrieve …
                                 → empty + reason         (≤1 revision, deterministic ladder; rare on dense data)
```

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
| `description` | one templated sentence: `"{adjective} {noun} for {topic}."` — 1,308 distinct strings over 50k rows. **8 adjective prefixes** (`AI-powered`, `data-driven`, …) and **6 noun heads** (`platform`, `engine`, `infrastructure`, `software`, `system`, `solution`) are pure filler; **27 core topic phrases** carry the meaning. |

Consequences that shaped the design:

- **Semantic search is low-value here.** Descriptions collapse to a handful of
  topic keywords, so FTS5 keyword matching over the structurally-filtered set
  captures nearly all of the signal. No embeddings (the path is described below).
- **The whole topic vocabulary is mapped.** All 27 core topics are in
  [app/lexicon.yaml](app/lexicon.yaml); `profile_dataset` writes
  `data/dataset_vocab.json` and a parametrised test
  (`test_lexicon.py::test_lexicon_covers_every_core_dataset_topic`, plus a
  `<topic> <noun-head>` variant) fails if the lexicon drifts out of sync with the
  data. Adjective prefixes and noun heads are stripped or ignored in parsing.
  The **interpret prompt is given the 27-topic list**, so free-text phrasing the
  lexicon doesn't contain still lands on a real topic at parse time
  ("cancer research" → `["drug discovery", "molecular analysis"]`); the lexicon is
  the deterministic backstop when the model keeps the raw phrase.
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
| `interpret_mandate` | **LLM #1** | NL mandate → `MandateCriteria` (mandatory / preferences / exclusions, resolved relative terms, `industries` + `capabilities` mapped onto the dataset's 10 industries + 27 topics, ambiguities). Five few-shot examples; strict `response_format`. |
| `build_search_plan` | deterministic | `MandateCriteria` → `SearchPlan`. Validates industries against the enum, resolves geography through `regions.yaml` ("United Kingdom" → "UK"), normalises capability phrases via the lexicon ("fraud-detection technology" → "fraud detection", hyphens and British spelling included), turns a region word inside a `serves` phrase ("European banks") into a location *preference*, builds the revision ladder. |
| `check_feasibility` | deterministic | `count_matching(mandatory structured filters)`. **0 → short-circuit** to an empty result + reason, spending zero further LLM calls (this is how Q5 stays cheap and correct). |
| `retrieve` | tools | `search_companies`: mandatory filters as a hard SQL `WHERE`, then FTS5 bm25 over ORed topic phrases, then the exclusion gate. A **second search filtered on the preferences too** is merged in (so preference-perfect rows aren't lost to the bm25 `LIMIT`), then the pool is re-ranked by preference fit. Combined pool ≤ 100. |
| `validate_and_rank` | **LLM #2 (#3)** | One batched call judges the top `validation_batch` (10) candidates — each capability / "serves" phrase **supported + a verbatim quote**; it does *not* decide the ranking or a score. The whole retrieval pool is then scored + tiered-sorted deterministically (see below); `result_limit` rows are returned, those in the batch `llm_validated: true`, any beyond keyword-matched only. **Skipped entirely** when the mandate is purely structural (no capability, "serves" or semantic exclusion) — 0 LLM calls, and every row `llm_validated: false` (nothing to check). |
| `relax_preferences` | deterministic | A fixed ladder (widen founding-year → drop it → drop the employee band → enlarge the pool). Mandatory criteria and exclusions are never touched. ≤ 1 revision. |
| `compose_response` | deterministic | Hydrates every returned company's fields **from the DB via `get_by_ids`** (never from LLM output), assembles `AgentResponse` with `interpreted_mandate`, `search_plan`, ranked `results`, `revision`, and `metadata` (including the funnel). |

**LLM-call budget**: infeasible = 1, purely structural = 1, typical = 2, revision
run = 3. Hard cap 5.

### Why a workflow, not agents

It is still an agent loop in the meaningful sense (plan → act → observe → adapt);
the loop just has a **deterministic controller** instead of an LLM one. By
Anthropic's "Building Effective Agents" taxonomy this is a *workflow*, and that
guidance says workflows win for well-defined tasks — this one is: the retrieval
strategy is fully derivable from the parsed mandate. An LLM-driven planner would
be warranted with a large, open-ended tool space where the right strategy can't be
determined up front; here there are three tools and one bounded revision, so an
LLM controller would add failure modes (uncontrolled loops, unpredictable cost)
without adding capability.

### Why LangGraph (and a framework-free fallback)

LangGraph maps cleanly onto a bounded state machine — explicit nodes, a couple of
conditional edges, no hidden control flow — which is exactly the shape here. The
system was built and shipped first on a **~40-line custom driver**
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
- **Keyword / topic** is FTS5 bm25 over the filtered id-set. On the templated
  descriptions bm25 barely varies, so it's used only for the pool `LIMIT` and as
  a final tie-break — not shown, not in the score.
- **Preferences are never filters.** But bm25 (near-random here) can truncate a
  preference-perfect company out of the pool `LIMIT`, so `retrieve` runs a
  **second search that also filters on the preferences** and merges its hits into
  the front of the pool (combined pool still ≤ 100). Preferences then drive the
  pool re-rank and the tiered sort.

**Why no embeddings.** Profiling showed keyword ≈ semantic on this data, and
FTS5 is one fewer dependency to build and defend. For real data the README's
"how it scales" section describes the local-ONNX re-rank and, at scale,
Elasticsearch + a vector DB.

### The narrowing funnel

Every run reports where the candidates go (`--verbose`, the `run.funnel` log
line, `metadata.funnel`):

```
mandatory_filters → topic_match → after_exclusions → retrieved_pool (≤100)
  → sent_to_validation (≤10) → passed_validation → returned (≤ result_limit, default 10)
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

- **Span-grounding check**: every `quote` is resolved to the *real* text field
  (`description` / `name`) it literally occurs in (via `get_by_ids`). A quote that
  occurs nowhere is demoted from `evidence` to `inference`, and the model's own
  `source_field` label is not trusted. *A cited quote is an LLM claim until
  verified.*
- **Capability gate**: `capabilities_any` → satisfied if ≥ 1 supported;
  `capabilities_all` → all supported. This set logic lives in code, never in the
  prompt. (An FTS match rescues a capability the model *omitted*, never one it
  explicitly rejected.) A candidate that does *none* of the required capabilities
  is dropped.
- **`serves` counts toward the mandate, but is rarely met.** A *"serves / sells
  to X"* phrase is a real mandatory requirement, so it's in `mandatory_total` —
  but this dataset has no customer field and the one-sentence descriptions never
  name customers, so it is almost always unverifiable. The result: a query like
  Q3 lands every row at `mand 3/4`, `full_match` is `false`, the gap is written
  into `inferences[]` ("serves: European banks — not stated…"), and
  `metadata.caveats` explains why. A customer mention the model *does* find only
  counts if its quote grounds in the text.

### Scoring — two fractions, a tiered sort, one readout

Each result reports:

| field | meaning |
|---|---|
| `mandatory_met / mandatory_total` | **every** mandatory requirement — location, industry, each numeric bound, each exclusion, the capability group, and each `serves` phrase. Structural ones hold by construction; `serves` usually does not (see above). The total shows the mandate's size. |
| `preferences_met / preferences_total` | soft preferences met; each miss named in `unmet_preferences` |
| `score` | `0.65·(mand fraction) + 0.35·(pref fraction)` — a **readout**, not the sort key |
| `keyword_score` | min-max normalised bm25 (JSON only; near-noise here, so not displayed) |
| `llm_validated` | text-checked by the model, or a deterministic-only tail row (rank > `validation_batch`)? |

The **sort is tiered** — `(all mandatory met? → # mandatory met → # preferences
met)`, then bm25 silently breaks an exact tie. No weights decide order.

`metadata.match_summary` covers the *whole* matched set, not the returned slice:
how many pass the filters, how many of those meet **every** preference (a SQL
count), how many the LLM confirmed. `results_are_top_ranked` is `false` when every
returned row sits in one tier and that tier is larger than the slice — then
`ranking_note` says e.g. *"25 companies meet every requirement and preference; the
10 returned are an arbitrary slice"*.

`evidence[]` (verified quotes) and `inferences[]` (model claims + basis) are
always split.

---

## HTTP API

`POST /agent/search` runs the full workflow synchronously and returns the same
`AgentResponse` the CLI produces. `GET /health` reports index status; `GET /docs`
is the generated OpenAPI UI (the authoritative schema). An `X-API-Key` header is
checked against `AGENT_API_KEY` when that env var is set — wallet protection for a
public URL, not a security boundary.

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
                   "exclusions": {…}, "preferences": {…}, "serves": ["…"],
                   "revision_policy": {…}, "notes": ["…"] },
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
    "candidates_retrieved_per_iteration": [100], "candidates_validated": 10,
    "funnel": {…}, "match_summary": {…}, "validation_outcome": {…},
    "revised_search_performed": false, "ranking_note": "", "caveats": ["…"],
    "llm_calls": 2, "prompt_tokens": 4700, "completion_tokens": 1400,
    "est_cost_usd": 0.0015, "latency_ms": 9000, "model": "gpt-4o-mini",
    "provider": "openai", "timed_out": false, "stop_reason": ""
  }
}
```

Two nested timeouts: `BudgetGuard`'s per-run deadline (`AGENT_DEADLINE_S`, ~90 s)
returns a partial body with `timed_out: true`; the client should also set its own
request timeout above that. Execution is synchronous — an async `202` + status
endpoint is described under "What was intentionally left out".

---

## Search-revision & failure recovery

- `check_feasibility` → 0 mandatory matches → empty result + which criteria
  failed, **no wasted LLM calls**. This is the primary "insufficient results"
  path, and Q5 exercises it.
- Otherwise, if fewer than `min_results` candidates clear the capability gate
  after iteration 1 and no revision yet → `relax_preferences` (a deterministic
  ladder, preferences only) → one more retrieve + validate. Bounded: ≤ 1
  revision, ≤ 2 iterations.
- `BudgetGuard` ([app/budget.py](app/budget.py)) is checked at every node entry on
  both engines: LLM calls ≤ 5, iterations ≤ 2, revisions ≤ 1, pool ≤ 100,
  validation batch ≤ 10, results ≤ 50 (default 10 — see D14a), wall-clock
  deadline. A breach raises `BudgetExceeded`; the runner records `stop_reason` and
  still returns a composed partial response — never an exception, never an
  unbounded loop.

The `relax_preferences` path (`≤ 1`, deterministic ladder) is implemented and
unit-tested (`test_workflow.test_revision_path_runs_once`, which scripts an empty
first validation pass). It is **not exercised by the evaluation queries**: this
dataset is dense enough that every *feasible* mandate returns a full result set,
so the honest "too few results" trigger is `check_feasibility` returning empty
(Q5), not a mid-run revision. On sparse real data, a restrictive preference could
leave the validated set below `min_results`, and the ladder would relax the
softest preference and re-search once.

---

## LLM & model choices

- **`openai:gpt-4o-mini`, single provider.** ~$1–3 to build and evaluate the
  whole system; cheap enough that rate-limit management and "$0-equivalent" cost
  reporting aren't worth it. Pricing verified against live docs (Aug 2026):
  $0.15 / $0.60 per 1M tokens.
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
The 5 core queries + supplements S1/S3 in `eval/queries.yaml` are the scored set,
each with a **hand-authored `expected` block** — the "correct parse" oracle. The
5 cover the main capability dimensions: qualitative + structured criteria (Q1),
mandatory vs preferred (Q2), exclusions + evidence (Q3), controlled revision (Q4),
and empty / conflicting handling (Q5).

**Parse accuracy.** Running the 20 dev queries through `check` (interpret + plan),
the parse was fully correct on **14/20** before any tuning; all five core queries
were correct. The failures clustered on *underspecified colloquial
phrasing* — the model fabricating a numeric threshold for a vague word
("innovative", "smaller/newer", "tiny"), or dropping a `<topic>` when it also
implied an industry. Three prompt rules address these (a `to/serves <customer>`
phrase is never a location filter; vague quality words go to `ambiguities`, never
a fabricated bound; `<topic> companies` emits the topic *and* the industry) plus
hyphen normalisation in the lexicon. An LLM self-verification pass over the parse
is the natural next transparency step — see "What I'd do next".

**Honest note on held-out purity:** the intent was to run `queries.yaml` once as a
direction check and not again until the final pass. In practice it was exercised
a few times while debugging the end-to-end wiring, so the strict "look once"
methodology is partly spent — but every prompt/code change was driven by *general*
correctness (the mandatory / preference split, capability OR-logic, geography
canonicalisation, British spelling, the vague-word and customer-vs-location rules
above), not by fitting the expected strings. `dev_queries.yaml` carried the real
iteration load.

**What counts as a successful result.** For a given query: every returned company
satisfies all *mandatory structured* criteria; each has ≥ 1 grounded evidence span
for the requirements the text can support (or the gap is labelled an inference,
not hidden); no company is fabricated (every id exists in the dataset); the run
stayed within every budget bound; when the data genuinely has no match the result
is an empty list + a stated reason, not a padded or forced list; and — a human
call — the ranking within the "match" tier is defensible.

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

**Latest pass: 7/7 green** on the deterministic checks. The full ranked output
for every query — interpreted mandate, ambiguities, results with evidence, the
match summary, and a one-line judgment note — is in [eval/eval6.txt](eval/eval6.txt).

Needs human judgment (not auto-scored): ranking quality, faithfulness to nuance,
category-exclusion correctness, and whether the evidence is *sufficient* (not just
present).

---

## Deployment

**Live:** `https://find-companies.onrender.com` (Render free tier — first request
after ~15 min idle cold-starts in ~30–60 s). `/docs` for the OpenAPI UI.

```bash
curl -s https://find-companies.onrender.com/health
curl -s -X POST https://find-companies.onrender.com/agent/search \
  -H 'X-API-Key: <key provided separately>' -H 'content-type: application/json' \
  -d '{"query":"German drug-discovery companies preferably founded after 2018"}'
```

**Docker, self-contained.** `python -m app.ingest` runs at image-build time, so
the container carries the 50k index and starts instantly. Runs as a non-root user
on port 7860.

```bash
docker build -t compara-search .
docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... -e AGENT_API_KEY=secret compara-search
curl -s localhost:7860/health
```

**Render.com (free tier, no card).** Create the service by hand — the Blueprint
flow ([render.yaml](render.yaml) is kept as a config reference) asks for a card
because a blueprint *could* provision paid resources; the manual path does not.

1. Render dashboard → **New → Web Service** → connect this GitHub repo.
2. Runtime **Docker** (auto-detected from the Dockerfile — no build/start command
   to set), instance type **Free**.
3. Under *Environment*, add `OPENAI_API_KEY` and `AGENT_API_KEY`.
4. Create. Render builds the image (the 50k index is baked in at build time — no
   persistent disk needed) and serves at `https://<name>.onrender.com`;
   `/health` is the health check.

The free instance has no persistent storage (fine — the only runtime write is the
optional LLM response cache) and spins down after ~15 min idle → the next request
cold-starts in ~30–60 s. Idle RAM is ~40 MB against the 512 MB limit.

**Alternatives.** *Google Cloud Run* (`gcloud run deploy --source .`,
scale-to-zero, generous free tier) — card required. *Hugging Face Spaces* — a
push to a Docker Space works, but HF now gates the Docker SDK behind PRO.
*Vercel* — analysed and rejected (below).

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
| `app/evaluate.py` | End-to-end evaluation harness → `eval/RESULTS.md`. |
| `app/config.py` · `app/*.yaml` | Taxonomy config: regions, lexicon, schema map, revenue buckets. |
| `eval/queries.yaml` · `eval/dev_queries.yaml` | Scored set (with `expected`) · dev iteration set. |
| `tests/` | 207 tests, offline, against the fake provider + a fixture index. |

CLI: `ingest` · `db "<sql>"` · `search` (tools, no LLM) · `check` (interpret +
plan + feasibility) · `run` · `eval`.

---

## Known limitations

- **Ranking is only as good as the signal in the query.** With no soft preference
  and templated descriptions (identical bm25), every match ties on the whole sort
  key — `match_summary.results_are_top_ranked` goes `false` and `ranking_note`
  says "N companies match equally; add a preference". On real, varied text bm25
  would separate them.
- **Recall is bm25-top-100.** Topic-matching companies past bm25 rank 100 never
  reach validation. On this templated data the top 100 almost certainly contains
  every real match; on real data this needs a bigger pool + embeddings.
- **"serves X" is usually unverifiable** from a one-sentence description, so Q3-
  style queries land every row at `mand 3/4` (never a full match), with the gap
  labelled an inference and a caveat — correct, but the ceiling is the source, not
  the system.
- **Revision never fires on the eval queries** — the dataset is dense enough that
  every feasible mandate returns a full set, so the honest "insufficient results"
  path is `check_feasibility` → empty (Q5). The `relax_preferences` ladder is
  implemented and unit-tested; on sparse real data it would relax the softest
  preference and re-search once.
- **The parse can still misread very vague phrasing** — see "Parse accuracy"
  above (14/20 dev queries clean pre-tuning, all 5 required correct).
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

- **LLM self-verification of the parse.** The interpret step is the highest-
  leverage failure point. A cheap next step: after `build_search_plan`, one call
  that gets the original query + the assembled plan and answers *"does this
  capture every hard requirement, and did it invent any constraint not in the
  query?"* — writing findings into `ambiguities`/`caveats` only (never mutating
  the plan), or triggering a single bounded re-parse the way schema-repair does.
  Deferred because the single interpret call was reliable on the target query
  shape (14/20 dev, 5/5 required); a clause-by-clause parse trace in the response
  is the lighter-weight version of the same idea.
- **350M+ companies:** retrieval → **Elasticsearch** (structured filters + BM25)
  **+ a vector DB** (Qdrant/Weaviate) with ANN, hybrid-ranked; tiered filtering
  (cheap structured gate → ANN → LLM validation); sharding by region/sector;
  embeddings precomputed in a batch pipeline.
- **Concurrent agent runs:** the workflow is already a pure `run_workflow(query,
  cfg)` with no shared mutable state, so it scales as stateless workers behind a
  queue. Add a Postgres run store (behind a `RunStore` protocol — in-memory dict
  today) + Redis for locks / idempotency keys; per-run budget isolation is already
  in `BudgetState`; pool provider rate limits across workers with a token-bucket.
- **OSS / self-hosted LLMs:** the `Provider` protocol behind `LLMClient` is the
  only seam — add a `VLLMProvider` serving Llama/Mistral. The open question is
  structured-output reliability without OpenAI strict mode: grammar-constrained
  decoding (Outlines / llama.cpp GBNF) or a stricter repair loop. Cost/latency vs
  a hosted model is a per-deployment call; the fake provider keeps tests free
  regardless.
- **Persistent workflow state:** swap the in-process driver for LangGraph with a
  Postgres checkpointer — every node boundary becomes a resumable savepoint, so a
  crashed or timed-out run continues instead of restarting, and human-in-the-loop
  pause points (approve the search plan, review the shortlist) drop in as
  interrupt nodes. `RunState` is already one Pydantic model, so it serialises
  as-is.
- **Production monitoring:** emit the structured log records (already keyed by
  `run_id`) and the `RunTrace` counters (stage latencies, tool ok/fail, llm_calls,
  tokens, cost) to OpenTelemetry → a metrics/traces backend. Dashboard the funnel
  drop-off, p50/p95 latency, cost per run, abstention rate, `stop_reason`
  breakdown; alert on drift in any of them (e.g. abstention rate spiking =
  interpret regression).
- **Model & prompt versioning:** move the prompts into a small registry with
  content-hash version IDs; stamp `(interpret_prompt_id, validate_prompt_id,
  model)` into `RunTrace` so every result is reproducible. Any prompt or model
  change runs the held-out `eval/queries.yaml` as a CI gate and diffs the
  deterministic columns + the parse against the previous version before merge.
- **Production eval:** labelled golden sets, retrieval recall@k over a judged
  candidate pool, an LLM-judge calibrated against human labels for the
  non-deterministic columns (ranking quality, evidence sufficiency), online
  feedback capture (thumbs / "not relevant" per result) feeding both the golden
  set and drift alerts.
- **Data lineage & governance:** an immutable run store; every evidence item
  points to `(company_id, source_field, record_version)` so a claim can be traced
  to the exact record it was drawn from; the interpret + validate prompts and
  model versions are retained per run for audit; PII / access policy enforced at
  the retrieval-tool boundary.
- **MCP-compatible tools:** the three tool contracts are already Pydantic in/out
  with no hidden state — wrap them as an MCP server so the same
  `search_companies` / `count_matching` / `get_by_ids` serve other agents and
  clients unchanged.

---

## Decision Log

<details>
<summary>The reasoning on record, in build order (D1–D18).</summary>

**D1 — Deterministic workflow, not autonomous agents.** The retrieval strategy is
derivable from the parsed mandate, so an LLM planner adds failure modes
(uncontrolled loops, unpredictable cost) without adding capability.

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

**D14 — Ranking: two fractions, a tiered sort, one readout.** The model reports
per-capability / per-"serves" support (with a quote); everything else is code.
Each result carries `mandatory_met/total` (all mandatory requirements —
structural + capability + each `serves` phrase) and `preferences_met/total` (every
miss named in `unmet_preferences`), plus a `score = 0.65·mand + 0.35·pref`
**readout**.
The **sort is tiered** — `(all mandatory met? → #mandatory → #preferences)` —
then bm25 silently breaks an exact tie. No weights decide order. The model's own
numeric score was tried and dropped (it clustered); bm25 got its own score-weight
and column, also dropped (near-noise on templated text). `match_summary` counts
the whole matched set (filters / meet-every-preference SQL count / LLM-confirmed)
and flags `results_are_top_ranked = false` with a specific note when the returned
rows are one tier of a larger set.

**D14a — Results can exceed the validation batch.** `validation_batch` (10)
bounds LLM cost; `result_limit` (default 10, up to 50 via `--limit` / the API
`limit`) bounds what's returned. The whole retrieval pool is scored + sorted
deterministically; rows past rank 10 come back `llm_validated: false`
(keyword-matched, text not confirmed).

**D14b — `serves` counts toward the mandate, and a customer adjective becomes a
location preference.** A *"serves / sells to X"* phrase is a real mandatory
requirement, so it's in `mandatory_total` — but the dataset has no customer field
and descriptions omit customers, so it is almost always unmet: Q3 lands every row
at `mand 3/4`, `full_match: false`, the gap in `inferences[]`, and a
`metadata.caveats` entry explains why. Separately, a region word *inside* a
`serves` phrase ("**European** banks") is added as a **location preference** (never
a filter) by `build_search_plan` — the mandate never said where the *company* is,
so European firms rank first without excluding anyone. (Earlier this phrase was
mis-parsed into a hard `location IN (…)` filter; the interpret prompt now forbids
that.)

**D15 — Skip the validation call when there is nothing to validate.** A purely
structural mandate ("Nordic medtech founded before 1996", "German energy, exclude
smart grid") is fully answered by the SQL `WHERE` gate; the model would only
rubber-stamp every row with empty evidence. So `validate_and_rank` detects the
no-capability / no-"serves" / no-semantic-exclusion case, builds the ranking
deterministically, and spends **zero** LLM calls. Keyword exclusions still apply
(they run in SQL); a *semantic* exclusion category keeps the model in the loop.

**D16 — Capability phrases contribute their industry.** "fraud-detection
technology" — gpt-4o-mini variously calls that industry `Technology`, `Fintech`,
or nothing. `build_search_plan` merges the lexicon-implied industry
(`fraud detection` → `Fintech`) into the `industry IN (…)` set rather than
treating it as a fallback. Broadening the set is recall-safe; the validator
narrows on the text. Hyphenated compounds ("fraud-detection") are normalised to
spaces before the lexicon lookup.

**D17 — Vague quality words never become filters.** "innovative", "AI", "smaller
/ newer", "tiny", "startup" carry no structured meaning here (every description
is "AI-powered …"). The interpret prompt forbids turning one into a capability,
an industry, or a fabricated numeric bound — it goes to `ambiguities`. Before this
rule, "lean towards smaller, newer firms" produced an invented
`founded_year_gte / employee_count_lte` pair.

**D18 — The interpret prompt carries the dataset's topic vocabulary.** The 10
industries were already a closed list in the prompt; the 27 core topic phrases
(`data/dataset_vocab.json`, injected as `_TOPICS`) now are too, as a *soft* target
("map to the closest 1–3 when one clearly fits, else keep the phrase + note an
ambiguity"). This shifts synonym normalisation ("cancer research" → drug
discovery + molecular analysis) to parse time, so fewer feasible queries hit an
empty topic match. Soft, not `MUST`: a genuinely absent capability
("quantum cryptography") is better left unmapped (→ honest thin result) than
forced onto a wrong topic. The lexicon stays as the deterministic backstop and
also seeds the industry from a topic. A few-shot example plus `cancer` / `oncology`
lexicon entries pin the one case gpt-4o-mini resisted.

</details>

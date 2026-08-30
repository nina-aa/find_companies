# dumbplan.md — PLAN.md explained in plain language

This is a companion to `PLAN.md`. No code, no jargon (or jargon explained the first
time it shows up). The goal: you understand **what** is being built, **why** it's
built that way, and **why not** some other way — well enough to defend it in an
interview.

---

## 1. What you are actually building

Someone types a sentence like:

> "Find fintech companies in Finland working on fraud detection. Prefer companies
> founded after 2015 with fewer than 250 employees."

Your system has to:

1. Understand that sentence — what's a hard requirement ("must be in Finland") vs a
   nice-to-have ("prefer founded after 2015").
2. Search a list of 50,000 fake companies for ones that fit.
3. Check each candidate actually matches — not just keyword-collides.
4. Return a ranked list, and for each company, **show the evidence** ("its description
   says 'fraud detection'") separately from **guesses** ("probably serves banks").
5. Never invent a company or a quote. If nothing matches, say "nothing matches" and
   explain why.
6. Do all this within strict limits — a capped number of AI calls, a time limit, a
   money limit — and never get stuck in a loop.

The 50,000 companies come in a file `companies.json`. Each row looks like this:

```
{ "id": 1, "name": "...", "description": "AI-powered platform for fraud detection.",
  "industry": "Fintech", "location": "Finland", "founded_year": 2018,
  "employee_count": 240, "revenue_range": "10M-50M" }
```

That's it. The descriptions are all one templated sentence — there are only ~195
distinct ones across all of Fintech. **This detail drives almost every design
decision** (see "semantic search" below).

This is a **take-home test for a job**. It is scored on judgement, not on how much you
build. A small system you can explain every line of beats a big one you can't.

---

## 2. The one big idea: AI where you need it, plain code everywhere else

An AI language model (the "LLM" — Large Language Model, e.g. GPT) is good at reading a
messy human sentence and turning it into structure. It is bad at being predictable,
cheap, fast, and traceable.

So the system uses the LLM in exactly **two places**:

- **Place 1:** read the user's sentence → produce structured criteria.
- **Place 2:** look at a candidate company and judge "does this genuinely match, and
  what's the evidence?"

**Everything else is ordinary Python code** that does the same thing every time:
filtering the database, doing the ranking maths, deciding whether to retry, enforcing
the limits, checking quotes are real, assembling the final answer.

The plan calls this ordinary predictable code **"deterministic"** — meaning: same
input, same output, every time, no surprises. When you see "deterministic" in
PLAN.md, read it as "plain rule-following code, not the AI."

Why this split matters: the test explicitly rewards a "clear separation between LLM
reasoning and deterministic controls" and warns against systems that "enter an
uncontrolled reasoning or tool-use loop." Keeping the AI on a short leash is the
point.

---

## 3. Glossary — the words you flagged

### SQLite / FTS5 / bm25

**SQLite** is a database that lives in a single file on disk. No server to run, no
setup — it's just `companies.db`, a file you can open in any database viewer. You load
the 50,000 companies into it once.

Why use a database at all when 50,000 rows would fit in memory? Because:

- Filtering becomes a standard database query ("give me rows where location = Finland
  AND industry = Fintech"). You're using a battle-tested tool instead of hand-writing
  filter logic.
- Counting matches ("how many companies fit?") is a one-line query — needed for the
  "is this even possible?" check.
- It's the honest engineering choice and it stays simple.

**FTS5** ("Full-Text Search version 5") is a feature built into SQLite for
**keyword search** over text. Instead of "find rows where description exactly equals
X," it's "find rows whose description *contains the words* 'fraud detection',"
including ranking them by how well they match.

**bm25** is the specific maths FTS5 uses to score "how well does this text match these
keywords" — it's the standard, decades-old formula search engines use. A row where
"fraud detection" appears prominently scores higher than one where it's buried. You
don't implement bm25; you just get it for free from FTS5.

Why this is enough here: the company descriptions are templated one-liners that boil
down to a topic keyword ("...for fraud detection."). Keyword search over those
captures basically everything there is to capture.

### embeddings / semantic search / ANN / FAISS  (the thing you're NOT building)

**Semantic search** means searching by *meaning* rather than by exact words — so a
search for "car company" would also find a description that says "automotive
manufacturer" even though no word overlaps.

The way you do that: an **embedding** is a list of numbers (a few hundred of them)
that represents the *meaning* of a piece of text. Texts with similar meaning get
similar number-lists. You turn every company description into an embedding, turn the
user's query into an embedding, and find the companies whose numbers are "closest" to
the query's numbers.

**ANN** ("Approximate Nearest Neighbour") is a fast way to find the closest
number-lists when you have millions of them — checking every single one is too slow at
scale, so ANN trades a tiny bit of accuracy for a lot of speed. **FAISS** is a popular
library (from Meta) that does ANN.

**Why the plan skips all of this:** the descriptions here are templated and nearly
identical. "Meaning-based" search and "keyword" search would return almost the same
results, because the meaning *is* basically just the topic keyword. Building the
embedding pipeline would add a dependency and complexity for near-zero benefit **on
this dataset**. The README will explain that for *real* company data (rich, varied
descriptions) you absolutely would use embeddings + a vector database — and describe
exactly how. Knowing when *not* to build something is part of the score.

(There's an optional middle-ground mentioned — `fastembed` — a lightweight embedding
tool that could re-rank results if you have spare time. It's explicitly a "only if
everything else is done" item.)

### fixtures

A **fixture** is a fixed, pre-prepared piece of data you use for testing. Instead of
hitting the real 50,000-row file or the real AI every time you run a test, you use a
small hand-made slice ("here are 20 fake companies I know the answer for") or a
hand-written "correct answer" to compare against.

Two kinds in this plan:

- **Test fixtures** — a small slice of the real data, so tests for the filtering code
  run instantly and offline.
- **Eval fixtures** (`eval/queries.yaml`) — for each of the 5 test queries, *you*
  hand-write what the correct interpretation should be ("this query means: countries =
  [Finland], industry = [Fintech], and 'founded after 2015' is a preference not a
  requirement"). This hand-written answer is the **oracle** — the thing you check the
  system's output against. You write these *first*, before building, so they guide
  development.

### Pydantic model / schema / "contract" / "schema-validated"

A **schema** is a description of the *shape* data must have: which fields exist, what
type each one is, which are required. Like a form with labelled boxes and rules
("Founded year: must be a 4-digit number; Countries: a list of text values").

**Pydantic** is a Python library that lets you write that form as a class, and then
*enforce* it. You hand Pydantic some data; it either gives you back a clean, validated
object, or it throws a clear error saying exactly what's wrong ("founded_year got the
text 'recently', expected a number").

A **Pydantic model** is one such form. When the plan says *"every LLM output has a
Pydantic model,"* it means: every time the AI is asked for something, we've defined in
advance the exact shape of answer we'll accept.

**"Schema-validated"** means: after the AI responds, we run its answer through that
form. If it fits — great, continue. If it doesn't fit — we don't just crash or pass
along garbage. We do **one "repair retry"**: send the AI its own answer plus the error
message ("you gave me text where I needed a number, fix it"). If it still fails, the
step fails cleanly with a structured error message. **No endless retrying.**

**"Contract"** is just the informal word for "the agreed shape." Each search tool has
an input contract and an output contract — "you give me this shape, I promise to give
you back that shape." Writing these down (as Pydantic models) is what makes the tools
predictable and testable.

Why this matters so much: an AI will occasionally return something malformed or
hallucinated. The schema is the wall that catches it before it corrupts the rest of
the run. This is a big chunk of the "reliability" score.

One rule the plan stresses (learned the hard way on a past project): keep each of
these forms **small and purpose-built** — only the fields the AI genuinely must
produce. Don't reuse a big internal data structure as the AI's form. IDs, timestamps,
and computed values get added by code *after* the AI call, not asked of the AI.

### span-grounding / evidence vs inference / "groundedness"

**Groundedness** = every claim the system makes is backed by something real in the
source data, not made up.

The system separates two things in its output:

- **Evidence** — a direct quote from a company's actual record. "The description field
  says: '...platform for fraud detection.'"
- **Inference** — a reasoned guess the AI made. "This company probably serves banks"
  — with the basis stated ("its topic is fraud detection and it's a Fintech").

**Span-grounding check** ("span" = a snippet of text): after the AI claims a quote is
evidence, plain code checks that the quoted text is **literally, character-for-
character present** in the field the AI said it came from. It looks it up in the
database and does a substring check.

If the quote is really there → it stays as evidence. If it's *not* there (the AI
paraphrased or invented it) → it gets **demoted to an inference**, never shown as a
hard quote.

Why: "a cited quote is an AI claim until verified against the source." This one check
is the core anti-hallucination control, and groundedness is 25% of the score. Because
the descriptions here are so thin, expect many answers where the evidence is one short
phrase and everything else is honestly labelled as inference — **that's fine and
expected**, not a weakness.

### workflow vs agents / "orchestration"

**Orchestration** = how the individual steps are wired together and how the system
decides what to do next.

There are two philosophies:

- **Agent:** you give the AI a set of tools and a goal, and *the AI decides* what to
  do next at each turn — which tool to call, whether to try again, when it's done. The
  AI is the controller. Powerful, but can wander, loop, and rack up cost.
- **Workflow:** *you* decide the steps and their order in advance, in plain code. The
  AI is called at specific points to do specific narrow jobs, but it never decides
  "what happens next." Code decides that.

This plan is a **workflow**. There are 7 steps ("nodes"). 2 of them make one AI call
each. The other 5 are plain code. Which step runs next is always decided by code.

Why a workflow and not agents:

- The task is well-defined. You know the steps up front: interpret → plan → check
  feasibility → search → validate → (maybe revise) → respond. There's no genuine "the
  AI needs to figure out the strategy" situation.
- There are only 3 search tools and the right way to use them falls out of the parsed
  query. An AI "planner" choosing between them would add ways to fail without adding
  ability.
- The test rewards bounded, controlled systems and says "the number of agents is not
  an evaluation criterion" and "complexity will not be considered a strength on its
  own." That's a strong hint.
- It's **still an agent loop in spirit** — plan, act, observe, adapt — you've just
  given the loop a code controller instead of an AI controller. Anthropic's own
  "Building Effective Agents" guidance says workflows win for well-defined tasks. Say
  this in the interview.

One deliberate choice to flag: even the "should we try again with looser criteria?"
decision is made by **code following a fixed rule**, not by the AI. One of the test
queries literally spells out the rule ("broaden the founding-year preference, but do
not broaden the country requirement"), so a fixed rule is both sufficient and more
traceable.

### LangGraph / "state machine" / nodes / edges

A **state machine** is a formal way of describing "a system that is always in exactly
one of a fixed set of states, and moves between them along defined paths." A traffic
light: Red → Green → Yellow → Red. It can't be "half green." The paths between states
are the only moves allowed.

Your workflow is a state machine: it's in the "interpreting" state, then the
"planning" state, then "searching," etc. A **node** is one state/step. An **edge** is
an allowed move from one step to the next. Most edges here are straight lines
(after planning, always go to feasibility-check). Two edges are **conditional** —
"if zero companies can possibly match, jump straight to the response; otherwise
continue" and "if too few good results and we haven't revised yet, go revise;
otherwise finish."

**LangGraph** is a library for building exactly this kind of step-graph for
AI systems. You define the nodes, the edges, and a shared "state" object that gets
passed along and updated at each step.

Why use it: the job description for this role explicitly lists LangGraph as a required
skill, so demonstrating it is on-target. It also gives you a clean structure and hooks
for future features (like pausing/resuming runs).

Why it's a *risk*: it's new to you and has its own concepts and quirks. So the plan
**de-risks** it: first build the whole system with a ~40-line hand-written loop
("do step 1, do step 2, check the condition, ...") that you understand completely and
that definitely works. Get everything green. **Then** swap in LangGraph as a separate,
time-boxed step, keeping the hand-written version as a safety net. If the LangGraph
port gets messy and runs ~1 hour over budget, ship the hand-written version and
explain the trade-off in the README — that reads as good judgement, not a gap.

### BudgetGuard / "bounded" / "uncontrolled loop"

**"Bounded"** = has hard limits it cannot exceed. The opposite — an **"uncontrolled
loop"** — is an AI system that keeps calling itself or its tools over and over,
burning money and time, with no built-in stop. The test explicitly says the system
"must not be able to enter an uncontrolled reasoning or tool-use loop." Avoiding this
is 20% of the score.

**BudgetGuard** is a small piece of code (that you write — not a library feature) that
is checked at the start of every single step and enforces every limit:

- at most 5 AI calls per run (a typical run uses 2, so there's visible headroom)
- at most 2 search iterations
- at most 1 "try again with looser criteria" revision
- at most 100 candidate companies pulled per search
- at most 10 companies sent to the AI for validation
- at most 10 results returned
- a wall-clock deadline (e.g. 90 seconds)

If any limit is about to be breached, the run **stops gracefully** and returns
whatever it has so far, plus a flag like `timed_out: true` or `budget_exhausted`. It
never throws an error at the caller, and never keeps going.

The key point for the interview: **BudgetGuard is the real control, and it's the same
whether you use LangGraph or the hand-written loop.** The framework is not what keeps
the system safe — your code is.

### feasibility gate / abstention

**Abstention** = deliberately answering "there is no answer" instead of forcing a
wrong one. A good search system, asked for something impossible, should say "nothing
matches" — not return the closest-but-wrong companies to look helpful.

Example: one test query asks for "Finnish fintech companies with more than 5,000
employees." The biggest company in the entire dataset has exactly 5,000 employees. So
**zero** companies can match. The right answer is an empty list plus "no companies
satisfy the mandatory criteria: employee_count > 5000."

The **feasibility gate** (`check_feasibility` step) is where this is caught. Before
spending any AI call on validation, plain code counts how many companies match the
*hard* requirements. If the count is zero, the system skips straight to composing an
empty response with an explanation. This makes the impossible query **cheap** (1 AI
call instead of 2) **and correct**.

The plan also notes the *scoring* of eval must credit a correct "nothing matches" as a
PASS — a naive scorer that marks every empty result as failure would misjudge a
system that's designed to abstain honestly.

### provider / fake provider / "seam"

A **provider** here means "whoever actually runs the AI model" — in this case OpenAI's
API serving `gpt-4o-mini` (the small, cheap GPT model chosen for cost reasons).

A **fake provider** is a stand-in that implements the same interface but doesn't call
any real AI — it just returns a valid dummy answer of the right shape. All the tests
use the fake provider, so:

- tests run offline, instantly, and cost **nothing** (zero tokens = zero money)
- the predictable logic (filtering, ranking, limits, grounding checks) is tested
  without ever depending on a live AI

Only a handful of explicit "smoke tests" and the final evaluation run use the real
paid API.

A **"seam"** is a deliberate clean dividing line in the code where you could swap one
implementation for another without rewriting everything around it. The plan builds a
thin `LLMClient` with a `Provider` behind it — that's the seam. Today there's one real
provider and one fake. Tomorrow, adding a second AI vendor, or a self-hosted model, is
a drop-in at that seam, not a rewrite. The README lists this as one of three seams
that make the "demo → production" path cheap.

One trap the plan calls out: the response cache (which saves AI answers so re-runs are
fast) must record *which provider* gave each answer. Otherwise a fake answer could get
served for a real request — no error, just wrong results.

### hydrated / "hydrate from DB"

**"Hydrating"** an object means filling in its full details from a trusted source,
starting from just an identifier.

In this system: the AI validation step works with company IDs and makes judgements.
But when building the **final response**, the system does **not** copy company names,
industries, founding years etc. from anything the AI said. Instead it takes the list
of IDs and looks each one up fresh in the database — "hydrating" each result from the
real record.

Why: this is what *guarantees* "every returned company actually exists in the dataset
with exactly these attributes." The AI is never trusted to report facts it could get
subtly wrong or invent. The AI decides *which* companies and *why*; the database
supplies *what they are*.

### held-out eval / oracle / recall@k / LLM-judge

**Eval** (evaluation) = systematically measuring how good the system's answers are,
rather than eyeballing one or two.

**Oracle** = the source of truth you compare against — the hand-written "correct
answer" for each test query.

**Held-out** = kept separate and untouched during development, so your final score
reflects "does this generalise to queries I didn't tune for" rather than "did I
tweak it until the 5 known queries passed." The plan:

- writes **10 of its own practice queries** (`dev_queries.yaml`) and iterates against
  those while building
- keeps the **5 required queries** (+ 2 supplements) sealed — runs them once as a
  direction check, then not again until the very end

This is itself a talking point: the job wants "evaluation frameworks for agent
reasoning quality."

**recall@k** = a standard search-quality metric: "of all the genuinely relevant
companies that exist, what fraction did we find in our top k results?" (e.g.
recall@10). The plan mentions it as a *production* measure — it needs labelled data
("here are ALL the right answers") which you don't have for 50,000 synthetic rows, so
it's described in the README, not built.

**LLM-judge** = using a second AI call to grade the system's output ("is this a good
answer? score 1–5"). Useful at scale but needs to be calibrated against human
judgement to be trustworthy. Again: described as a production technique, not built
here. For this assessment, quality checks are either deterministic (did the filters
apply correctly? do the quotes really exist?) or a human eyeballing the ranking.

### Docker / Hugging Face Spaces / serverless

**Docker** packages your app plus everything it needs to run (Python version,
libraries, the data file, the pre-built database) into one **image** — a sealed box
that runs the same on your laptop and on a server. "It works on my machine" stops
being a problem.

**Hugging Face Spaces** is a free hosting service that can run a Docker image and give
you a public URL. The plan picks it because: no credit card required, it keeps a
**persistent container** running (see next point), and secrets like the OpenAI key can
be set safely. Downside: free Spaces go to sleep after inactivity and take a moment to
wake up ("cold start").

**Serverless** (e.g. Vercel, AWS Lambda) is the alternative model: there's no
always-on process; each request spins up a short-lived function that runs and then
disappears. The plan **rejects serverless** for this project because:

- an agent run here can take 30–90 seconds; serverless platforms often kill functions
  before that
- there's no persistent process to hold run state in memory
- every request pays a cold-start cost

A bounded-but-slowish agent needs a persistent container, so Spaces (or Google Cloud
Run) fits and serverless doesn't. Writing this comparison down shows the analysis was
done.

The plan also says **Docker is last** — you verify everything through the CLI
(command-line tool) first, locally, and only package it near the end. A working local
setup is an acceptable final deliverable if deployment runs out of time.

### FastAPI / endpoint / sync vs async

**FastAPI** is a Python library for building a web API — code that listens for HTTP
requests and returns responses (usually JSON).

An **endpoint** is one URL the API answers at. Here the main one is
`POST /agent/search` — you send it `{"query": "find fintech companies..."}` and it
returns the structured result.

**Sync (synchronous):** you send the request and the connection stays open, waiting,
until the whole search is done (up to ~120 seconds), then the full answer comes back
in that same response. Simple. Fine when a run takes under 2 minutes and the server is
always on.

**Async (asynchronous):** you send the request and *immediately* get back "accepted,
here's a run ID" (HTTP status `202`). The work happens in the background. You then poll
a second endpoint ("is run 123 done yet?") until the answer is ready. More
production-shaped — it doesn't tie up a connection for minutes — but it's more code
(background task runner + a status endpoint + somewhere to store in-progress runs).

The plan's choice: **build sync** (simplest, and safe because the container is
persistent), but structure the run-state storage behind a clean interface so adding
the async version later is a small change, not a rewrite. Async is described in the
README as the production path.

The same core function (`run_workflow`) is called by both the CLI and the API — both
are ~20-line thin wrappers. You build the engine once.

---

## 4. What actually happens on one search (plain walkthrough)

The 7 steps, in order. "AI" = one call to GPT; everything else is plain code.

1. **interpret_mandate** *(AI call #1)* — Read the user's sentence. Produce structured
   criteria: which countries, which industries, year/employee/revenue ranges, topics,
   and crucially **which of these are hard requirements vs preferences vs
   exclusions**. Resolve fuzzy terms ("Nordic" → Finland/Norway/Sweden; "after 2015" →
   2016 onwards). List anything ambiguous. The answer is checked against its schema.

2. **build_search_plan** *(plain code)* — Turn those criteria into an actual search
   plan: database filter conditions, keyword terms, the order to call tools in.
   Validate the industry names against the known list of 10. Expand regions to country
   lists using a fixed map. Parse "below €10M" into actual numbers. No AI — this is
   mechanical translation and stays fully traceable.

3. **check_feasibility** *(plain code)* — Count how many companies match the **hard**
   requirements. If zero → skip to step 7 with an empty result and an explanation.
   Spend no more AI calls. (This is the abstention gate.)

4. **retrieve** *(plain code calling search tools)* — Run the search: apply the hard
   filters as a database query (this is the gate — preferences never filter here),
   then rank the survivors by keyword match. Pull at most 100 candidates, then narrow
   to the top ~10.

5. **validate_and_rank** *(AI call #2)* — Send the AI the ~10 candidates **in one
   batched call**. For each: it already knows which hard filters they passed (code
   checked that); the AI's job is the judgement calls the data can't answer
   structurally — "does this company *directly provide* fraud detection?", "does it
   serve *European banks*?" — plus pulling evidence quotes. Then **plain code**:
   - checks every quote is literally in the real record (span-grounding); demotes any
     that aren't to "inference"
   - drops any candidate the AI said "no" to, or that has an unmet hard requirement
   - sorts the rest and takes the top 10

6. **relax_preferences** *(plain code, only sometimes)* — If there were too few strong
   matches *and* we haven't revised yet: loosen the **preferences only** by a fixed
   rule (e.g. widen the founding-year window), then go back to step 4 once more. Hard
   requirements and exclusions are never touched. This can happen at most once.

7. **compose_response** *(plain code)* — Build the final answer. Look up every returned
   company fresh from the database (hydrate). Assemble: the run ID, the interpreted
   mandate, the search plan, the ranked results (each with verdict, score, evidence,
   inferences, unmet preferences), whether a revision happened, and metadata (steps
   run, tools called and whether they succeeded, counts, AI call count, tokens, cost,
   time taken).

Cost of a run: impossible query = 1 AI call. Normal query = 2. Query that needed a
revision = 3. Hard cap = 5.

---

## 5. The 5 test queries — why each one exists

Each query is chosen to exercise a different behaviour:

- **Q1 — Finnish fintech, fraud/banking analytics.** The straightforward case. Checks:
  hard filters applied correctly, "founded after 2015 / under 250 employees" treated
  as *ranking preferences*, not filters.

- **Q2 — Nordic energy companies, energy forecasting, prefer 50–250 employees.**
  Checks the system does **not** filter on employee count here — the query explicitly
  says treat it as a preference. A system that filters on it fails.

- **Q3 — companies that directly serve European banks with fraud detection; exclude
  cybersecurity consultancies.** Two traps: (a) "directly provides" and "serves banks"
  can't be checked from structured fields — they're validation-time text judgements;
  (b) there are **no cybersecurity companies in the data at all**, so the exclusion
  removes nothing. The response must *say* "the exclusion criteria matched no
  candidates" — not silently imply it filtered some out.

- **Q4 — German drug-discovery companies, prefer founded after 2018; broaden the year
  if fewer than 3 strong matches.** Checks domain-term handling ("drug discovery" →
  industry Biotech + a topic filter, not a literal industry). The revision probably
  *won't* trigger here (plenty of matches) — that's expected.

- **Q5 — Finnish fintech, founded after 2022, more than 5,000 employees, revenue under
  €10M.** **Impossible** — max employee count in the data is 5,000. The feasibility
  gate must return an empty list + reason, using only 1 AI call. The "don't force
  matches" guarantee.

Two **supplements** cover paths the required 5 miss:

- **S1 — German drug-discovery founded after 2023.** Deliberately restrictive so the
  revision path actually fires and can be shown working.
- **S3 — German energy companies, exclude smart grid.** "Smart grid" is a real topic
  in the data, so this exclusion actually removes candidates — shows exclusion working
  (unlike Q3's empty one).

---

## 6. Build order (5 milestones)

You work one milestone at a time: finish it, run its tests, eyeball the result,
*then* start the next. Never batch them — you want to be at every checkpoint making
the call yourself.

| Milestone | What | Roughly |
|---|---|---|
| **M1 — Data foundation** | Load `companies.json` into SQLite + FTS5. Build the region map, revenue parser, domain-term map. Hand-write the eval oracle answers. | 3h |
| **M2 — Search tools** | The 3 search tools with input/output schemas + tests proving the filters are correct. No AI involved. | 2h |
| **M3 — AI layer + step logic** | The thin AI client + fake provider. Each of the 7 steps as a standalone function, tested against the fake. Schema validation + repair retry. One real AI call to sanity-check assumptions. | 4h |
| **M4 — Runner + CLI + API** | Wire the steps together (hand-written loop first, LangGraph second). The command-line tool. The web API. The limits + logging. | 4h |
| **M5 — Eval + deploy + README** | Run the sealed test queries end-to-end, produce the results table. Docker image. Deploy. Write the README. | 3h |

The estimates sum to ~20h; the 12–16h target is met by staying simple and cutting from
the bottom of the trim list, not by rushing.

---

## 7. What you're allowed to not finish

The brief literally says it doesn't expect a finished production platform and values
clear prioritisation over feature count. Three tiers:

- **Floor** — CLI + core workflow working locally on the hand-written loop; filters
  correct; evidence grounded; impossible query returns empty; limits enforced; results
  table for Q1–Q5; README stating what's cut. **This is a legitimate submission.**
- **Target** — Floor + the web API + LangGraph + structured logging + deployed live +
  the 2 supplements + full README.
- **Upside** — embeddings re-rank, extra endpoints, async API, a small UI. Only if
  everything above is done.

**Cut in this order** if time runs short: UI → extra endpoints → async → embeddings →
automated eval (hand-run instead) → LangGraph (ship the hand-written loop) →
deployment (local + a note) → supplement S3 → the revision ladder (one rule instead).

**Never cut:** the AI/plain-code split, the span-grounding check, the limits,
schema validation, the feasibility gate, Q1–Q5 in the results table, "no forced
matches," and a README that's honest about what was left out.

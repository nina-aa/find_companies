"""Workflow data models: the mandate, the search plan, per-run state, the
validation schemas, and the final response.

Two kinds of model live here:

* **LLM response schemas** — ``MandateCriteria`` (interpret) and ``ValidationBatch``
  (validate). Narrow, purpose-built, no open ``dict`` fields, safe for OpenAI
  strict ``response_format``. ``ValidationBatch`` deliberately carries *no* company
  fields — those are hydrated from the DB afterwards, so the model never supplies
  a record fact.
* **Internal state / transport** — everything else. Never sent to the model.
"""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas import Candidate, Company, Exclusions, SearchResult, StructuredFilters
from app.revenue import RevenueRange

Verdict = Literal["match", "partial", "no"]


# ==========================================================================
# interpret_mandate  —  LLM call #1 schema
# ==========================================================================
class MandateConstraints(BaseModel):
    """One side of the mandate (mandatory OR preferences), same shape for both."""

    countries: list[str] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)
    industries: list[str] = Field(default_factory=list)
    founded_year_gte: int | None = None
    founded_year_lte: int | None = None
    employee_count_gte: int | None = None
    employee_count_lte: int | None = None
    revenue_eur_gte: int | None = None
    revenue_eur_lte: int | None = None
    capabilities_any: list[str] = Field(default_factory=list)
    capabilities_all: list[str] = Field(default_factory=list)
    serves: list[str] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return self.model_dump(exclude_defaults=True) == {}


class MandateExclusions(BaseModel):
    categories: list[str] = Field(default_factory=list)   # semantic, for the validator
    keywords: list[str] = Field(default_factory=list)     # deterministic substring gate
    industries: list[str] = Field(default_factory=list)   # structured NOT IN


class MandateCriteria(BaseModel):
    """Structured form of the natural-language mandate (interpret_mandate output)."""

    mandatory: MandateConstraints = Field(default_factory=MandateConstraints)
    preferences: MandateConstraints = Field(default_factory=MandateConstraints)
    exclusions: MandateExclusions = Field(default_factory=MandateExclusions)
    semantic_focus: str = ""
    ambiguities: list[str] = Field(default_factory=list)


# ==========================================================================
# build_search_plan  —  DETERMINISTIC
# ==========================================================================
class RevisionStep(BaseModel):
    action: Literal[
        "widen_founded_year", "drop_founded_year_pref",
        "drop_employee_pref", "raise_limit",
    ]
    detail: str = ""


class RevisionPolicy(BaseModel):
    min_results: int = 3
    steps: list[RevisionStep] = Field(default_factory=list)
    max_revisions: int = 1


class SearchPlan(BaseModel):
    filters: StructuredFilters = Field(default_factory=StructuredFilters)
    topic_terms: list[str] = Field(default_factory=list)
    topic_mode: Literal["any", "all"] = "any"
    semantic_query: str | None = None
    exclusions: Exclusions = Field(default_factory=Exclusions)
    preferences: MandateConstraints = Field(default_factory=MandateConstraints)
    serves: list[str] = Field(default_factory=list)
    revision_policy: RevisionPolicy = Field(default_factory=RevisionPolicy)
    tool_sequence: list[str] = Field(default_factory=lambda: ["search_companies", "get_by_ids"])
    notes: list[str] = Field(default_factory=list)   # deterministic adjustments made


# ==========================================================================
# check_feasibility  —  DETERMINISTIC
# ==========================================================================
class FeasibilityResult(BaseModel):
    feasible: bool
    matched: int
    reason: str = ""


# ==========================================================================
# validate_and_rank  —  LLM call #2/#3 schema  (NO company fields)
# ==========================================================================
class TextFinding(BaseModel):
    """The model's judgement on ONE requirement it was asked about. The AND/OR
    logic over these is applied deterministically in code, not by the model."""

    requirement: str          # the exact capability / serve phrase it was given
    supported: bool
    source_field: str = ""    # "description" | "name" | "" if not quotable
    quote: str = ""           # verbatim substring of source_field (verified later)
    note: str = ""            # reasoning, esp. when supported is false or unquotable


class CandidateJudgement(BaseModel):
    candidate_id: int
    capability_findings: list[TextFinding] = Field(default_factory=list)
    serves_findings: list[TextFinding] = Field(default_factory=list)
    rationale: str = ""


class ValidationBatch(BaseModel):
    judgements: list[CandidateJudgement] = Field(default_factory=list)


# ==========================================================================
# post-validation internal + final response
# ==========================================================================
class Evidence(BaseModel):
    requirement: str
    source_field: str
    quote: str


class Inference(BaseModel):
    claim: str
    basis: str


class MatchScore(BaseModel):
    """How a company scores against the mandate. The ranking is a *tiered* sort on
    these fields, in order: all mandatory met → # mandatory met → # preferences met
    → keyword strength. `score` is a 0.6/0.3/0.1 weighted readout of the same
    parts — shown for convenience, not used as the sort key."""

    mandatory_met: int = 0
    mandatory_total: int = 0            # text requirements only (structural = the SQL gate)
    preferences_met: int = 0
    preferences_total: int = 0
    keyword_score: float | None = None  # min-max normalised bm25 in [0,1]; None if no topic
    score: float = 0.0
    llm_validated: bool = False         # were the text requirements checked by the model?

    @property
    def full_match(self) -> bool:
        return self.mandatory_met >= self.mandatory_total


class RankedCompany(BaseModel):
    company: Company
    match: MatchScore
    evidence: list[Evidence] = Field(default_factory=list)
    inferences: list[Inference] = Field(default_factory=list)
    unmet_preferences: list[str] = Field(default_factory=list)
    rationale: str = ""


class RevisionRecord(BaseModel):
    performed: bool = False
    relaxed: list[str] = Field(default_factory=list)
    reason: str = ""


class ToolCall(BaseModel):
    name: str
    ok: bool
    result_count: int = 0
    detail: str = ""


class StageRecord(BaseModel):
    stage: str
    ok: bool = True
    note: str = ""
    llm_calls: int = 0


class Funnel(BaseModel):
    """How many companies survive each narrowing stage."""

    mandatory_filters: int = 0     # structured WHERE (feasibility gate)
    topic_match: int = 0           # + FTS5 topic query
    after_exclusions: int = 0      # - exclusion gate
    retrieved_pool: int = 0        # - pool cap (<=100), bm25-ranked
    sent_to_validation: int = 0    # - validation batch cap (<=10)
    passed_validation: int = 0     # - LLM verdict + capability gate
    returned: int = 0              # - result cap (<=10)


class RunTrace(BaseModel):
    run_id: str
    funnel: Funnel = Field(default_factory=Funnel)
    stages: list[StageRecord] = Field(default_factory=list)
    tools: list[ToolCall] = Field(default_factory=list)
    llm_calls: int = 0
    llm_attempts: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    est_cost_usd: float = 0.0
    repairs: int = 0

    def add_stage(self, stage: str, *, ok: bool = True, note: str = "", llm_calls: int = 0):
        self.stages.append(StageRecord(stage=stage, ok=ok, note=note, llm_calls=llm_calls))

    def add_tool(self, name: str, *, ok: bool, result_count: int = 0, detail: str = ""):
        self.tools.append(ToolCall(name=name, ok=ok, result_count=result_count, detail=detail))

    def record_llm(self, result) -> None:
        self.llm_calls += 1
        self.llm_attempts += getattr(result, "attempts", 1)
        if getattr(result, "cached", False):
            self.cache_hits += 1
        if getattr(result, "repaired", False):
            self.repairs += 1
        u = result.usage
        self.prompt_tokens += u.prompt_tokens
        self.completion_tokens += u.completion_tokens
        self.est_cost_usd = round(self.est_cost_usd + u.est_cost_usd, 6)


class BudgetState(BaseModel):
    max_llm_calls: int = 5
    max_iterations: int = 2
    max_revisions: int = 1
    max_pool: int = 100
    max_validation_batch: int = 10
    max_results: int = 50
    llm_calls: int = 0
    iterations: int = 0
    revisions: int = 0


class MatchSummary(BaseModel):
    """Counts over the *whole* matched set, not just the returned slice."""

    matched_filters: int = 0            # pass the structured + topic filters
    matched_all_preferences: int = 0    # of those, also meet every preference (SQL count)
    matched_some_preferences: int = 0   # meet requirements, miss >= 1 preference
    sent_to_validation: int = 0         # top slice handed to the LLM
    validated_full: int = 0             # LLM confirmed every text requirement
    validated_gap: int = 0             # keyword-matched, a text/serves requirement unconfirmed
    returned: int = 0
    results_are_top_ranked: bool = True  # False -> every returned row ties; slice is arbitrary


class ResponseMetadata(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    stages_executed: list[str] = Field(default_factory=list)
    tools_called: list[ToolCall] = Field(default_factory=list)
    candidates_retrieved_per_iteration: list[int] = Field(default_factory=list)
    candidates_validated: int = 0
    funnel: Funnel = Field(default_factory=Funnel)
    validation_outcome: dict = Field(default_factory=dict)
    revised_search_performed: bool = False
    ranking_note: str = ""
    caveats: list[str] = Field(default_factory=list)
    match_summary: MatchSummary = Field(default_factory=MatchSummary)
    llm_calls: int = 0
    llm_attempts: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    est_cost_usd: float = 0.0
    latency_ms: int = 0
    model: str = ""
    provider: str = ""
    timed_out: bool = False
    stop_reason: str = ""
    cache_hits: int = 0
    revision: RevisionRecord | None = None


class ResultItem(BaseModel):
    rank: int
    company_id: int
    name: str
    industry: str | None
    location: str | None
    founded_year: int | None
    employee_count: int | None
    revenue_range: str | None
    mandatory_met: int
    mandatory_total: int
    preferences_met: int
    preferences_total: int
    keyword_score: float | None = None
    score: float = 0.0
    llm_validated: bool = False
    evidence: list[Evidence] = Field(default_factory=list)
    inferences: list[Inference] = Field(default_factory=list)
    unmet_preferences: list[str] = Field(default_factory=list)


class AgentResponse(BaseModel):
    run_id: str
    query: str
    interpreted_mandate: MandateCriteria
    search_plan: SearchPlan
    results: list[ResultItem] = Field(default_factory=list)
    revision: RevisionRecord = Field(default_factory=RevisionRecord)
    ambiguities: list[str] = Field(default_factory=list)   # mirror of the parse's ambiguities, surfaced
    empty_reason: str = ""
    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)


# ==========================================================================
# RunState — the object every node takes and returns
# ==========================================================================
class RunConfig(BaseModel):
    provider: Literal["openai", "fake"] = "fake"
    model: str = "gpt-4o-mini"
    use_cache: bool = True
    min_results: int = 3
    deadline_s: float = 90.0
    enable_embeddings: bool = False
    validation_batch: int = 10        # how many candidates the LLM judges (cost bound)
    result_limit: int = 10            # how many results to return (may exceed the batch)
    pool_limit: int = 100
    engine: Literal["driver", "graph"] = "driver"


class RunState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    query: str = ""
    cfg: RunConfig = Field(default_factory=RunConfig)
    started_ts: float = Field(default_factory=time.monotonic)
    deadline_ts: float | None = None

    criteria: MandateCriteria | None = None
    plan: SearchPlan | None = None
    feasibility: FeasibilityResult | None = None

    pool: list[Candidate] = Field(default_factory=list)
    last_search: SearchResult | None = None
    iteration: int = 0
    retrieved_per_iteration: list[int] = Field(default_factory=list)

    judgements: list[CandidateJudgement] = Field(default_factory=list)
    ranked: list[RankedCompany] = Field(default_factory=list)
    results: list[RankedCompany] = Field(default_factory=list)

    revision: RevisionRecord = Field(default_factory=RevisionRecord)
    budget: BudgetState = Field(default_factory=BudgetState)
    trace: RunTrace | None = None
    response: AgentResponse | None = None

    stop_reason: str = ""          # "" = ran to completion; else deadline/budget/degraded:<kind>
    timed_out: bool = False
    ranking_note: str = ""         # set when the returned slice is an arbitrary tie
    results_are_top_ranked: bool = True
    matched_all_preferences: int = 0

    def ensure_trace(self) -> RunTrace:
        if self.trace is None:
            self.trace = RunTrace(run_id=self.run_id)
        return self.trace

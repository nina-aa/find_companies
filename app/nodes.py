"""The seven workflow nodes as plain ``RunState -> RunState`` functions.

Two make one schema-constrained LLM call each (``interpret_mandate``,
``validate_and_rank``); the other five are deterministic. The runner that chains
them under BudgetGuard is C5 (M4) — here each node stands alone and is tested
against the fake provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app import config
from app.db import DEFAULT_DB
from app.llm import LLMClient, LLMError
from app.prompts import interpret_messages, validate_messages
from app.revenue import RevenueRange
from app.schemas import Exclusions, StructuredFilters
from app.state import (
    AgentResponse,
    Evidence,
    FeasibilityResult,
    Inference,
    MandateConstraints,
    MandateCriteria,
    RankedCompany,
    ResponseMetadata,
    ResultItem,
    RevisionRecord,
    RevisionPolicy,
    RevisionStep,
    RunState,
    SearchPlan,
    ValidationBatch,
)
from app.tools import count_matching, get_by_ids, search_companies

VERDICT_RANK = {"match": 2, "partial": 1, "no": 0}


@dataclass
class NodeDeps:
    llm: LLMClient
    db_path: Path | str = DEFAULT_DB


# ==========================================================================
# 1. interpret_mandate  —  LLM call #1
# ==========================================================================
def interpret_mandate(state: RunState, deps: NodeDeps) -> RunState:
    trace = state.ensure_trace()
    try:
        result = deps.llm.complete(interpret_messages(state.query), MandateCriteria)
    except LLMError as exc:
        trace.add_stage("interpret_mandate", ok=False, note=f"{exc.kind}: {exc}")
        raise
    state.criteria = result.parsed
    state.budget.llm_calls += 1
    trace.record_llm(result)
    trace.add_stage(
        "interpret_mandate", llm_calls=1,
        note=f"cached={result.cached} repaired={result.repaired}",
    )
    return state


# ==========================================================================
# 2. build_search_plan  —  DETERMINISTIC
# ==========================================================================
def build_search_plan(state: RunState, deps: NodeDeps) -> RunState:
    trace = state.ensure_trace()
    criteria = state.criteria or MandateCriteria()
    m = criteria.mandatory
    notes: list[str] = []

    # industries: validate against the 10-label enum
    industries: list[str] = []
    for raw in m.industries:
        canon = config.canonical_industry(raw)
        if canon:
            industries.append(canon)
        else:
            notes.append(f"dropped unknown industry {raw!r} (not one of the 10 labels)")

    # geography: resolve every country/region/alias term to concrete dataset
    # country names. "United Kingdom" -> "UK", "Nordic" -> FI/NO/SE, etc.
    countries: set[str] = set()
    empty_region_terms: list[str] = []
    for raw in list(m.countries) + list(m.regions):
        res = config.resolve_region(raw)
        if not res.known:
            notes.append(f"unknown location {raw!r} — recorded as ambiguity, not filtered")
        elif res.empty_region:
            empty_region_terms.append(raw)
            notes.append(f"region {raw!r} contains no companies in this dataset")
        else:
            countries.update(res.countries)

    # topic terms: capabilities_all takes precedence for validator intent
    topic_all, ind_all = _normalise_topics(m.capabilities_all, notes)
    topic_any, ind_any = _normalise_topics(m.capabilities_any, notes)
    topic_terms = _dedupe(topic_all + topic_any)
    topic_mode = "all" if topic_all else "any"

    # lexicon safety net: if no industry survived, infer from the capability terms,
    # then from the raw query text.
    if not industries:
        for cand_ind in ind_all + ind_any:
            industries.append(cand_ind)
            notes.append(f"industry {cand_ind!r} inferred from a capability phrase")
            break
    if not industries:
        for hit in config.lookup_phrases(state.query):
            if hit.industry:
                industries.append(hit.industry)
                notes.append(f"industry {hit.industry!r} inferred from lexicon phrase "
                             f"{hit.phrase!r}")
                break

    filters = StructuredFilters(
        countries=sorted(countries),
        regions=empty_region_terms,
        industries=_dedupe(industries),
        founded_year_gte=m.founded_year_gte,
        founded_year_lte=m.founded_year_lte,
        employee_count_gte=m.employee_count_gte,
        employee_count_lte=m.employee_count_lte,
        revenue=RevenueRange(min_eur=m.revenue_eur_gte, max_eur=m.revenue_eur_lte),
    )

    excl_industries = [config.canonical_industry(i) or i for i in criteria.exclusions.industries]
    exclusions = Exclusions(
        industries=_dedupe(excl_industries),
        keywords=_dedupe(criteria.exclusions.keywords),
    )

    steps: list[RevisionStep] = []
    p = criteria.preferences
    if p.founded_year_gte is not None or p.founded_year_lte is not None:
        steps.append(RevisionStep(action="widen_founded_year",
                                  detail="widen the founding-year preference by 5 years"))
        steps.append(RevisionStep(action="drop_founded_year_pref",
                                  detail="drop the founding-year preference entirely"))
    if p.employee_count_gte is not None or p.employee_count_lte is not None:
        steps.append(RevisionStep(action="drop_employee_pref",
                                  detail="drop the employee-count preference"))
    steps.append(RevisionStep(action="raise_limit",
                              detail="retrieve and validate a larger candidate pool"))

    state.plan = SearchPlan(
        filters=filters,
        topic_terms=topic_terms,
        topic_mode=topic_mode,
        semantic_query=criteria.semantic_focus or None,
        exclusions=exclusions,
        preferences=p,
        serves=_dedupe(m.serves + p.serves),
        revision_policy=RevisionPolicy(min_results=state.cfg.min_results, steps=steps),
        notes=notes,
    )
    trace.add_stage("build_search_plan", note="; ".join(notes) or "no adjustments")
    return state


_GENERIC_TAIL = ("technology", "technologies", "software", "solutions", "solution",
                 "platform", "platforms", "systems", "system", "services", "tools")


def _normalise_topics(terms, notes: list[str]) -> tuple[list[str], list[str]]:
    """Map free-text capability phrases onto the lexicon's canonical topic terms
    (so "fraud detection technology" -> "fraud detection"), and collect any
    industry each phrase implies. Unknown phrases are kept as-is."""
    topics: list[str] = []
    industries: list[str] = []
    for raw in terms:
        term = (raw or "").strip()
        if not term:
            continue
        words = term.split()
        if len(words) > 1 and words[-1].lower() in _GENERIC_TAIL:
            term = " ".join(words[:-1])
        hits = config.lookup_phrases(term)
        canon = next((h for h in hits if h.phrase.lower() in term.lower()), None)
        if canon and canon.topics:
            for t in canon.topics:
                if t not in topics:
                    topics.append(t)
            if canon.industry:
                industries.append(canon.industry)
            if term.lower() != raw.strip().lower() or canon.topics != [raw.strip()]:
                notes.append(f"capability {raw.strip()!r} -> {canon.topics}")
        else:
            if term not in topics:
                topics.append(term)
    return topics, industries


def _dedupe(items) -> list[str]:
    seen: dict[str, None] = {}
    for it in items:
        it = (it or "").strip()
        if it and it not in seen:
            seen[it] = None
    return list(seen)


# ==========================================================================
# 3. check_feasibility  —  DETERMINISTIC
# ==========================================================================
def check_feasibility(state: RunState, deps: NodeDeps) -> RunState:
    trace = state.ensure_trace()
    plan = state.plan or SearchPlan()
    matched = count_matching(plan.filters, db_path=deps.db_path)
    feasible = matched > 0
    reason = "" if feasible else (
        "no companies satisfy the mandatory criteria: "
        + _describe_filters(plan.filters)
    )
    state.feasibility = FeasibilityResult(feasible=feasible, matched=matched, reason=reason)
    trace.funnel.mandatory_filters = matched
    trace.add_tool("count_matching", ok=True, result_count=matched)
    trace.add_stage("check_feasibility",
                    note=f"{matched} rows match the mandatory filters"
                    + ("" if feasible else " -> infeasible, short-circuiting"))
    return state


def _describe_filters(f: StructuredFilters) -> str:
    parts: list[str] = []
    if f.industries:
        parts.append("industry in " + "/".join(f.industries))
    locs = f.resolved_locations()
    if locs:
        parts.append("location in " + "/".join(locs))
    if f.founded_year_gte:
        parts.append(f"founded >= {f.founded_year_gte}")
    if f.founded_year_lte:
        parts.append(f"founded <= {f.founded_year_lte}")
    if f.employee_count_gte:
        parts.append(f"employees >= {f.employee_count_gte}")
    if f.employee_count_lte:
        parts.append(f"employees <= {f.employee_count_lte}")
    if f.revenue.min_eur is not None:
        parts.append(f"revenue >= EUR {f.revenue.min_eur:,}")
    if f.revenue.max_eur is not None:
        parts.append(f"revenue <= EUR {f.revenue.max_eur:,}")
    return "; ".join(parts) or "(none)"


# ==========================================================================
# 4. retrieve  —  DETERMINISTIC tool call
# ==========================================================================
def retrieve(state: RunState, deps: NodeDeps) -> RunState:
    trace = state.ensure_trace()
    plan = state.plan or SearchPlan()
    pool_limit = min(state.cfg.pool_limit, state.budget.max_pool)

    result = search_companies(
        plan.filters,
        topic_terms=plan.topic_terms,
        exclusions=plan.exclusions,
        semantic_query=plan.semantic_query if state.cfg.enable_embeddings else None,
        limit=pool_limit,
        db_path=deps.db_path,
    )
    # bm25 is recall-oriented and preference-blind. Re-rank the pool by structured
    # preference fit (then bm25 order) so preference-satisfying companies reach the
    # validation batch instead of being truncated out. Preferences never filter.
    ordered = list(enumerate(result.candidates))
    ordered.sort(key=lambda pair: (
        -_preference_fit(pair[1], plan.preferences)[0],
        pair[0],
    ))
    for new_rank, (_, cand) in enumerate(ordered, start=1):
        cand.rank = new_rank
    state.pool = [cand for _, cand in ordered]
    state.last_search = result
    state.iteration += 1
    state.budget.iterations += 1
    state.retrieved_per_iteration.append(len(result.candidates))

    f = trace.funnel
    f.topic_match = result.matched_query
    f.after_exclusions = result.matched_query - result.excluded
    f.retrieved_pool = len(result.candidates)

    trace.add_tool(
        "search_companies", ok=True, result_count=len(result.candidates),
        detail=f"matched_filters={result.matched_filters} "
               f"topic_match={result.matched_query} excluded={result.excluded} "
               f"fts={result.fts_query!r}",
    )
    capped = " (pool cap)" if result.matched_query - result.excluded > len(result.candidates) else ""
    trace.add_stage(
        "retrieve",
        note=(f"iteration {state.iteration}: "
              f"{result.matched_filters} filtered -> {result.matched_query} topic"
              + (f" -> {f.after_exclusions} after exclusions" if result.excluded else "")
              + f" -> {len(result.candidates)} pool{capped}"),
    )
    return state


# ==========================================================================
# 5. validate_and_rank  —  LLM call #2 / #3  + deterministic post-processing
# ==========================================================================
def validate_and_rank(state: RunState, deps: NodeDeps) -> RunState:
    trace = state.ensure_trace()
    plan = state.plan or SearchPlan()
    criteria = state.criteria or MandateCriteria()
    batch = state.pool[: state.cfg.validation_batch]

    if not batch:
        state.ranked = []
        state.results = []
        trace.add_stage("validate_and_rank", note="no candidates to validate")
        return state

    try:
        result = deps.llm.complete(
            validate_messages(state.query, criteria, plan, batch), ValidationBatch
        )
    except LLMError as exc:
        trace.add_stage("validate_and_rank", ok=False, note=f"{exc.kind}: {exc}")
        raise
    state.budget.llm_calls += 1
    trace.record_llm(result)
    state.judgements = result.parsed.judgements

    companies = {c.id: c for c in get_by_ids([c.id for c in batch], db_path=deps.db_path)}
    fts_matched = {c.id: set(c.matched_topics) for c in batch}
    required_caps = list(plan.topic_terms)
    require_all = plan.topic_mode == "all"
    ranked: list[RankedCompany] = []
    demoted = 0

    for j in result.parsed.judgements:
        company = companies.get(j.candidate_id)
        if company is None:
            continue

        evidence: list[Evidence] = []
        inferences: list[Inference] = []

        def _record(finding, *, as_requirement: str):
            nonlocal demoted
            grounded = _span_grounded(finding.quote, finding.source_field, company)
            if finding.supported and finding.quote and grounded:
                evidence.append(Evidence(requirement=as_requirement,
                                         source_field=finding.source_field or "description",
                                         quote=finding.quote))
            elif finding.supported:
                if finding.quote and not grounded:
                    demoted += 1
                inferences.append(Inference(
                    claim=as_requirement,
                    basis=finding.note or ("unverified quote: " + finding.quote
                                           if finding.quote else "supported, no quote"),
                ))

        supported_caps = {f.requirement for f in j.capability_findings if f.supported}
        rejected_caps = {f.requirement for f in j.capability_findings if not f.supported}
        for f in j.capability_findings:
            _record(f, as_requirement=f"capability: {f.requirement}")
        for f in j.serves_findings:
            _record(f, as_requirement=f"serves: {f.requirement}")

        # FTS only rescues a capability the model did not mention at all — never one
        # it explicitly judged unsupported.
        cap_signal = supported_caps | (
            fts_matched.get(j.candidate_id, set()) - rejected_caps
        )

        # deterministic capability gate (the AND/OR logic lives here, not in the LLM)
        if not required_caps:
            cap_ok = True
        elif require_all:
            cap_ok = all(c in cap_signal for c in required_caps)
        else:
            cap_ok = bool(cap_signal & set(required_caps))

        serves_required = list(plan.serves)
        serves_ok = (not serves_required) or all(
            any(f.requirement == s and f.supported for f in j.serves_findings)
            for s in serves_required
        )

        if not cap_ok:
            continue
        verdict = "match" if serves_ok else "partial"
        met = len(supported_caps) + sum(1 for f in j.serves_findings if f.supported)

        # Structured preferences are scored deterministically (more reliable than the
        # model's relevance_score); non-structural preference signals come from the LLM.
        # structured preferences are scored + reported deterministically
        pref_score, pref_unmet = _preference_fit(company, plan.preferences)
        ranked.append(RankedCompany(
            company=company,
            verdict=verdict,
            relevance_score=max(0.0, min(1.0, j.relevance_score)),
            preference_score=pref_score,
            mandatory_met=met,
            evidence=evidence,
            inferences=inferences,
            unmet_preferences=pref_unmet,
            rationale=j.rationale,
        ))

    ranked.sort(key=lambda r: (
        VERDICT_RANK[r.verdict],
        r.preference_score,
        r.relevance_score,
        r.mandatory_met,
    ), reverse=True)

    state.ranked = ranked
    state.results = ranked[: state.cfg.result_limit]
    n_match = sum(1 for r in ranked if r.verdict == "match")

    f = trace.funnel
    f.sent_to_validation = len(batch)
    f.passed_validation = len(ranked)
    f.returned = len(state.results)

    trace.add_stage(
        "validate_and_rank", llm_calls=1,
        note=f"{len(batch)} sent -> {len(ranked)} passed ({n_match} match, "
             f"{len(ranked) - n_match} partial) -> {len(state.results)} returned; "
             f"{demoted} quote(s) demoted to inference",
    )
    return state


def _preference_fit(company, prefs) -> tuple[float, list[str]]:
    """Deterministic score in [0, 1] for how well a company's structured fields
    match the soft preferences, plus a list of the ones it misses. Preferences
    never filter — this only re-ranks."""
    checks: list[tuple[bool, str]] = []
    fy = company.founded_year
    ec = company.employee_count

    if prefs.founded_year_gte is not None:
        checks.append(((fy or 0) >= prefs.founded_year_gte,
                       f"founded before {prefs.founded_year_gte}"))
    if prefs.founded_year_lte is not None:
        checks.append(((fy or 9999) <= prefs.founded_year_lte,
                       f"founded after {prefs.founded_year_lte}"))
    if prefs.employee_count_gte is not None:
        checks.append(((ec or 0) >= prefs.employee_count_gte,
                       f"fewer than {prefs.employee_count_gte} employees"))
    if prefs.employee_count_lte is not None:
        checks.append(((ec or 10**9) <= prefs.employee_count_lte,
                       f"more than {prefs.employee_count_lte} employees"))
    if prefs.revenue_eur_lte is not None and company.revenue_min_eur is not None:
        checks.append((company.revenue_min_eur < prefs.revenue_eur_lte,
                       f"revenue above EUR {prefs.revenue_eur_lte:,}"))
    if prefs.revenue_eur_gte is not None and company.revenue_max_eur is not None:
        checks.append((company.revenue_max_eur >= prefs.revenue_eur_gte,
                       f"revenue below EUR {prefs.revenue_eur_gte:,}"))

    if not checks:
        return 1.0, []
    hits = sum(1 for ok, _ in checks if ok)
    return hits / len(checks), [msg for ok, msg in checks if not ok]


def _span_grounded(quote: str, source_field: str, company) -> bool:
    """A quote is grounded only if it is a literal substring of the named field
    of the real record (case-insensitive, whitespace-normalised)."""
    if not quote:
        return False
    value = getattr(company, source_field, None)
    if not isinstance(value, str):
        # try the obvious text fields if the model named something odd
        value = " ".join(str(getattr(company, f, "") or "")
                         for f in ("name", "description"))
    return _norm(quote) in _norm(value)


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


# ==========================================================================
# 6. relax_preferences  —  DETERMINISTIC
# ==========================================================================
def relax_preferences(state: RunState, deps: NodeDeps) -> RunState:
    trace = state.ensure_trace()
    plan = state.plan or SearchPlan()
    policy = plan.revision_policy

    if state.revision.performed or state.budget.revisions >= state.budget.max_revisions:
        trace.add_stage("relax_preferences", note="skipped — revision already performed")
        return state

    relaxed: list[str] = []
    prefs = plan.preferences.model_copy(deep=True)

    for step in policy.steps:
        if step.action == "widen_founded_year" and prefs.founded_year_gte is not None:
            prefs.founded_year_gte = max(1995, prefs.founded_year_gte - 5)
            relaxed.append(f"founded_year_gte -> {prefs.founded_year_gte}")
            break
        if step.action == "drop_founded_year_pref" and (
            prefs.founded_year_gte is not None or prefs.founded_year_lte is not None
        ):
            prefs.founded_year_gte = prefs.founded_year_lte = None
            relaxed.append("dropped founding-year preference")
            break
        if step.action == "drop_employee_pref" and (
            prefs.employee_count_gte is not None or prefs.employee_count_lte is not None
        ):
            prefs.employee_count_gte = prefs.employee_count_lte = None
            relaxed.append("dropped employee-count preference")
            break
        if step.action == "raise_limit":
            state.cfg.validation_batch = min(
                state.budget.max_validation_batch, state.cfg.validation_batch * 2
            )
            state.cfg.pool_limit = min(state.budget.max_pool, state.cfg.pool_limit * 2)
            relaxed.append(f"raised pool to {state.cfg.pool_limit}, "
                           f"validation batch to {state.cfg.validation_batch}")
            break

    plan.preferences = prefs
    state.revision = RevisionRecord(
        performed=True, relaxed=relaxed,
        reason=f"fewer than {policy.min_results} strong matches after iteration "
               f"{state.iteration}",
    )
    state.budget.revisions += 1
    trace.add_stage("relax_preferences", note="; ".join(relaxed) or "no applicable step")
    return state


# ==========================================================================
# 7. compose_response  —  DETERMINISTIC terminal
# ==========================================================================
def compose_response(state: RunState, deps: NodeDeps) -> RunState:
    trace = state.ensure_trace()
    criteria = state.criteria or MandateCriteria()
    plan = state.plan or SearchPlan()

    empty_reason = ""
    results: list[ResultItem] = []

    if state.feasibility and not state.feasibility.feasible:
        empty_reason = state.feasibility.reason
    elif state.stop_reason.startswith("degraded:") and not state.results:
        empty_reason = (
            "the run stopped early (" + state.stop_reason.split(":", 1)[1]
            + ") before candidates could be validated"
        )
    else:
        for i, r in enumerate(state.results, start=1):
            c = r.company
            results.append(ResultItem(
                rank=i, company_id=c.id, name=c.name, industry=c.industry,
                location=c.location, founded_year=c.founded_year,
                employee_count=c.employee_count, revenue_range=c.revenue_range,
                verdict=r.verdict, relevance_score=r.relevance_score,
                preference_score=r.preference_score,
                evidence=r.evidence, inferences=r.inferences,
                unmet_preferences=r.unmet_preferences,
            ))
        if not results and not empty_reason:
            empty_reason = "no candidate passed validation against the mandatory criteria"

    trace.add_stage("compose_response",
                    note=f"{len(results)} result(s)"
                    + (f"; empty: {empty_reason}" if empty_reason else ""))

    n_match = sum(1 for r in state.ranked if r.verdict == "match")
    n_partial = sum(1 for r in state.ranked if r.verdict == "partial")
    metadata = ResponseMetadata(
        stages_executed=[s.stage for s in trace.stages],
        tools_called=list(trace.tools),
        candidates_retrieved_per_iteration=list(state.retrieved_per_iteration),
        candidates_validated=len(state.judgements),
        funnel=trace.funnel,
        validation_outcome={
            "matched": n_match,
            "partial": n_partial,
            "rejected": max(0, len(state.judgements) - len(state.ranked)),
            "empty_reason": empty_reason or None,
        },
        revised_search_performed=state.revision.performed,
        revision=state.revision,
        llm_calls=trace.llm_calls,
        llm_attempts=trace.llm_attempts,
        cache_hits=trace.cache_hits,
        prompt_tokens=trace.prompt_tokens,
        completion_tokens=trace.completion_tokens,
        est_cost_usd=trace.est_cost_usd,
        model=state.cfg.model,
        provider=state.cfg.provider,
        timed_out=state.timed_out,
        stop_reason=state.stop_reason,
    )

    state.response = AgentResponse(
        run_id=state.run_id,
        query=state.query,
        interpreted_mandate=criteria,
        search_plan=plan,
        results=results,
        revision=state.revision,
        empty_reason=empty_reason,
        metadata=metadata,
    )
    return state

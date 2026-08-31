"""The seven workflow nodes as plain ``RunState -> RunState`` functions.

Two make one schema-constrained LLM call each (``interpret_mandate``,
``validate_and_rank``); the other five are deterministic. ``run_workflow``
([app/workflow.py]) chains them under BudgetGuard — each node here also stands
alone and is tested against the fake provider.
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
    MatchScore,
    MatchSummary,
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

    # A capability phrase can carry a strong industry signal ("fraud detection" ->
    # Fintech). Merge those in — the model sometimes names a different or no
    # industry for a phrase like "fraud-detection technology". Broadening the
    # industry IN-set is recall-safe; the validator narrows on the text.
    for cand_ind in ind_all + ind_any:
        if cand_ind not in industries:
            industries.append(cand_ind)
            notes.append(f"industry {cand_ind!r} added from a capability phrase")

    # last resort: infer from the raw query text
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

    # A "serves <region> <customer>" phrase ("European banks") hints the company
    # probably operates in that region — but the mandate never said so. Add the
    # region as a location PREFERENCE, never a filter: region-based firms rank
    # higher, without excluding a company that serves the region from elsewhere.
    p = criteria.preferences
    serves_terms = _dedupe(m.serves + p.serves)
    pref_regions = list(p.regions)
    if not countries and not empty_region_terms:
        have = {r.lower() for r in pref_regions}
        for phrase in serves_terms:
            for token in phrase.replace("-", " ").split():
                res = config.resolve_region(token)
                if res.known and res.countries and token.lower() not in have:
                    pref_regions.append(token.lower())
                    have.add(token.lower())
                    notes.append(f"serves {phrase!r} -> {token.lower()!r} added as a "
                                 f"location preference (not a filter)")
    plan_preferences = (p.model_copy(update={"regions": pref_regions})
                        if pref_regions != list(p.regions) else p)

    steps: list[RevisionStep] = []
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
        preferences=plan_preferences,
        serves=serves_terms,
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
        term = config.normalise_spelling((raw or "").strip()).replace("-", " ")
        term = " ".join(term.split())
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

    # bm25 (used for the LIMIT above) is preference-blind and near-random on this
    # templated text, so it can truncate out companies that meet every preference.
    # Run a second search that ALSO filters on the preferences, and merge its hits
    # into the front of the pool so they survive the cap. Preferences still never
    # filter the final result — this only decides who reaches validation.
    candidates = list(result.candidates)
    if plan.preferences.is_empty():
        state.matched_all_preferences = result.matched_query - result.excluded
    else:
        pref_res = search_companies(
            _with_preferences(plan.filters, plan.preferences),
            topic_terms=plan.topic_terms, exclusions=plan.exclusions,
            limit=min(pool_limit, 50), db_path=deps.db_path,
        )
        state.matched_all_preferences = pref_res.matched_query - pref_res.excluded
        seen, merged = set(), []
        for c in list(pref_res.candidates) + candidates:      # preferred first
            if c.id not in seen and len(merged) < pool_limit:
                seen.add(c.id)
                merged.append(c)
        candidates = merged

    # Re-rank the combined pool by structured preference fit so the strongest
    # candidates reach the validation batch first.
    ordered = list(enumerate(candidates))
    ordered.sort(key=lambda pair: (-_preference_fraction(pair[1], plan.preferences), pair[0]))
    for new_rank, (_, cand) in enumerate(ordered, start=1):
        cand.rank = new_rank
    state.pool = [cand for _, cand in ordered]
    state.last_search = result
    state.iteration += 1
    state.budget.iterations += 1
    state.retrieved_per_iteration.append(len(state.pool))

    f = trace.funnel
    f.topic_match = result.matched_query
    f.after_exclusions = result.matched_query - result.excluded
    f.retrieved_pool = len(state.pool)

    trace.add_tool(
        "search_companies", ok=True, result_count=len(state.pool),
        detail=f"matched_filters={result.matched_filters} "
               f"topic_match={result.matched_query} excluded={result.excluded} "
               f"pref_perfect={state.matched_all_preferences} fts={result.fts_query!r}",
    )
    capped = " (pool cap)" if f.after_exclusions > len(state.pool) else ""
    trace.add_stage(
        "retrieve",
        note=(f"iteration {state.iteration}: "
              f"{result.matched_filters} filtered -> {result.matched_query} topic"
              + (f" -> {f.after_exclusions} after exclusions" if result.excluded else "")
              + f" -> {len(state.pool)} pool{capped}"),
    )
    return state


# ==========================================================================
# 5. validate_and_rank  —  LLM call #2 / #3  + deterministic scoring of the pool
# ==========================================================================
SCORE_WEIGHTS = (0.65, 0.35)   # mandatory · preference  (readout only; bm25 breaks ties)


def _structural_requirement_count(plan) -> int:
    """Mandatory requirements enforced by the SQL WHERE gate — one per distinct
    constraint. A returned row satisfies all of them by construction; they are
    counted so `mandatory X/Y` reflects the *whole* mandate, not just the text
    part."""
    f = plan.filters
    n = 0
    if f.countries or f.regions:
        n += 1                       # location
    if f.industries:
        n += 1                       # industry
    for bound in (f.founded_year_gte, f.founded_year_lte, f.employee_count_gte,
                  f.employee_count_lte, f.revenue.min_eur, f.revenue.max_eur):
        if bound is not None:
            n += 1
    n += len(plan.exclusions.industries) + len(plan.exclusions.keywords)
    return n


def validate_and_rank(state: RunState, deps: NodeDeps) -> RunState:
    trace = state.ensure_trace()
    plan = state.plan or SearchPlan()
    criteria = state.criteria or MandateCriteria()
    pool = list(state.pool)

    if not pool:
        state.ranked = []
        state.results = []
        trace.add_stage("validate_and_rank", note="no candidates to validate")
        return state

    batch = pool[: state.cfg.validation_batch]
    batch_ids = {c.id for c in batch}
    companies = {c.id: c for c in get_by_ids([c.id for c in pool], db_path=deps.db_path)}
    bm25_by_id = {c.id: c.bm25_score for c in pool}
    fts_matched = {c.id: set(c.matched_topics) for c in pool}

    required_caps = list(plan.topic_terms)
    require_all = plan.topic_mode == "all"
    serves_required = list(plan.serves)
    # mandatory_total = structural clauses (always met for a returned row) + the
    # capability requirement (1 for an OR-group, one per term for an AND-group) +
    # one per `serves` phrase. A `serves` requirement counts toward the total but
    # is rarely met on this dataset (no customer field), so a query like Q3 lands
    # at e.g. 3/4 — `full_match` is False and a caveat explains why.
    struct_reqs = _structural_requirement_count(plan)
    cap_req_total = 0 if not required_caps else (len(required_caps) if require_all else 1)
    mandatory_total = struct_reqs + cap_req_total + len(serves_required)

    is_structural = (not required_caps and not serves_required
                     and not criteria.exclusions.categories)

    # --- the LLM judges only the top slice, and only when there is text to judge ---
    judgements_by_id: dict[int, object] = {}
    llm_calls_made = 0
    if batch and not is_structural:
        try:
            result = deps.llm.complete(
                validate_messages(state.query, criteria, plan, batch), ValidationBatch
            )
        except LLMError as exc:
            trace.add_stage("validate_and_rank", ok=False, note=f"{exc.kind}: {exc}")
            raise
        state.budget.llm_calls += 1
        llm_calls_made = 1
        trace.record_llm(result)
        state.judgements = list(result.parsed.judgements)
        judgements_by_id = {j.candidate_id: j for j in result.parsed.judgements}

    ranked: list[RankedCompany] = []
    demoted = 0

    for cand in pool:
        company = companies.get(cand.id)
        if company is None:
            continue
        j = judgements_by_id.get(cand.id)
        in_batch = cand.id in batch_ids
        fts_here = fts_matched.get(cand.id, set())
        evidence: list[Evidence] = []
        inferences: list[Inference] = []

        # --- capability signal ---
        supported = {f.requirement for f in j.capability_findings if f.supported} if j else set()
        rejected = {f.requirement for f in j.capability_findings if not f.supported} if j else set()
        if j:
            for f in j.capability_findings:
                field = _resolve_source_field(f.quote, company)
                if f.supported and f.quote and field:
                    evidence.append(Evidence(requirement=f"capability: {f.requirement}",
                                             source_field=field, quote=f.quote))
                elif f.supported:
                    if f.quote and not field:
                        demoted += 1
                    inferences.append(Inference(
                        claim=f"capability: {f.requirement}",
                        basis=f.note or ("unverified quote: " + f.quote if f.quote
                                         else "supported, no quote")))
        cap_signal = (supported | (fts_here - rejected)) if j else fts_here

        if not required_caps:
            cap_ok, cap_met = True, 0
        elif require_all:
            hits = [c for c in required_caps if c in cap_signal]
            cap_ok, cap_met = len(hits) == len(required_caps), len(hits)
        else:
            cap_ok = bool(cap_signal & set(required_caps))
            cap_met = 1 if cap_ok else 0
        if not cap_ok:
            continue   # does not do the thing at all — not a match

        # --- serves (customer/market): counts toward the mandatory total. Rarely
        #     met here — the one-sentence descriptions do not name customers — so an
        #     unmet one is spelled out as an inference and compose_response adds a
        #     caveat. A model-supported one only counts if it grounds in the text.
        serves_met = 0
        for s in serves_required:
            finding = next((f for f in (j.serves_findings if j else []) if f.requirement == s), None)
            field = _resolve_source_field(finding.quote, company) if finding else None
            if finding and finding.supported and finding.quote and field:
                serves_met += 1
                evidence.append(Evidence(requirement=f"serves: {s}",
                                         source_field=field, quote=finding.quote))
            elif finding and finding.supported:
                inferences.append(Inference(claim=f"serves: {s}",
                    basis=finding.note or "stated as served, but no verifiable quote in the record"))
            else:
                inferences.append(Inference(claim=f"serves: {s}", basis=(
                    (finding.note if finding else "")
                    or "not stated — this dataset has no customer field to confirm it")))

        pref_met, pref_total, pref_unmet = _preference_fit(company, plan.preferences)
        ranked.append(RankedCompany(
            company=company,
            match=MatchScore(
                mandatory_met=struct_reqs + cap_met + serves_met,
                mandatory_total=mandatory_total,
                preferences_met=pref_met,
                preferences_total=pref_total,
                llm_validated=in_batch and j is not None,
            ),
            evidence=evidence,
            inferences=inferences,
            unmet_preferences=pref_unmet,
            rationale=(j.rationale if j else
                       ("all structured criteria met; no text judgement required"
                        if is_structural else "")),
        ))

    note = "purely structural — LLM validation skipped" if is_structural else ""
    _finalise_ranking(state, trace, ranked, bm25_by_id,
                      sent=len(batch) if llm_calls_made else 0,
                      llm_calls=llm_calls_made, demoted=demoted, note=note)
    return state


def _finalise_ranking(state, trace, ranked, bm25_by_id, *,
                      sent: int, llm_calls: int, demoted: int, note: str = "") -> None:
    """Score + **tiered** sort + write results.

    Sort priority (each a pure tiebreaker, no weights):
        1. every mandatory requirement met?
        2. how many mandatory requirements met
        3. how many soft preferences met
        4. bm25 keyword strength — near-noise on this templated data, so it only
           ever separates an exact tie; never shown, never in the score.
    `match.score` = 0.65·(mandatory fit) + 0.35·(preference fit) — a readout.
    """
    bvals = [b for b in (bm25_by_id.get(r.company.id) for r in ranked) if b is not None]
    span = (max(bvals) - min(bvals)) if len(bvals) > 1 else 0.0
    worst = max(bvals) if bvals else 0.0
    wm, wp = SCORE_WEIGHTS
    for r in ranked:
        b = bm25_by_id.get(r.company.id)
        r.match.keyword_score = (None if (b is None or span <= 1e-9)
                                 else round((worst - b) / span, 3))
        m = r.match
        mand_frac = 1.0 if m.mandatory_total == 0 else m.mandatory_met / m.mandatory_total
        pref_frac = 1.0 if m.preferences_total == 0 else m.preferences_met / m.preferences_total
        m.score = round(wm * mand_frac + wp * pref_frac, 3)

    def _tier(r):
        m = r.match
        return (m.mandatory_met >= m.mandatory_total, m.mandatory_met, m.preferences_met)

    ranked.sort(key=lambda r: (
        *_tier(r),
        (r.match.keyword_score if r.match.keyword_score is not None else 0.0),
        -r.company.id,
    ), reverse=True)

    limit = min(state.cfg.result_limit, state.budget.max_results)
    state.ranked = ranked
    state.results = ranked[:limit]

    f = trace.funnel
    f.sent_to_validation = sent
    f.passed_validation = len(ranked)
    f.returned = len(state.results)

    # arbitrary-slice flag: every returned row sits in the same tier (bm25 noise
    # ignored) and more companies are in that tier than we return
    matched_total = f.after_exclusions or f.topic_match or f.mandatory_filters
    tiers = {_tier(r) for r in state.results}
    top = state.results[0].match if state.results else None
    returned_full = bool(top and top.full_match)
    returned_all_prefs = bool(top and top.preferences_total
                              and top.preferences_met >= top.preferences_total)
    # when the returned rows all satisfy every preference, the tie is over the
    # SQL "meets every preference" count; otherwise over the whole matched set.
    tier_size = state.matched_all_preferences if returned_all_prefs else matched_total
    state.results_are_top_ranked = not (len(tiers) == 1 and tier_size > len(state.results))
    if not state.results_are_top_ranked:
        if returned_full and returned_all_prefs:
            what = "meet every requirement and preference"
        elif returned_full:
            what = "meet every requirement"
        elif returned_all_prefs:
            what = "meet every preference (and every verifiable requirement)"
        else:
            what = "match equally on every ranking criterion"
        state.ranking_note = (
            f"{tier_size} companies {what}; the {len(state.results)} returned are an "
            f"arbitrary slice — add a preference (size, recency, a narrower "
            f"capability) to rank within them"
        )

    full = sum(1 for r in ranked if r.match.full_match)
    detail = note or (
        f"{sent} to LLM -> {len(ranked)} kept ({full} full, {len(ranked) - full} with a "
        f"text/serves gap) -> {len(state.results)} returned; {demoted} quote(s) -> inference"
    )
    trace.add_stage("validate_and_rank", llm_calls=llm_calls, note=detail)


def _preference_fraction(company, prefs) -> float:
    met, total, _ = _preference_fit(company, prefs)
    return 1.0 if total == 0 else met / total


def _with_preferences(f, p):
    """Fold the soft preference bounds into a StructuredFilters, for a "how many
    also meet every preference" SQL count. Takes the tighter of each bound."""
    def tighter(a, b, fn):
        vals = [v for v in (a, b) if v is not None]
        return fn(vals) if vals else None

    return f.model_copy(update=dict(
        countries=sorted(set(f.countries) | set(p.countries)),
        regions=sorted(set(f.regions) | set(p.regions)),
        founded_year_gte=tighter(f.founded_year_gte, p.founded_year_gte, max),
        founded_year_lte=tighter(f.founded_year_lte, p.founded_year_lte, min),
        employee_count_gte=tighter(f.employee_count_gte, p.employee_count_gte, max),
        employee_count_lte=tighter(f.employee_count_lte, p.employee_count_lte, min),
        revenue=RevenueRange(
            min_eur=tighter(f.revenue.min_eur, p.revenue_eur_gte, max),
            max_eur=tighter(f.revenue.max_eur, p.revenue_eur_lte, min),
        ),
    ))


def _preference_fit(company, prefs) -> tuple[int, int, list[str]]:
    """(# preferences met, # preferences total, [labels of the misses]).
    Preferences never filter — this only re-ranks."""
    checks: list[tuple[bool, str]] = []
    fy = company.founded_year
    ec = company.employee_count

    pref_locs = StructuredFilters(
        countries=list(prefs.countries), regions=list(prefs.regions)
    ).resolved_locations()
    if pref_locs:
        checks.append((company.location in pref_locs,
                       "not located in " + "/".join(pref_locs)))

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

    met = sum(1 for ok, _ in checks if ok)
    return met, len(checks), [msg for ok, msg in checks if not ok]


def _resolve_source_field(quote: str, company) -> str | None:
    """Which real text field of the record contains ``quote`` verbatim (case- and
    whitespace-insensitive). Returns ``None`` if the quote is not grounded anywhere
    — the model's own ``source_field`` label is not trusted (it sometimes echoes a
    payload key)."""
    if not quote:
        return None
    q = _norm(quote)
    for field in ("description", "name"):
        value = getattr(company, field, "") or ""
        if q in _norm(str(value)):
            return field
    return None


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
            c, m = r.company, r.match
            results.append(ResultItem(
                rank=i, company_id=c.id, name=c.name, industry=c.industry,
                location=c.location, founded_year=c.founded_year,
                employee_count=c.employee_count, revenue_range=c.revenue_range,
                mandatory_met=m.mandatory_met, mandatory_total=m.mandatory_total,
                preferences_met=m.preferences_met, preferences_total=m.preferences_total,
                keyword_score=m.keyword_score, score=m.score, llm_validated=m.llm_validated,
                evidence=r.evidence, inferences=r.inferences,
                unmet_preferences=r.unmet_preferences,
            ))
        if not results and not empty_reason:
            empty_reason = "no candidate passed validation against the mandatory criteria"

    trace.add_stage("compose_response",
                    note=f"{len(results)} result(s)"
                    + (f"; empty: {empty_reason}" if empty_reason else ""))

    caveats: list[str] = []
    if plan.serves:
        caveats.append(
            "the mandate asks about a customer / market ("
            + "; ".join(f"serves: {s}" for s in plan.serves)
            + ") — this dataset has no customer field and the one-sentence "
            "descriptions do not name customers, so that requirement cannot be "
            "verified. Results match the other criteria; any customer mention "
            "found is recorded as evidence but not scored."
        )
    caveats.extend(n for n in plan.notes if "unknown location" in n or "no companies" in n)

    full = sum(1 for r in state.ranked if r.match.full_match)
    matched_total = (trace.funnel.after_exclusions or trace.funnel.topic_match
                     or trace.funnel.mandatory_filters)
    summary = MatchSummary(
        matched_filters=matched_total,
        matched_all_preferences=min(state.matched_all_preferences, matched_total),
        matched_some_preferences=max(0, matched_total - state.matched_all_preferences),
        sent_to_validation=trace.funnel.sent_to_validation,
        validated_full=sum(1 for r in state.ranked if r.match.llm_validated and r.match.full_match),
        validated_gap=sum(1 for r in state.ranked if r.match.llm_validated and not r.match.full_match),
        returned=len(results),
        results_are_top_ranked=state.results_are_top_ranked,
    )
    metadata = ResponseMetadata(
        stages_executed=[s.stage for s in trace.stages],
        tools_called=list(trace.tools),
        candidates_retrieved_per_iteration=list(state.retrieved_per_iteration),
        candidates_validated=len(state.judgements),
        funnel=trace.funnel,
        match_summary=summary,
        validation_outcome={
            "full_match": full,
            "with_text_gap": len(state.ranked) - full,
            "empty_reason": empty_reason or None,
        },
        revised_search_performed=state.revision.performed,
        ranking_note=state.ranking_note,
        caveats=caveats,
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
        ambiguities=list(criteria.ambiguities),
        empty_reason=empty_reason,
        metadata=metadata,
    )
    return state

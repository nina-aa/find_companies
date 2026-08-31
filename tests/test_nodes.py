"""Each workflow node in isolation, against the fixture index + fake provider."""

import pytest

from app.llm import FakeProvider, LLMClient
from app.nodes import (
    NodeDeps,
    build_search_plan,
    check_feasibility,
    compose_response,
    interpret_mandate,
    relax_preferences,
    retrieve,
    validate_and_rank,
)
from app.state import (
    CandidateJudgement,
    MandateConstraints,
    MandateCriteria,
    MandateExclusions,
    RunConfig,
    RunState,
    TextFinding,
    ValidationBatch,
)


def _deps(fixture_db, *, handler=None, responses=None):
    return NodeDeps(
        llm=LLMClient(FakeProvider(handler=handler, responses=responses)),
        db_path=fixture_db,
    )


def _cap(term, supported=True, quote=""):
    return TextFinding(requirement=term, supported=supported,
                       source_field="description" if quote else "", quote=quote)


def _state(query="q", **cfg):
    return RunState(query=query, cfg=RunConfig(**cfg))


# --------------------------------------------------------------------------- #
# interpret_mandate
# --------------------------------------------------------------------------- #
def test_interpret_sets_criteria_and_records_call(fixture_db):
    canned = MandateCriteria(
        mandatory=MandateConstraints(countries=["Finland"], industries=["Fintech"]),
        semantic_focus="finnish fintech",
    )
    state = interpret_mandate(_state(), _deps(fixture_db, responses={MandateCriteria: canned}))
    assert state.criteria.mandatory.countries == ["Finland"]
    assert state.budget.llm_calls == 1
    assert state.trace.llm_calls == 1
    assert state.trace.stages[-1].stage == "interpret_mandate"


# --------------------------------------------------------------------------- #
# build_search_plan  (deterministic)
# --------------------------------------------------------------------------- #
def test_plan_validates_industries_against_enum(fixture_db):
    state = _state()
    state.criteria = MandateCriteria(
        mandatory=MandateConstraints(industries=["Fintech", "Cybersecurity"])
    )
    state = build_search_plan(state, _deps(fixture_db))
    assert state.plan.filters.industries == ["Fintech"]
    assert any("Cybersecurity" in n for n in state.plan.notes)


def test_plan_flags_unknown_region_but_does_not_filter(fixture_db):
    state = _state()
    state.criteria = MandateCriteria(mandatory=MandateConstraints(regions=["Atlantis"]))
    state = build_search_plan(state, _deps(fixture_db))
    assert state.plan.filters.regions == []
    assert any("Atlantis" in n for n in state.plan.notes)


def test_plan_flags_empty_region(fixture_db):
    state = _state()
    state.criteria = MandateCriteria(mandatory=MandateConstraints(regions=["Baltics"]))
    state = build_search_plan(state, _deps(fixture_db))
    assert "Baltics" in state.plan.filters.regions
    assert any("no companies" in n for n in state.plan.notes)


def test_plan_capabilities_all_sets_topic_mode(fixture_db):
    state = _state()
    state.criteria = MandateCriteria(
        mandatory=MandateConstraints(capabilities_all=["drug discovery", "gene editing"])
    )
    state = build_search_plan(state, _deps(fixture_db))
    assert state.plan.topic_mode == "all"
    assert set(state.plan.topic_terms) == {"drug discovery", "gene editing"}


def test_plan_lexicon_safety_net_infers_industry(fixture_db):
    state = _state(query="companies doing drug discovery")
    state.criteria = MandateCriteria()          # LLM left everything empty
    state = build_search_plan(state, _deps(fixture_db))
    assert state.plan.filters.industries == ["Biotech"]
    assert any("lexicon" in n for n in state.plan.notes)


def test_plan_revenue_bounds_become_range(fixture_db):
    state = _state()
    state.criteria = MandateCriteria(
        mandatory=MandateConstraints(revenue_eur_lte=10_000_000)
    )
    state = build_search_plan(state, _deps(fixture_db))
    assert state.plan.filters.revenue.max_eur == 10_000_000


def test_plan_serves_region_becomes_location_preference_not_filter(fixture_db):
    state = _state()
    state.criteria = MandateCriteria(
        mandatory=MandateConstraints(industries=["Fintech"],
                                     capabilities_any=["fraud detection"],
                                     serves=["European banks"]),
    )
    state = build_search_plan(state, _deps(fixture_db))
    assert not state.plan.filters.countries                      # never a hard filter
    assert not state.plan.filters.resolved_locations()
    assert "european" in [r.lower() for r in state.plan.preferences.regions]
    assert any("location preference" in n for n in state.plan.notes)


def test_plan_serves_region_not_added_when_company_location_given(fixture_db):
    state = _state()
    state.criteria = MandateCriteria(
        mandatory=MandateConstraints(countries=["Finland"], industries=["Fintech"],
                                     capabilities_any=["fraud detection"],
                                     serves=["European banks"]),
    )
    state = build_search_plan(state, _deps(fixture_db))
    assert state.plan.filters.countries == ["Finland"]
    assert not state.plan.preferences.regions                    # company location was stated


# --------------------------------------------------------------------------- #
# check_feasibility
# --------------------------------------------------------------------------- #
def test_feasibility_feasible(fixture_db):
    state = _state()
    state.criteria = MandateCriteria(
        mandatory=MandateConstraints(countries=["Finland"], industries=["Fintech"])
    )
    state = build_search_plan(state, _deps(fixture_db))
    state = check_feasibility(state, _deps(fixture_db))
    assert state.feasibility.feasible is True
    assert state.feasibility.matched == 4


def test_feasibility_infeasible_has_reason(fixture_db):
    state = _state()
    state.criteria = MandateCriteria(
        mandatory=MandateConstraints(countries=["Finland"], industries=["Fintech"],
                                     employee_count_gte=5001)
    )
    state = build_search_plan(state, _deps(fixture_db))
    state = check_feasibility(state, _deps(fixture_db))
    assert state.feasibility.feasible is False
    assert state.feasibility.matched == 0
    assert "employees >= 5001" in state.feasibility.reason


# --------------------------------------------------------------------------- #
# retrieve
# --------------------------------------------------------------------------- #
def test_retrieve_populates_pool_and_iteration(fixture_db):
    state = _state()
    state.criteria = MandateCriteria(
        mandatory=MandateConstraints(countries=["Finland"], industries=["Fintech"],
                                     capabilities_any=["fraud detection"])
    )
    state = build_search_plan(state, _deps(fixture_db))
    state = retrieve(state, _deps(fixture_db))
    assert {c.id for c in state.pool} == {1, 3}
    assert state.iteration == 1
    assert state.retrieved_per_iteration == [2]
    assert state.last_search.fts_query == '"fraud detection"'


# --------------------------------------------------------------------------- #
# validate_and_rank
# --------------------------------------------------------------------------- #
def _fraud_state(fixture_db):
    state = _state()
    state.criteria = MandateCriteria(
        mandatory=MandateConstraints(countries=["Finland"], industries=["Fintech"],
                                     capabilities_any=["fraud detection"])
    )
    state = build_search_plan(state, _deps(fixture_db))
    return retrieve(state, _deps(fixture_db))


def test_validate_keeps_supported_capability_drops_rejected(fixture_db):
    state = _fraud_state(fixture_db)
    batch = ValidationBatch(judgements=[
        CandidateJudgement(candidate_id=1,
                           capability_findings=[_cap("fraud detection", quote="fraud detection")]),
        # id 3 explicitly judged unsupported -> FTS must not rescue it
        CandidateJudgement(candidate_id=3,
                           capability_findings=[_cap("fraud detection", supported=False)]),
    ])
    state = validate_and_rank(state, _deps(fixture_db, responses={ValidationBatch: batch}))
    assert [r.company.id for r in state.results] == [1]
    m = state.results[0].match
    assert m.mandatory_met == m.mandatory_total == 3   # location + industry + capability
    assert state.results[0].evidence[0].quote == "fraud detection"


def test_validate_span_grounding_demotes_ungrounded_quote(fixture_db):
    state = _fraud_state(fixture_db)
    batch = ValidationBatch(judgements=[
        CandidateJudgement(candidate_id=1, capability_findings=[
            _cap("fraud detection", quote="fraud detection"),
            _cap("banking analytics", quote="quantum blockchain casino"),
        ]),
    ])
    state = validate_and_rank(state, _deps(fixture_db, responses={ValidationBatch: batch}))
    r = next(x for x in state.ranked if x.company.id == 1)
    assert [e.quote for e in r.evidence] == ["fraud detection"]
    assert any("quantum blockchain casino" in inf.basis for inf in r.inferences)


def test_preferences_met_fraction_and_ranking(fixture_db):
    state = _state()
    state.criteria = MandateCriteria(
        mandatory=MandateConstraints(countries=["Finland"], industries=["Fintech"],
                                     capabilities_any=["fraud detection"]),
        preferences=MandateConstraints(founded_year_gte=2020),   # id 1 (2019) fails, id 3 (2021) passes
    )
    state = build_search_plan(state, _deps(fixture_db))
    state = retrieve(state, _deps(fixture_db))
    batch = ValidationBatch(judgements=[
        CandidateJudgement(candidate_id=1,
                           capability_findings=[_cap("fraud detection", quote="fraud detection")]),
        CandidateJudgement(candidate_id=3,
                           capability_findings=[_cap("fraud detection", quote="fraud detection")]),
    ])
    state = validate_and_rank(state, _deps(fixture_db, responses={ValidationBatch: batch}))
    by_id = {r.company.id: r for r in state.ranked}
    assert (by_id[3].match.preferences_met, by_id[3].match.preferences_total) == (1, 1)
    assert (by_id[1].match.preferences_met, by_id[1].match.preferences_total) == (0, 1)
    assert by_id[1].unmet_preferences == ["founded before 2020"]
    assert [r.company.id for r in state.results] == [3, 1]   # more preferences met ranks first


def test_location_preference_is_scored_in_preference_fit(fixture_db):
    state = _fraud_state(fixture_db)
    state.plan.preferences = MandateConstraints(regions=["nordic"])   # FI/NO/SE
    batch = ValidationBatch(judgements=[
        CandidateJudgement(candidate_id=1,
                           capability_findings=[_cap("fraud detection", quote="fraud detection")]),
    ])
    state = validate_and_rank(state, _deps(fixture_db, responses={ValidationBatch: batch}))
    r = next(x for x in state.ranked if x.company.id == 1)
    assert (r.match.preferences_met, r.match.preferences_total) == (1, 1)   # Finland ∈ Nordic


def test_purely_structural_query_skips_the_llm(fixture_db):
    state = _state()
    state.criteria = MandateCriteria(
        mandatory=MandateConstraints(countries=["Finland"], industries=["Fintech"])
    )
    state = build_search_plan(state, _deps(fixture_db))
    state = retrieve(state, _deps(fixture_db))

    calls = {"n": 0}

    def handler(messages, model):
        calls["n"] += 1
        return ValidationBatch()

    state = validate_and_rank(state, _deps(fixture_db, handler=handler))
    assert calls["n"] == 0                              # LLM never invoked
    assert state.budget.llm_calls == 0
    assert {r.company.id for r in state.results} == {1, 2, 3, 4}
    assert all(r.match.full_match for r in state.results)
    # location + industry, no capability -> 2/2 for every row
    assert all((r.match.mandatory_met, r.match.mandatory_total) == (2, 2) for r in state.ranked)
    assert state.trace.stages[-1].note == "purely structural — LLM validation skipped"


def test_keyword_exclusion_query_still_skips_llm(fixture_db):
    """S3 shape: exclusion is a keyword (deterministic), no capability -> still skip."""
    state = _state()
    state.criteria = MandateCriteria(
        mandatory=MandateConstraints(countries=["Germany"], industries=["Energy"]),
        exclusions=MandateExclusions(keywords=["smart grid"]),
    )
    state = build_search_plan(state, _deps(fixture_db))
    state = retrieve(state, _deps(fixture_db))
    assert 8 not in {c.id for c in state.pool}          # Berlin GridSense excluded
    state = validate_and_rank(state, _deps(fixture_db))  # must not need the LLM
    assert state.budget.llm_calls == 0
    assert 9 in {r.company.id for r in state.results}    # Hamburg Volt kept


def test_semantic_exclusion_category_still_uses_llm(fixture_db):
    state = _fraud_state(fixture_db)
    state.plan.serves = []
    # a semantic category the model must judge -> do NOT skip
    state.criteria.exclusions = MandateExclusions(categories=["consumer lending apps"])
    batch = ValidationBatch(judgements=[
        CandidateJudgement(candidate_id=1,
                           capability_findings=[_cap("fraud detection", quote="fraud detection")]),
    ])
    state = validate_and_rank(state, _deps(fixture_db, responses={ValidationBatch: batch}))
    assert state.budget.llm_calls == 1


def test_serves_counts_toward_mandatory_total_but_is_usually_unmet(fixture_db):
    state = _fraud_state(fixture_db)
    state.plan.serves = ["European banks"]
    batch = ValidationBatch(judgements=[
        CandidateJudgement(candidate_id=1,
                           capability_findings=[_cap("fraud detection", quote="fraud detection")],
                           serves_findings=[TextFinding(requirement="European banks",
                                                        supported=False, note="not stated")]),
        CandidateJudgement(candidate_id=3,
                           capability_findings=[_cap("fraud detection", quote="fraud detection")]),
    ])
    state = validate_and_rank(state, _deps(fixture_db, responses={ValidationBatch: batch}))
    r = next(x for x in state.ranked if x.company.id == 1)
    # serves now counts: location + industry + capability + serves = 4 total, 3 met
    assert (r.match.mandatory_met, r.match.mandatory_total) == (3, 4)
    assert not r.match.full_match
    assert any("serves: European banks" in i.claim for i in r.inferences)
    state.feasibility = None
    state = compose_response(state, _deps(fixture_db))
    assert any("customer / market" in c for c in state.response.metadata.caveats)


def test_serves_grounded_quote_counts_as_met(fixture_db):
    state = _fraud_state(fixture_db)
    state.plan.serves = ["banks"]
    batch = ValidationBatch(judgements=[
        CandidateJudgement(candidate_id=1,
            capability_findings=[_cap("fraud detection", quote="fraud detection")],
            # "fraud detection" is a real substring of company 1's description
            serves_findings=[TextFinding(requirement="banks", supported=True,
                                         source_field="description", quote="fraud detection")]),
    ])
    state = validate_and_rank(state, _deps(fixture_db, responses={ValidationBatch: batch}))
    r = next(x for x in state.ranked if x.company.id == 1)
    assert (r.match.mandatory_met, r.match.mandatory_total) == (4, 4)
    assert r.match.full_match


def test_validate_drops_candidate_with_no_capability_support(fixture_db):
    state = _fraud_state(fixture_db)
    batch = ValidationBatch(judgements=[
        CandidateJudgement(candidate_id=1,
                           capability_findings=[_cap("fraud detection", supported=False)]),
        CandidateJudgement(candidate_id=3,
                           capability_findings=[_cap("fraud detection", supported=False)]),
    ])
    state = validate_and_rank(state, _deps(fixture_db, responses={ValidationBatch: batch}))
    assert state.results == []


def test_validate_empty_batch_is_graceful(fixture_db):
    state = _state()
    state.criteria = MandateCriteria(
        mandatory=MandateConstraints(countries=["Finland"], industries=["Fintech"],
                                     capabilities_any=["energy forecasting"])
    )
    state = build_search_plan(state, _deps(fixture_db))
    state = retrieve(state, _deps(fixture_db))
    assert state.pool == []
    state = validate_and_rank(state, _deps(fixture_db))   # must not call the LLM / crash
    assert state.results == []


# --------------------------------------------------------------------------- #
# relax_preferences
# --------------------------------------------------------------------------- #
def test_relax_widens_founded_year_and_records_revision(fixture_db):
    state = _state()
    state.criteria = MandateCriteria(
        mandatory=MandateConstraints(industries=["Biotech"]),
        preferences=MandateConstraints(founded_year_gte=2024),
    )
    state = build_search_plan(state, _deps(fixture_db))
    state = relax_preferences(state, _deps(fixture_db))
    assert state.revision.performed is True
    assert state.plan.preferences.founded_year_gte == 2019
    assert state.budget.revisions == 1


def test_relax_skipped_when_already_performed(fixture_db):
    state = _state()
    state.criteria = MandateCriteria(preferences=MandateConstraints(founded_year_gte=2024))
    state = build_search_plan(state, _deps(fixture_db))
    state = relax_preferences(state, _deps(fixture_db))
    state = relax_preferences(state, _deps(fixture_db))    # second call is a no-op
    assert state.budget.revisions == 1


# --------------------------------------------------------------------------- #
# compose_response
# --------------------------------------------------------------------------- #
def test_compose_infeasible_sets_empty_reason(fixture_db):
    state = _state()
    state.criteria = MandateCriteria(
        mandatory=MandateConstraints(countries=["Finland"], industries=["Fintech"],
                                     employee_count_gte=5001)
    )
    state = build_search_plan(state, _deps(fixture_db))
    state = check_feasibility(state, _deps(fixture_db))
    state = compose_response(state, _deps(fixture_db))
    assert state.response.results == []
    assert "employees >= 5001" in state.response.empty_reason
    assert state.response.metadata.validation_outcome["empty_reason"]


def test_compose_with_results_builds_result_items(fixture_db):
    state = _fraud_state(fixture_db)
    state.plan.preferences = MandateConstraints(founded_year_gte=2020)  # id 1 founded 2019
    batch = ValidationBatch(judgements=[
        CandidateJudgement(candidate_id=1,
                           capability_findings=[_cap("fraud detection", quote="fraud detection")]),
        CandidateJudgement(candidate_id=3,
                           capability_findings=[_cap("fraud detection", quote="fraud detection")]),
    ])
    state = validate_and_rank(state, _deps(fixture_db, responses={ValidationBatch: batch}))
    state.feasibility = None
    state = compose_response(state, _deps(fixture_db))
    item = next(x for x in state.response.results if x.company_id == 1)
    assert item.name == "Helsinki Fraud Labs"
    assert item.evidence[0].quote == "fraud detection"
    assert (item.mandatory_met, item.mandatory_total) == (3, 3)   # location + industry + capability
    assert (item.preferences_met, item.preferences_total) == (0, 1)
    assert item.unmet_preferences == ["founded before 2020"]
    assert item.llm_validated is True
    assert state.response.metadata.stages_executed[-1] == "compose_response"
    assert state.response.metadata.match_summary.returned == len(state.response.results)

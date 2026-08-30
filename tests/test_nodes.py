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
    PreferenceSignal,
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
        CandidateJudgement(candidate_id=1, relevance_score=0.9,
                           capability_findings=[_cap("fraud detection", quote="fraud detection")]),
        # id 3 explicitly judged unsupported -> FTS must not rescue it
        CandidateJudgement(candidate_id=3, relevance_score=0.1,
                           capability_findings=[_cap("fraud detection", supported=False)]),
    ])
    state = validate_and_rank(state, _deps(fixture_db, responses={ValidationBatch: batch}))
    assert [r.company.id for r in state.results] == [1]
    assert state.results[0].verdict == "match"
    assert state.results[0].evidence[0].quote == "fraud detection"


def test_validate_span_grounding_demotes_ungrounded_quote(fixture_db):
    state = _fraud_state(fixture_db)
    batch = ValidationBatch(judgements=[
        CandidateJudgement(candidate_id=1, relevance_score=0.8, capability_findings=[
            _cap("fraud detection", quote="fraud detection"),
            _cap("banking analytics", quote="quantum blockchain casino"),
        ]),
    ])
    state = validate_and_rank(state, _deps(fixture_db, responses={ValidationBatch: batch}))
    r = state.results[0]
    assert [e.quote for e in r.evidence] == ["fraud detection"]
    assert any("quantum blockchain casino" in inf.basis for inf in r.inferences)


def test_validate_serves_unmet_makes_partial_not_dropped(fixture_db):
    state = _fraud_state(fixture_db)
    state.plan.serves = ["European banks"]
    batch = ValidationBatch(judgements=[
        CandidateJudgement(candidate_id=1, relevance_score=0.5,
                           capability_findings=[_cap("fraud detection", quote="fraud detection")],
                           serves_findings=[TextFinding(requirement="European banks",
                                                        supported=False, note="not stated")]),
    ])
    state = validate_and_rank(state, _deps(fixture_db, responses={ValidationBatch: batch}))
    assert [r.company.id for r in state.results] == [1]
    assert state.results[0].verdict == "partial"


def test_validate_drops_candidate_with_no_capability_support(fixture_db):
    state = _fraud_state(fixture_db)
    batch = ValidationBatch(judgements=[
        CandidateJudgement(candidate_id=1, relevance_score=0.5,
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
        CandidateJudgement(candidate_id=1, relevance_score=0.9,
                           capability_findings=[_cap("fraud detection", quote="fraud detection")]),
    ])
    state = validate_and_rank(state, _deps(fixture_db, responses={ValidationBatch: batch}))
    state.feasibility = None
    state = compose_response(state, _deps(fixture_db))
    item = state.response.results[0]
    assert item.rank == 1 and item.company_id == 1
    assert item.name == "Helsinki Fraud Labs"
    assert item.evidence[0].quote == "fraud detection"
    assert item.unmet_preferences == ["founded before 2020"]
    assert state.response.metadata.stages_executed[-1] == "compose_response"

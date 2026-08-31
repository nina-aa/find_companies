"""run_workflow end to end on both engines: happy path, abstention, revision,
and the budget bounds actually holding under pressure."""

import time

import pytest

from app.llm import FakeProvider, LLMClient
from app.state import (
    CandidateJudgement,
    MandateConstraints,
    MandateCriteria,
    RunConfig,
    TextFinding,
    ValidationBatch,
)
from app.workflow import run_workflow

ENGINES = ["driver", "graph"]


def scripted(mandate: MandateCriteria, validations: list[ValidationBatch]):
    """A fake LLM that answers interpret with `mandate` and successive validate
    calls from `validations` (repeating the last one if it runs out)."""
    queue = list(validations)

    def handler(messages, model):
        if model is MandateCriteria:
            return mandate
        if model is ValidationBatch:
            return queue.pop(0) if len(queue) > 1 else (queue[0] if queue else ValidationBatch())
        raise AssertionError(f"unexpected model {model}")

    return LLMClient(FakeProvider(handler=handler))


def _finnish_fraud():
    return MandateCriteria(
        mandatory=MandateConstraints(countries=["Finland"], industries=["Fintech"],
                                     capabilities_any=["fraud detection"]),
    )


def _match(cid):
    return CandidateJudgement(
        candidate_id=cid,
        capability_findings=[TextFinding(requirement="fraud detection", supported=True,
                                         source_field="description", quote="fraud detection")],
    )


# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("engine", ENGINES)
def test_happy_path(engine, fixture_db):
    client = scripted(_finnish_fraud(), [ValidationBatch(judgements=[_match(1), _match(3)])])
    resp, state = run_workflow(
        "finnish fintech fraud detection",
        RunConfig(provider="fake", engine=engine, min_results=1),
        db_path=fixture_db, llm_client=client,
    )
    # both fixture rows match "fraud detection"; exact intra-tier order is bm25-driven
    assert {r.company_id for r in resp.results} == {1, 3}
    assert all(r.mandatory_met == r.mandatory_total for r in resp.results)
    assert resp.metadata.llm_calls == 2                 # interpret + one validate
    assert resp.metadata.stages_executed[-1] == "compose_response"
    assert "relax_preferences" not in resp.metadata.stages_executed
    assert resp.metadata.revised_search_performed is False
    assert resp.empty_reason == ""


@pytest.mark.parametrize("engine", ENGINES)
def test_infeasible_short_circuits_before_validation(engine, fixture_db):
    mandate = MandateCriteria(
        mandatory=MandateConstraints(countries=["Finland"], industries=["Fintech"],
                                     employee_count_gte=5001)
    )
    client = scripted(mandate, [])
    resp, state = run_workflow("impossible", RunConfig(provider="fake", engine=engine),
                               db_path=fixture_db, llm_client=client)
    assert resp.results == []
    assert "employees >= 5001" in resp.empty_reason
    assert resp.metadata.llm_calls == 1                 # interpret only
    assert "retrieve" not in resp.metadata.stages_executed


@pytest.mark.parametrize("engine", ENGINES)
def test_revision_path_runs_once(engine, fixture_db):
    mandate = _finnish_fraud()
    mandate.preferences = MandateConstraints(founded_year_gte=2024)
    # first validate: nothing matches -> triggers revision; second: one match
    client = scripted(mandate, [
        ValidationBatch(judgements=[]),
        ValidationBatch(judgements=[_match(1)]),
    ])
    resp, state = run_workflow(
        "finnish fintech fraud detection preferably founded after 2024",
        RunConfig(provider="fake", engine=engine, min_results=3),
        db_path=fixture_db, llm_client=client,
    )
    assert resp.metadata.revised_search_performed is True
    assert state.revision.relaxed                       # something was relaxed
    assert resp.metadata.stages_executed.count("retrieve") == 2
    assert resp.metadata.llm_calls == 3                 # interpret + 2 validates
    assert state.budget.revisions == 1


@pytest.mark.parametrize("engine", ENGINES)
def test_llm_call_budget_bounds_the_run(engine, fixture_db):
    client = scripted(_finnish_fraud(), [ValidationBatch(judgements=[])])
    cfg = RunConfig(provider="fake", engine=engine, min_results=99)  # force endless revision hunger
    resp, state = run_workflow("x", cfg, db_path=fixture_db, llm_client=client)
    assert state.budget.llm_calls <= state.budget.max_llm_calls
    assert state.budget.iterations <= state.budget.max_iterations
    assert state.budget.revisions <= state.budget.max_revisions
    # ran to a real end, not an exception
    assert resp.metadata.stages_executed[-1] == "compose_response"


@pytest.mark.parametrize("engine", ENGINES)
def test_deadline_returns_partial_not_exception(engine, fixture_db):
    client = scripted(_finnish_fraud(), [ValidationBatch(judgements=[_match(1)])])
    cfg = RunConfig(provider="fake", engine=engine, deadline_s=-1.0)  # already expired
    resp, state = run_workflow("x", cfg, db_path=fixture_db, llm_client=client)
    assert state.timed_out is True
    assert resp.metadata.stop_reason == "deadline"
    assert resp.metadata.stages_executed[-1] == "compose_response"


@pytest.mark.parametrize("engine", ENGINES)
def test_llm_error_degrades_gracefully(engine, fixture_db):
    def handler(messages, model):
        from app.llm import LLMError
        raise LLMError("boom", kind="provider")

    client = LLMClient(FakeProvider(handler=handler))
    resp, state = run_workflow("x", RunConfig(provider="fake", engine=engine),
                               db_path=fixture_db, llm_client=client)
    assert state.stop_reason == "degraded:provider"
    assert resp.results == []
    assert resp.metadata.stages_executed[-1] == "compose_response"


def test_both_engines_agree(fixture_db):
    outs = {}
    for engine in ENGINES:
        client = scripted(_finnish_fraud(), [ValidationBatch(judgements=[_match(1), _match(3)])])
        resp, _ = run_workflow("x", RunConfig(provider="fake", engine=engine, min_results=1),
                               db_path=fixture_db, llm_client=client)
        outs[engine] = ([r.company_id for r in resp.results], resp.metadata.llm_calls,
                        resp.metadata.stages_executed)
    assert outs["driver"] == outs["graph"]

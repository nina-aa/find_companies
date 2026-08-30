"""Opt-in live smoke test — the one place the real paid API is exercised in the
test suite. Skipped unless RUN_LIVE=1 and an OpenAI key is available.

    RUN_LIVE=1 python -m pytest tests/test_live_smoke.py -q -s

Cost: one gpt-4o-mini structured call, well under $0.001.
"""

import os

import pytest

from app import config

config.load_env()

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE") != "1" or not os.environ.get("OPENAI_API_KEY"),
    reason="set RUN_LIVE=1 and OPENAI_API_KEY to run the live smoke test",
)


def test_interpret_mandate_real_call():
    from app.llm import build_client
    from app.nodes import NodeDeps, build_search_plan, check_feasibility, interpret_mandate
    from app.state import RunConfig, RunState

    cfg = RunConfig(provider="openai", use_cache=False)
    deps = NodeDeps(llm=build_client("openai", model=cfg.model, use_cache=False))

    state = RunState(
        query=("Find fintech companies in Finland working on fraud detection or "
               "banking analytics. Prefer companies founded after 2015 with fewer "
               "than 250 employees."),
        cfg=cfg,
    )
    state = interpret_mandate(state, deps)
    state = build_search_plan(state, deps)
    state = check_feasibility(state, deps)

    m = state.criteria.mandatory
    # a country may land in countries or regions — both resolve to a location filter
    assert "Finland" in (m.countries + m.regions)
    assert "Fintech" in m.industries
    assert {"fraud detection", "banking analytics"} & set(
        m.capabilities_any + m.capabilities_all
    )
    # "prefer ... founded after 2015 ... fewer than 250" -> preferences, not mandatory
    assert state.criteria.preferences.founded_year_gte in (2015, 2016)
    assert m.founded_year_gte is None

    # the plan must still resolve Finland to a location filter
    assert state.plan.filters.resolved_locations() == ["Finland"]
    assert state.feasibility.feasible is True
    assert state.trace.est_cost_usd > 0
    print(f"\nlive: {state.trace.prompt_tokens}+{state.trace.completion_tokens} tok, "
          f"${state.trace.est_cost_usd:.5f}, plan.industries={state.plan.filters.industries}")

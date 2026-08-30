"""LangGraph port of the workflow.

Same seven node functions as the custom driver ([app/nodes.py]), same
``BudgetGuard`` — LangGraph only supplies the graph structure (explicit nodes,
two conditional edges, a bounded loop). The state is one Pydantic ``RunState``
carried under a single key, so the nodes are used unchanged (no partial-dict
reducers to reason about). ``recursion_limit`` is a secondary backstop; the guard
is the real bound.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.budget import BudgetGuard
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
from app.state import RunState

RECURSION_LIMIT = 15


class GraphState(TypedDict):
    rs: RunState


def _wrap(fn, name: str, guard: BudgetGuard, log):
    def node(gs: GraphState) -> GraphState:
        rs = gs["rs"]
        guard.before(name)
        log.info("stage.enter", stage=name, llm_calls=rs.budget.llm_calls,
                 iteration=rs.iteration, time_left_s=_round(guard.time_left()))
        fn(rs, node.deps)
        return {"rs": rs}

    node.deps = None
    return node


def _after_feasibility(gs: GraphState) -> str:
    rs = gs["rs"]
    return "retrieve" if (rs.feasibility and rs.feasibility.feasible) else "compose"


def _after_validate(gs: GraphState) -> str:
    rs = gs["rs"]
    usable = sum(1 for r in rs.ranked if r.verdict in ("match", "partial"))
    needs_revision = (
        usable < rs.cfg.min_results
        and not rs.revision.performed
        and rs.budget.revisions < rs.budget.max_revisions
    )
    return "relax" if needs_revision else "compose"


def build_graph(deps: NodeDeps, guard: BudgetGuard, log):
    nodes = {
        "interpret": _wrap(interpret_mandate, "interpret_mandate", guard, log),
        "plan": _wrap(build_search_plan, "build_search_plan", guard, log),
        "feasibility": _wrap(check_feasibility, "check_feasibility", guard, log),
        "retrieve": _wrap(retrieve, "retrieve", guard, log),
        "validate": _wrap(validate_and_rank, "validate_and_rank", guard, log),
        "relax": _wrap(relax_preferences, "relax_preferences", guard, log),
        "compose": _wrap(compose_response, "compose_response", guard, log),
    }
    for n in nodes.values():
        n.deps = deps

    g = StateGraph(GraphState)
    for name, node in nodes.items():
        g.add_node(name, node)

    g.add_edge(START, "interpret")
    g.add_edge("interpret", "plan")
    g.add_edge("plan", "feasibility")
    g.add_conditional_edges("feasibility", _after_feasibility,
                            {"retrieve": "retrieve", "compose": "compose"})
    g.add_edge("retrieve", "validate")
    g.add_conditional_edges("validate", _after_validate,
                            {"relax": "relax", "compose": "compose"})
    g.add_edge("relax", "retrieve")
    g.add_edge("compose", END)
    return g.compile()


def run_graph(state: RunState, deps: NodeDeps, guard: BudgetGuard, log) -> None:
    compiled = build_graph(deps, guard, log)
    compiled.invoke({"rs": state}, {"recursion_limit": RECURSION_LIMIT})


def _round(x):
    return None if x is None else round(x, 1)

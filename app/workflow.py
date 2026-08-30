"""``run_workflow`` — the single entry point the CLI, the HTTP API and the eval
harness all call.

It builds the initial ``RunState``, runs the seven nodes under ``BudgetGuard``,
and returns the composed ``AgentResponse`` (plus the ``RunState`` for callers that
want the full trace). Two execution engines share this signature:

* ``driver``  — a ~40-line bounded loop in this file. The safety net; always works.
* ``graph``   — the LangGraph port in ``app/graph.py``. Same nodes, same guard.

``cfg.engine`` selects; ``driver`` is the default.
"""

from __future__ import annotations

import time
from pathlib import Path

from app.budget import BudgetExceeded, BudgetGuard
from app.db import DEFAULT_DB
from app.llm import LLMClient, LLMError, build_client
from app.logging import get_logger
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
from app.state import AgentResponse, RunConfig, RunState


def run_workflow(
    query: str,
    cfg: RunConfig | None = None,
    *,
    db_path: Path | str = DEFAULT_DB,
    llm_client: LLMClient | None = None,
) -> tuple[AgentResponse, RunState]:
    cfg = cfg or RunConfig()
    state = RunState(query=query, cfg=cfg)
    state.deadline_ts = state.started_ts + cfg.deadline_s
    trace = state.ensure_trace()
    log = get_logger(state.run_id)

    client = llm_client or build_client(
        cfg.provider, model=cfg.model, use_cache=cfg.use_cache
    )
    deps = NodeDeps(llm=client, db_path=db_path)
    guard = BudgetGuard(state.budget, state.deadline_ts)
    guard.clamp_config(cfg)

    log.info("run.start", query=query, provider=cfg.provider, engine=cfg.engine,
             deadline_s=cfg.deadline_s)

    engine = _run_graph if cfg.engine == "graph" else _run_driver
    try:
        engine(state, deps, guard, log)
    except BudgetExceeded as exc:
        state.stop_reason = "deadline" if exc.kind == "deadline" else f"budget:{exc.kind}"
        state.timed_out = exc.kind == "deadline"
        log.warning("run.budget_exceeded", kind=exc.kind, detail=str(exc))
    except LLMError as exc:
        state.stop_reason = f"degraded:{exc.kind}"
        log.error("run.degraded", kind=exc.kind, detail=str(exc))

    if state.response is None:
        compose_response(state, deps)

    trace.est_cost_usd = round(trace.est_cost_usd, 6)
    state.response.metadata.latency_ms = int((time.monotonic() - state.started_ts) * 1000)
    _log_summary(log, state)
    return state.response, state


# --------------------------------------------------------------------------- #
# custom bounded driver
# --------------------------------------------------------------------------- #
def _step(fn, name, state, deps, guard, log):
    guard.before(name)                       # may raise BudgetExceeded
    log.info("stage.enter", stage=name,
             llm_calls=state.budget.llm_calls, iteration=state.iteration,
             time_left_s=_round(guard.time_left()))
    return fn(state, deps)


def _enough(state: RunState) -> bool:
    """Enough *usable* results to skip revision: full matches, plus partials (on
    this dataset a thin description often caps a genuine fit at 'partial')."""
    usable = sum(1 for r in state.ranked if r.verdict in ("match", "partial"))
    return usable >= state.cfg.min_results


def _run_driver(state: RunState, deps: NodeDeps, guard: BudgetGuard, log) -> None:
    _step(interpret_mandate, "interpret_mandate", state, deps, guard, log)
    _step(build_search_plan, "build_search_plan", state, deps, guard, log)
    _step(check_feasibility, "check_feasibility", state, deps, guard, log)

    if not (state.feasibility and state.feasibility.feasible):
        _step(compose_response, "compose_response", state, deps, guard, log)
        return

    # retrieve -> validate, with at most one bounded revision in between
    max_passes = state.budget.max_iterations
    for _ in range(max_passes):
        _step(retrieve, "retrieve", state, deps, guard, log)
        _step(validate_and_rank, "validate_and_rank", state, deps, guard, log)

        if _enough(state) or state.revision.performed:
            break
        try:
            _step(relax_preferences, "relax_preferences", state, deps, guard, log)
        except BudgetExceeded as exc:
            log.info("revision.skipped", reason=exc.kind)
            break
        if not state.revision.performed:      # ladder had no applicable step
            break

    _step(compose_response, "compose_response", state, deps, guard, log)


# --------------------------------------------------------------------------- #
# langgraph engine
# --------------------------------------------------------------------------- #
def _run_graph(state: RunState, deps: NodeDeps, guard: BudgetGuard, log) -> None:
    from app.graph import run_graph

    run_graph(state, deps, guard, log)


# --------------------------------------------------------------------------- #
def _round(x):
    return None if x is None else round(x, 1)


def _log_summary(log, state: RunState) -> None:
    t = state.trace
    meta = state.response.metadata
    log.info("run.funnel", **t.funnel.model_dump())
    log.info(
        "run.done",
        results=len(state.response.results),
        empty_reason=state.response.empty_reason or None,
        stop_reason=state.stop_reason or None,
        llm_calls=t.llm_calls, llm_attempts=t.llm_attempts, cache_hits=t.cache_hits,
        repairs=t.repairs,
        prompt_tokens=t.prompt_tokens, completion_tokens=t.completion_tokens,
        est_cost_usd=t.est_cost_usd,
        revision=state.revision.performed,
        latency_ms=meta.latency_ms,
    )

"""Assessment-level evaluation harness.

Runs each query in ``eval/queries.yaml`` through ``run_workflow`` and scores the
**deterministic** columns against the hand-authored ``expected`` block: parsed
criteria, mandatory-filter correctness (re-checked against the DB), dataset
membership, evidence grounding, revision flag, correct abstention, budget
adherence. Ranking quality / evidence sufficiency stay a manual column.

A correct abstention (Q5: ``results == []`` + reason, ``should_be_empty: true``)
scores as PASS (lesson 11).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from app import config
from app.db import DEFAULT_DB
from app.nodes import _norm
from app.revenue import RevenueRange
from app.schemas import StructuredFilters
from app.state import RunConfig
from app.tools import get_by_ids
from app.workflow import run_workflow

QUERIES_PATH = config.REPO_ROOT / "eval" / "queries.yaml"
RESULTS_PATH = config.REPO_ROOT / "eval" / "RESULTS.md"


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class QueryResult:
    qid: str
    checks: list[Check] = field(default_factory=list)
    n_retrieved: int = 0
    n_returned: int = 0
    llm_calls: int = 0
    est_cost_usd: float = 0.0
    latency_ms: int = 0
    revised: bool = False
    empty: bool = False
    note: str = ""

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks)


def _expected_filters(expected_mandatory: dict) -> StructuredFilters:
    em = expected_mandatory or {}
    fy = em.get("founded_year", {}) or {}
    ec = em.get("employee_count", {}) or {}
    rev = em.get("revenue_eur", {}) or {}
    return StructuredFilters(
        countries=em.get("countries", []),
        regions=em.get("regions", []),
        industries=em.get("industries", []),
        founded_year_gte=em.get("founded_year_gte", fy.get("gte")),
        founded_year_lte=em.get("founded_year_lte", fy.get("lte")),
        employee_count_gte=em.get("employee_count_gte", ec.get("gte")),
        employee_count_lte=em.get("employee_count_lte", ec.get("lte")),
        revenue=RevenueRange(min_eur=rev.get("gte"), max_eur=rev.get("lte")),
    )


def _row_satisfies(company, f: StructuredFilters) -> bool:
    if f.industries and company.industry not in f.industries:
        return False
    locs = f.resolved_locations()
    if locs and company.location not in locs:
        return False
    if f.founded_year_gte and (company.founded_year or 0) < f.founded_year_gte:
        return False
    if f.founded_year_lte and (company.founded_year or 9999) > f.founded_year_lte:
        return False
    if f.employee_count_gte and (company.employee_count or 0) < f.employee_count_gte:
        return False
    if f.employee_count_lte and (company.employee_count or 10**9) > f.employee_count_lte:
        return False
    return True


def evaluate_query(qid: str, spec: dict, cfg: RunConfig, db_path: Path | str) -> QueryResult:
    expected = spec.get("expected", {})
    started = time.monotonic()
    q_cfg = cfg.model_copy(update=spec.get("config", {}) or {})
    response, state = run_workflow(spec["query"], q_cfg, db_path=db_path)
    qr = QueryResult(
        qid=qid,
        n_retrieved=sum(response.metadata.candidates_retrieved_per_iteration),
        n_returned=len(response.results),
        llm_calls=response.metadata.llm_calls,
        est_cost_usd=response.metadata.est_cost_usd,
        latency_ms=response.metadata.latency_ms,
        revised=response.metadata.revised_search_performed,
        empty=not response.results,
    )
    c = qr.checks.append
    parsed = response.interpreted_mandate

    # --- parsed criteria: expected mandatory countries/regions/industries present ---
    em = expected.get("mandatory", {}) or {}
    got_loc = set(parsed.mandatory.countries) | set(parsed.mandatory.regions)
    want_loc = set(em.get("countries", [])) | set(em.get("regions", []))
    got_ind = set(parsed.mandatory.industries)
    got_cap = set(parsed.mandatory.capabilities_any) | set(parsed.mandatory.capabilities_all)
    want_cap = set(em.get("capabilities_any", [])) | set(em.get("capabilities_all", []))
    plan_ind = set(response.search_plan.filters.industries)
    plan_loc = set(response.search_plan.filters.resolved_locations())
    c(Check("parsed: mandatory location",
            want_loc.issubset(got_loc) or _loc_equiv(want_loc, got_loc)
            or _loc_equiv(want_loc, plan_loc) or want_loc.issubset(plan_loc),
            f"want {want_loc or '∅'} got mandate={got_loc or '∅'} plan={plan_loc or '∅'}"))
    want_ind = set(em.get("industries", []))
    c(Check("parsed: mandatory industry",
            want_ind.issubset(got_ind) or want_ind.issubset(plan_ind),
            f"want {want_ind or '∅'} got mandate={got_ind or '∅'} plan={plan_ind or '∅'}"))
    if want_cap:
        # substring-tolerant: "fraud detection" ~ "fraud detection technology"
        overlap = any(w in g or g in w for w in want_cap for g in got_cap)
        # also accept a match in the plan's normalised topic terms
        overlap = overlap or bool(want_cap & set(response.search_plan.topic_terms))
        c(Check("parsed: capabilities overlap", overlap, f"want {want_cap} got {got_cap}"))

    # --- mandatory filters actually applied to every returned company ---
    f = _expected_filters(em)
    ids = [r.company_id for r in response.results]
    companies = get_by_ids(ids, db_path=db_path)
    c(Check("all returned companies exist", len(companies) == len(ids),
            f"{len(companies)}/{len(ids)}"))
    bad = [c_.id for c_ in companies if not _row_satisfies(c_, f)]
    c(Check("returned rows satisfy mandatory filters", not bad, f"violations: {bad}"))

    # --- evidence grounding: every evidence quote is a literal substring ---
    ungrounded = 0
    by_id = {c_.id: c_ for c_ in companies}
    for r in response.results:
        comp = by_id.get(r.company_id)
        if not comp:
            continue
        for e in r.evidence:
            value = getattr(comp, e.source_field, "") or ""
            if _norm(e.quote) not in _norm(str(value)):
                ungrounded += 1
    c(Check("evidence quotes grounded", ungrounded == 0, f"ungrounded: {ungrounded}"))

    # --- revision flag ---
    if "should_revise" in expected:
        c(Check("revision performed == expected",
                qr.revised == expected["should_revise"],
                f"want {expected['should_revise']} got {qr.revised}"))

    # --- correct abstention / non-empty ---
    if expected.get("should_be_empty"):
        c(Check("correctly empty (abstention)", qr.empty and bool(response.empty_reason),
                f"empty={qr.empty} reason={response.empty_reason[:60]!r}"))
    elif "should_be_empty" in expected:
        c(Check("returned at least one result", not qr.empty, ""))

    # --- budget adherence ---
    b = state.budget
    c(Check("budget respected",
            b.llm_calls <= b.max_llm_calls and b.iterations <= b.max_iterations
            and b.revisions <= b.max_revisions,
            f"llm={b.llm_calls} iter={b.iterations} rev={b.revisions}"))

    qr.latency_ms = int((time.monotonic() - started) * 1000)
    return qr


def _loc_equiv(want: set[str], got: set[str]) -> bool:
    """Treat a region and its country expansion as equivalent (model may emit either)."""
    def expand(s):
        out = set()
        for term in s:
            res = config.resolve_region(term)
            out |= set(res.countries) if res.known else {term}
        return out
    return expand(want) == expand(got) and bool(want)


def run_eval(
    *, query_ids: list[str] | None = None, cfg: RunConfig | None = None,
    queries_path: Path | str = QUERIES_PATH, db_path: Path | str = DEFAULT_DB,
) -> list[QueryResult]:
    cfg = cfg or RunConfig(provider="openai", use_cache=True)
    specs = yaml.safe_load(Path(queries_path).read_text(encoding="utf-8"))
    ids = query_ids or list(specs)
    return [evaluate_query(qid, specs[qid], cfg, db_path) for qid in ids if qid in specs]


def render_report(results: list[QueryResult]) -> str:
    lines = ["# Evaluation results", ""]
    lines.append(f"_{time.strftime('%Y-%m-%d %H:%M')} · {len(results)} queries · "
                 f"{sum(r.ok for r in results)}/{len(results)} fully green_")
    lines.append("")
    lines.append("| Query | Verdict | Retrieved | Returned | Revised | LLM calls | Cost $ | Latency ms |")
    lines.append("|---|---|--:|--:|:--:|--:|--:|--:|")
    for r in results:
        verdict = "✅ PASS" if r.ok else "❌ " + ", ".join(c.name for c in r.checks if not c.passed)
        lines.append(f"| {r.qid} | {verdict} | {r.n_retrieved} | {r.n_returned} | "
                     f"{'yes' if r.revised else '—'} | {r.llm_calls} | "
                     f"{r.est_cost_usd:.5f} | {r.latency_ms} |")
    lines.append("")
    lines.append("## Per-check detail")
    for r in results:
        lines.append(f"\n### {r.qid}")
        for ch in r.checks:
            mark = "✅" if ch.passed else "❌"
            lines.append(f"- {mark} {ch.name}" + (f" — {ch.detail}" if ch.detail else ""))
    lines.append("")
    lines.append("_Ranking quality, exclusion correctness and evidence *sufficiency* "
                 "are judgment columns — reviewed by hand, not scored here._")
    return "\n".join(lines) + "\n"

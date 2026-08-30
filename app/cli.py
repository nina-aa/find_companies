"""Command-line entry point — the primary test surface for the system.

Subcommands land as their dependencies are built:

* ``ingest``  (M1) — build the retrieval index.
* ``db``      (M1) — run a raw ``SELECT`` against the index for sanity checks.
* ``search``  (M2) — exercise the retrieval tools directly (no LLM), for checking
                     filter / FTS / exclusion behaviour by hand.
* ``check``   (M3) — interpret + plan + feasibility only (1 LLM call).
* ``run``     (M4) — full workflow.
* ``eval``    (M4) — run the eval query set and write RESULTS.md.

Everything except ``ingest`` / ``db`` is a thin wrapper over
``run_workflow(query, cfg)``, added later.
"""

from __future__ import annotations

import argparse
import json
import sys

from app import ingest as ingest_mod


def _cmd_ingest(args: argparse.Namespace) -> int:
    return ingest_mod.main(
        [
            "--input", str(args.input),
            "--output", str(args.output),
            "--manifest", str(args.manifest),
        ]
    )


def _cmd_db(args: argparse.Namespace) -> int:
    from app.db import connect, load_manifest

    sql = " ".join(args.sql).strip()
    if not sql.lower().startswith(("select", "with", "explain", "pragma")):
        print("db: only read-only statements (SELECT / WITH / EXPLAIN / PRAGMA) allowed",
              file=sys.stderr)
        return 2

    conn = connect(args.db)
    try:
        rows = conn.execute(sql).fetchall()
    finally:
        conn.close()

    if args.json:
        print(json.dumps([dict(r) for r in rows], indent=2, default=str))
        return 0

    if not rows:
        print("(0 rows)")
        return 0
    headers = rows[0].keys()
    widths = {h: max(len(h), *(len(str(r[h])) for r in rows)) for h in headers}
    print("  ".join(h.ljust(widths[h]) for h in headers))
    print("  ".join("-" * widths[h] for h in headers))
    for r in rows[:args.limit]:
        print("  ".join(str(r[h]).ljust(widths[h]) for h in headers))
    if len(rows) > args.limit:
        print(f"... ({len(rows)} rows total, showing {args.limit})")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    """interpret_mandate -> build_search_plan -> check_feasibility (1 LLM call)."""
    from app import config
    from app.llm import build_client
    from app.nodes import NodeDeps, build_search_plan, check_feasibility, interpret_mandate
    from app.state import RunConfig, RunState

    config.load_env()
    cfg = RunConfig(provider=args.provider, use_cache=not args.no_cache)
    client = build_client(cfg.provider, model=cfg.model, use_cache=cfg.use_cache)
    deps = NodeDeps(llm=client, db_path=args.db)

    state = RunState(query=args.mandate, cfg=cfg)
    state = interpret_mandate(state, deps)
    state = build_search_plan(state, deps)
    state = check_feasibility(state, deps)

    if args.json:
        print(json.dumps({
            "run_id": state.run_id,
            "interpreted_mandate": state.criteria.model_dump(),
            "search_plan": state.plan.model_dump(),
            "feasibility": state.feasibility.model_dump(),
            "trace": state.trace.model_dump(),
        }, indent=2, default=str))
        return 0

    c, p, f = state.criteria, state.plan, state.feasibility
    print(f"run {state.run_id}  provider={cfg.provider}\n")
    print("INTERPRETED MANDATE")
    print("  mandatory  :", c.mandatory.model_dump(exclude_defaults=True) or "{}")
    print("  preferences:", c.preferences.model_dump(exclude_defaults=True) or "{}")
    print("  exclusions :", c.exclusions.model_dump(exclude_defaults=True) or "{}")
    print("  semantic   :", repr(c.semantic_focus))
    if c.ambiguities:
        for a in c.ambiguities:
            print("  ambiguity  :", a)
    print("\nSEARCH PLAN")
    where, params = p.filters.to_sql()
    print("  SQL WHERE  :", where, "  params:", params)
    print("  topic_terms:", p.topic_terms, f"(mode={p.topic_mode})")
    print("  exclusions :", p.exclusions.model_dump(exclude_defaults=True) or "{}")
    if p.notes:
        for n in p.notes:
            print("  note       :", n)
    print("\nFEASIBILITY")
    print(f"  {f.matched} companies match the mandatory filters -> "
          f"{'feasible' if f.feasible else 'INFEASIBLE'}")
    if not f.feasible:
        print("  reason:", f.reason)
    t = state.trace
    print(f"\ntokens: {t.prompt_tokens}+{t.completion_tokens}  "
          f"est_cost=${t.est_cost_usd:.5f}  llm_calls={t.llm_calls}  repairs={t.repairs}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from app import config
    from app.state import RunConfig
    from app.workflow import run_workflow

    import os
    config.load_env()
    os.environ.setdefault("LOG_LEVEL", "INFO" if args.verbose else "WARNING")

    cfg = RunConfig(
        provider="fake" if args.fake else args.provider,
        use_cache=not args.no_cache,
        min_results=args.min_results,
        deadline_s=args.deadline_s,
        engine=args.engine,
    )
    response, state = run_workflow(args.mandate, cfg, db_path=args.db)

    if args.json:
        print(response.model_dump_json(indent=2))
        return 0
    if args.trace:
        print(state.trace.model_dump_json(indent=2))
        return 0

    _print_run(response, state, explain=args.explain, verbose=args.verbose)
    return 0


def _print_run(response, state, *, explain: bool, verbose: bool) -> None:
    m = response.metadata
    print(f"run {response.run_id}  provider={m.provider}  engine={state.cfg.engine}")
    c = response.interpreted_mandate
    print("\nMANDATE")
    print("  mandatory  :", c.mandatory.model_dump(exclude_defaults=True) or "{}")
    print("  preferences:", c.preferences.model_dump(exclude_defaults=True) or "{}")
    if c.exclusions.model_dump(exclude_defaults=True):
        print("  exclusions :", c.exclusions.model_dump(exclude_defaults=True))
    for a in c.ambiguities:
        print("  ambiguity  :", a)

    p = response.search_plan
    where, params = p.filters.to_sql()
    print("\nPLAN")
    print("  WHERE      :", where, " params:", params)
    print("  topics     :", p.topic_terms, f"(mode={p.topic_mode})")
    for n in p.notes:
        print("  note       :", n)

    if verbose:
        print("\nSTAGES")
        for s in state.trace.stages:
            flag = "ok" if s.ok else "FAIL"
            print(f"  [{flag}] {s.stage}" + (f" — {s.note}" if s.note else ""))
        for t in state.trace.tools:
            print(f"  tool {t.name}: ok={t.ok} count={t.result_count} {t.detail}")

    fn = m.funnel
    print("\nFUNNEL")
    for label, n in (
        ("mandatory filters", fn.mandatory_filters),
        ("+ topic match", fn.topic_match),
        ("- exclusions", fn.after_exclusions),
        ("retrieved pool (cap 100)", fn.retrieved_pool),
        ("sent to validation (cap 10)", fn.sent_to_validation),
        ("passed validation", fn.passed_validation),
        ("returned", fn.returned),
    ):
        print(f"  {label:<30} {n}")

    if response.revision.performed:
        print("\nREVISION")
        print("  reason :", response.revision.reason)
        for r in response.revision.relaxed:
            print("  relaxed:", r)

    print(f"\nRESULTS ({len(response.results)})")
    if response.empty_reason:
        print("  (empty)", response.empty_reason)
    for item in response.results:
        pset = "" if not item.unmet_preferences else f" −{len(item.unmet_preferences)}pref"
        print(f"  #{item.rank} {item.name}  [{item.verdict} rel={item.relevance_score:.2f}"
              f"{pset}]  {item.location}/{item.industry}  founded={item.founded_year} "
              f"emp={item.employee_count}")
        if explain:
            for e in item.evidence:
                print(f"      evidence  ({e.source_field}): \"{e.quote}\"  <- {e.requirement}")
            for inf in item.inferences:
                print(f"      inference : {inf.claim}  ({inf.basis})")
            for u in item.unmet_preferences:
                print(f"      unmet pref: {u}")

    vo = m.validation_outcome
    print(f"\n{m.candidates_validated} validated -> "
          f"{vo.get('matched', 0)} match / {vo.get('partial', 0)} partial / "
          f"{vo.get('rejected', 0)} rejected")
    print(f"llm_calls={m.llm_calls} (attempts={m.llm_attempts}, cache_hits={m.cache_hits}, "
          f"repairs={state.trace.repairs})  tokens={m.prompt_tokens}+{m.completion_tokens}  "
          f"est_cost=${m.est_cost_usd:.5f}  latency={m.latency_ms}ms")
    if m.stop_reason:
        print(f"stop_reason={m.stop_reason}  timed_out={m.timed_out}")


def _cmd_eval(args: argparse.Namespace) -> int:
    import os
    from app import config
    from app.evaluate import RESULTS_PATH, QUERIES_PATH, render_report, run_eval
    from app.state import RunConfig

    config.load_env()
    os.environ.setdefault("LOG_LEVEL", "WARNING")
    cfg = RunConfig(provider="fake" if args.fake else args.provider,
                    use_cache=not args.no_cache)
    from pathlib import Path
    queries_path = Path(args.queries) if args.queries else QUERIES_PATH
    results = run_eval(
        query_ids=args.id or None,
        cfg=cfg,
        queries_path=queries_path,
        db_path=args.db,
    )
    report = render_report(results)
    if args.out:
        out = Path(args.out)
    elif queries_path == QUERIES_PATH:
        out = RESULTS_PATH
    else:                       # non-canonical query set -> don't clobber RESULTS.md
        out = queries_path.parent / f"RESULTS_{queries_path.stem}.md"
    out.write_text(report, encoding="utf-8")

    for r in results:
        verdict = "PASS" if r.ok else "FAIL"
        fails = "" if r.ok else "  <- " + "; ".join(c.name for c in r.checks if not c.passed)
        print(f"  {r.qid:4} {verdict:4}  retrieved={r.n_retrieved:<4} returned={r.n_returned:<3} "
              f"revised={'y' if r.revised else 'n'} llm={r.llm_calls} "
              f"${r.est_cost_usd:.5f} {r.latency_ms}ms{fails}")
    n_ok = sum(r.ok for r in results)
    print(f"\n{n_ok}/{len(results)} fully green — wrote {out}")
    return 0 if n_ok == len(results) else 1


def _cmd_search(args: argparse.Namespace) -> int:
    from app.revenue import RevenueRange
    from app.schemas import Exclusions, StructuredFilters
    from app.tools import search_companies

    filters = StructuredFilters(
        countries=args.country,
        regions=args.region,
        industries=args.industry,
        founded_year_gte=args.founded_gte,
        founded_year_lte=args.founded_lte,
        employee_count_gte=args.emp_gte,
        employee_count_lte=args.emp_lte,
        revenue=RevenueRange(min_eur=args.revenue_gte, max_eur=args.revenue_lte),
    )
    exclusions = Exclusions(industries=args.exclude_industry, keywords=args.exclude_keyword)
    result = search_companies(
        filters,
        topic_terms=args.topic,
        exclusions=exclusions,
        limit=args.limit,
        db_path=args.db,
    )

    if args.json:
        print(result.model_dump_json(indent=2))
        return 0

    print(f"matched_filters={result.matched_filters}  excluded={result.excluded}  "
          f"truncated={result.truncated}")
    print(f"fts_query={result.fts_query!r}")
    print(f"returned {len(result.candidates)} candidate(s):")
    for c in result.candidates:
        score = f"{c.bm25_score:+.2f}" if c.bm25_score is not None else " n/a "
        topics = f"  topics={c.matched_topics}" if c.matched_topics else ""
        print(f"  #{c.rank:>2} [{score}] id={c.id:<6} {c.name[:34]:<34} "
              f"{(c.location or '?')[:11]:<11} {(c.industry or '?')[:10]:<10} "
              f"founded={c.founded_year} emp={c.employee_count}{topics}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="build the SQLite + FTS5 index")
    p_ingest.add_argument("--input", default=ingest_mod.DEFAULT_INPUT)
    p_ingest.add_argument("--output", default=ingest_mod.DEFAULT_DB)
    p_ingest.add_argument("--manifest", default=ingest_mod.DEFAULT_MANIFEST)
    p_ingest.set_defaults(func=_cmd_ingest)

    p_db = sub.add_parser("db", help="run a read-only SQL query against the index")
    p_db.add_argument("sql", nargs="+", help="a SELECT / WITH / EXPLAIN / PRAGMA statement "
                                             "(put --json / --limit before it)")
    p_db.add_argument("--db", default=ingest_mod.DEFAULT_DB, help="path to companies.db")
    p_db.add_argument("--json", action="store_true", help="emit JSON rows")
    p_db.add_argument("--limit", type=int, default=50, help="max rows to print")
    p_db.set_defaults(func=_cmd_db)

    p_s = sub.add_parser("search", help="run the retrieval tools directly (no LLM)")
    p_s.add_argument("--country", action="append", default=[], metavar="NAME")
    p_s.add_argument("--region", action="append", default=[], metavar="NAME",
                     help="e.g. Nordic, Europe, Benelux")
    p_s.add_argument("--industry", action="append", default=[], metavar="LABEL")
    p_s.add_argument("--topic", action="append", default=[], metavar="PHRASE",
                     help="FTS5 topic phrase; repeat for an OR query")
    p_s.add_argument("--founded-gte", type=int, default=None)
    p_s.add_argument("--founded-lte", type=int, default=None)
    p_s.add_argument("--emp-gte", type=int, default=None)
    p_s.add_argument("--emp-lte", type=int, default=None)
    p_s.add_argument("--revenue-gte", type=int, default=None, metavar="EUR")
    p_s.add_argument("--revenue-lte", type=int, default=None, metavar="EUR")
    p_s.add_argument("--exclude-industry", action="append", default=[], metavar="LABEL")
    p_s.add_argument("--exclude-keyword", action="append", default=[], metavar="PHRASE")
    p_s.add_argument("--limit", type=int, default=10)
    p_s.add_argument("--json", action="store_true")
    p_s.add_argument("--db", default=ingest_mod.DEFAULT_DB)
    p_s.set_defaults(func=_cmd_search)

    p_c = sub.add_parser("check", help="interpret + plan + feasibility only (1 LLM call)")
    p_c.add_argument("mandate", help="the natural-language mandate")
    p_c.add_argument("--provider", choices=["fake", "openai"], default="fake")
    p_c.add_argument("--no-cache", action="store_true", help="bypass the response cache")
    p_c.add_argument("--json", action="store_true")
    p_c.add_argument("--db", default=ingest_mod.DEFAULT_DB)
    p_c.set_defaults(func=_cmd_check)

    p_r = sub.add_parser("run", help="run the full workflow on a mandate")
    p_r.add_argument("mandate", help="the natural-language mandate")
    p_r.add_argument("--provider", choices=["fake", "openai"], default="openai")
    p_r.add_argument("--fake", action="store_true", help="shorthand for --provider fake")
    p_r.add_argument("--engine", choices=["driver", "graph"], default="driver")
    p_r.add_argument("--verbose", action="store_true", help="stream every stage + tool call")
    p_r.add_argument("--explain", action="store_true", help="show evidence/inference per result")
    p_r.add_argument("--json", action="store_true", help="raw AgentResponse JSON")
    p_r.add_argument("--trace", action="store_true", help="full RunTrace JSON")
    p_r.add_argument("--no-cache", action="store_true")
    p_r.add_argument("--min-results", type=int, default=3, help="revision trigger threshold")
    p_r.add_argument("--deadline-s", type=float, default=90.0)
    p_r.add_argument("--db", default=ingest_mod.DEFAULT_DB)
    p_r.set_defaults(func=_cmd_run)

    p_e = sub.add_parser("eval", help="run the eval query set -> report table + RESULTS.md")
    p_e.add_argument("--id", action="append", default=[], help="run only these query ids")
    p_e.add_argument("--provider", choices=["fake", "openai"], default="openai")
    p_e.add_argument("--fake", action="store_true")
    p_e.add_argument("--no-cache", action="store_true")
    p_e.add_argument("--queries", default=None, help="path to queries.yaml")
    p_e.add_argument("--out", default=None, help="path to write RESULTS.md")
    p_e.add_argument("--db", default=ingest_mod.DEFAULT_DB)
    p_e.set_defaults(func=_cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

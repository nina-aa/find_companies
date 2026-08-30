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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

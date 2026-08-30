"""Command-line entry point — the primary test surface for the system.

Subcommands land as their dependencies are built:

* ``ingest``  (M1) — build the retrieval index.
* ``db``      (M1) — run a raw ``SELECT`` against the index for sanity checks.
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

    sql = args.sql.strip()
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="build the SQLite + FTS5 index")
    p_ingest.add_argument("--input", default=ingest_mod.DEFAULT_INPUT)
    p_ingest.add_argument("--output", default=ingest_mod.DEFAULT_DB)
    p_ingest.add_argument("--manifest", default=ingest_mod.DEFAULT_MANIFEST)
    p_ingest.set_defaults(func=_cmd_ingest)

    p_db = sub.add_parser("db", help="run a read-only SQL query against the index")
    p_db.add_argument("sql", help="a SELECT / WITH / EXPLAIN / PRAGMA statement")
    p_db.add_argument("--db", default=ingest_mod.DEFAULT_DB, help="path to companies.db")
    p_db.add_argument("--json", action="store_true", help="emit JSON rows")
    p_db.add_argument("--limit", type=int, default=50, help="max rows to print")
    p_db.set_defaults(func=_cmd_db)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

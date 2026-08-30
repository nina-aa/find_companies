"""Retrieval tools — the structured tool surface the workflow calls.

Three tools, each a pure function of ``(inputs, index) -> Pydantic model``, all
deterministic and costing **zero LLM tokens**:

| tool | purpose | budget cost |
|---|---|---|
| ``search_companies`` | mandatory structured gate + FTS5 bm25 topic rank + exclusions | 0 tokens, 1 SQL query |
| ``count_matching``   | feasibility gate + revision reasoning | 0 tokens, 1 SQL count |
| ``get_by_ids``       | hydrate full records for evidence / final response | 0 tokens, 1 SQL query |

Error modes: ``IndexMissingError`` if the DB is absent; every other failure
(bad filter values, empty result) is a normal return, never an exception.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from app import config
from app.db import DEFAULT_DB, connect
from app.schemas import (
    MAX_LIMIT,
    Candidate,
    Company,
    Exclusions,
    SearchResult,
    StructuredFilters,
)

_CANDIDATE_COLUMNS = (
    "id", "name", "description", "industry", "location",
    "founded_year", "employee_count", "revenue_range",
)

# FTS5 syntax characters we strip from user/LLM-supplied topic terms so a term can
# only ever be a phrase, never an injected operator.
_FTS_STRIP = re.compile(r'["*():^]')


def _phrase(term: str) -> str | None:
    cleaned = _FTS_STRIP.sub(" ", term).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return f'"{cleaned}"' if cleaned else None


def fts_match_string(topic_terms) -> str | None:
    """Topic terms -> an FTS5 MATCH string: each term a phrase, ORed together."""
    phrases = [p for p in (_phrase(t) for t in topic_terms) if p]
    return " OR ".join(phrases) if phrases else None


def _like_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _exclusion_sql(exclusions: Exclusions, *, alias: str = "c") -> tuple[str, list]:
    """Extra AND-clauses that drop excluded rows. Empty -> ('', [])."""
    clauses: list[str] = []
    params: list = []

    industries = [config.canonical_industry(i) or i for i in exclusions.industries]
    if industries:
        clauses.append(
            f"{alias}.industry NOT IN ({', '.join('?' for _ in industries)})"
        )
        params.extend(industries)

    for kw in exclusions.keywords:
        kw = kw.strip()
        if not kw:
            continue
        clauses.append(
            f"(LOWER({alias}.name) NOT LIKE ? ESCAPE '\\' "
            f"AND LOWER({alias}.description) NOT LIKE ? ESCAPE '\\')"
        )
        pattern = f"%{_like_escape(kw.lower())}%"
        params.extend([pattern, pattern])

    return (" AND ".join(clauses), params) if clauses else ("", [])


def _with_conn(conn, db_path):
    if conn is not None:
        return conn, False
    return connect(db_path), True


def count_matching(
    filters: StructuredFilters,
    *,
    conn: sqlite3.Connection | None = None,
    db_path: Path | str = DEFAULT_DB,
) -> int:
    """Rows satisfying the mandatory structured filters. Exclusions are *not*
    applied here — feasibility is about whether the mandate itself can be met."""
    conn, owns = _with_conn(conn, db_path)
    try:
        where, params = filters.to_sql(alias="c")
        row = conn.execute(
            f"SELECT COUNT(*) AS n FROM companies c WHERE {where}", params
        ).fetchone()
        return int(row["n"])
    finally:
        if owns:
            conn.close()


def search_companies(
    filters: StructuredFilters,
    topic_terms=(),
    *,
    exclusions: Exclusions | None = None,
    semantic_query: str | None = None,
    limit: int = 50,
    conn: sqlite3.Connection | None = None,
    db_path: Path | str = DEFAULT_DB,
) -> SearchResult:
    """Mandatory structured gate (hard WHERE) → FTS5 bm25 rank on ``topic_terms``
    (ORed phrases) → exclusion gate → truncate to ``limit`` (≤ 100).

    ``semantic_query`` is accepted for interface stability but ignored: no
    embedding index is built (see README D3).
    """
    exclusions = exclusions or Exclusions()
    limit = max(1, min(int(limit), MAX_LIMIT))
    topic_terms = [t for t in topic_terms if t and t.strip()]
    match = fts_match_string(topic_terms)

    where, params = filters.to_sql(alias="c")
    excl_sql, excl_params = _exclusion_sql(exclusions, alias="c")

    conn, owns = _with_conn(conn, db_path)
    try:
        cols = ", ".join(f"c.{c}" for c in _CANDIDATE_COLUMNS)

        if match is not None:
            base_from = (
                "FROM companies_fts "
                "JOIN companies c ON c.id = companies_fts.rowid "
                f"WHERE companies_fts MATCH ? AND ({where})"
            )
            base_params = [match, *params]
            score_expr = "bm25(companies_fts) AS bm25_score"
            order = "ORDER BY bm25_score"
        else:
            base_from = f"FROM companies c WHERE ({where})"
            base_params = list(params)
            score_expr = "NULL AS bm25_score"
            order = "ORDER BY c.id"

        # rows matching the positive predicate, before exclusions
        n_before = conn.execute(
            f"SELECT COUNT(*) AS n {base_from}", base_params
        ).fetchone()["n"]

        full_where = base_from + (f" AND ({excl_sql})" if excl_sql else "")
        full_params = base_params + excl_params

        n_after = conn.execute(
            f"SELECT COUNT(*) AS n {full_where}", full_params
        ).fetchone()["n"]

        rows = conn.execute(
            f"SELECT {cols}, {score_expr} {full_where} {order} LIMIT ?",
            [*full_params, limit],
        ).fetchall()

        matched_filters = count_matching(filters, conn=conn)
    finally:
        if owns:
            conn.close()

    lowered_terms = [t.lower() for t in topic_terms]
    candidates: list[Candidate] = []
    for i, r in enumerate(rows, start=1):
        haystack = f"{r['name']} {r['description']}".lower()
        matched = [
            term for term, low in zip(topic_terms, lowered_terms) if low in haystack
        ]
        candidates.append(
            Candidate(
                id=r["id"], name=r["name"], description=r["description"],
                industry=r["industry"], location=r["location"],
                founded_year=r["founded_year"], employee_count=r["employee_count"],
                revenue_range=r["revenue_range"],
                bm25_score=r["bm25_score"],
                matched_topics=matched,
                rank=i,
            )
        )

    return SearchResult(
        candidates=candidates,
        matched_filters=matched_filters,
        excluded=n_before - n_after,
        fts_query=match,
        truncated=n_after > limit,
    )


def get_by_ids(
    ids,
    *,
    conn: sqlite3.Connection | None = None,
    db_path: Path | str = DEFAULT_DB,
) -> list[Company]:
    """Hydrate full records for the given ids, preserving input order. Unknown
    ids are silently skipped (the caller decides whether that is an error)."""
    ids = [int(i) for i in ids]
    if not ids:
        return []

    conn, owns = _with_conn(conn, db_path)
    try:
        placeholders = ", ".join("?" for _ in ids)
        rows = {
            r["id"]: r
            for r in conn.execute(
                f"SELECT * FROM companies WHERE id IN ({placeholders})", ids
            )
        }
        regions: dict[int, list[str]] = {}
        for r in conn.execute(
            f"SELECT company_id, region FROM company_regions "
            f"WHERE company_id IN ({placeholders}) ORDER BY region",
            ids,
        ):
            regions.setdefault(r["company_id"], []).append(r["region"])
    finally:
        if owns:
            conn.close()

    out: list[Company] = []
    for cid in ids:
        r = rows.get(cid)
        if r is None:
            continue
        out.append(
            Company(
                id=r["id"], name=r["name"], description=r["description"],
                industry=r["industry"], location=r["location"],
                founded_year=r["founded_year"], employee_count=r["employee_count"],
                revenue_range=r["revenue_range"],
                revenue_min_eur=r["revenue_min_eur"],
                revenue_max_eur=r["revenue_max_eur"],
                regions=regions.get(cid, []),
            )
        )
    return out

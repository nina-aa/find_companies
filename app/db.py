"""Thin SQLite access layer over the built index.

M1 scope: a connection helper, a manifest check, and ``count_matching`` for
feasibility sanity checks. The full retrieval tools (``search_companies``,
``get_by_ids``) arrive in M2 (C2) on top of this.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from app import config
from app.revenue import RevenueRange, buckets_matching

DEFAULT_DB = config.REPO_ROOT / "data" / "index" / "companies.db"
DEFAULT_MANIFEST = config.REPO_ROOT / "data" / "index" / "manifest.json"


class IndexMissingError(RuntimeError):
    pass


def connect(db_path: Path | str = DEFAULT_DB, *, read_only: bool = True) -> sqlite3.Connection:
    path = Path(db_path)
    if not path.exists():
        raise IndexMissingError(
            f"no index at {path} — run `python -m app.cli ingest` first"
        )
    if read_only:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def load_manifest(manifest_path: Path | str = DEFAULT_MANIFEST) -> dict:
    path = Path(manifest_path)
    if not path.exists():
        raise IndexMissingError(f"no manifest at {path} — run `python -m app.cli ingest`")
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class StructuredFilters:
    """Mandatory structured constraints. Every field is an AND; list fields are
    set membership (IN). ``None`` / empty means unconstrained."""

    countries: list[str] = field(default_factory=list)
    industries: list[str] = field(default_factory=list)
    regions: list[str] = field(default_factory=list)
    founded_year_gte: int | None = None
    founded_year_lte: int | None = None
    employee_count_gte: int | None = None
    employee_count_lte: int | None = None
    revenue: RevenueRange = field(default_factory=RevenueRange)

    def to_sql(self, *, alias: str = "c") -> tuple[str, list]:
        """Render to a ``WHERE`` fragment (without the ``WHERE`` keyword) + params."""
        clauses: list[str] = []
        params: list = []

        def _in(column: str, values: list[str]) -> None:
            if values:
                clauses.append(f"{alias}.{column} IN ({', '.join('?' for _ in values)})")
                params.extend(values)

        _in("industry", self.industries)

        # countries + regions both constrain location; combine as a single IN set.
        location_set: set[str] = set(self.countries)
        for region in self.regions:
            location_set.update(config.resolve_region(region).countries)
        if location_set:
            ordered = sorted(location_set)
            clauses.append(f"{alias}.location IN ({', '.join('?' for _ in ordered)})")
            params.extend(ordered)

        for column, op, value in (
            ("founded_year", ">=", self.founded_year_gte),
            ("founded_year", "<=", self.founded_year_lte),
            ("employee_count", ">=", self.employee_count_gte),
            ("employee_count", "<=", self.employee_count_lte),
        ):
            if value is not None:
                clauses.append(f"{alias}.{column} {op} ?")
                params.append(value)

        if not self.revenue.is_empty():
            allowed = buckets_matching(self.revenue)
            if allowed:
                clauses.append(
                    f"{alias}.revenue_range IN ({', '.join('?' for _ in allowed)})"
                )
                params.extend(allowed)
            else:
                clauses.append("0")  # constraint excludes every bucket

        where = " AND ".join(clauses) if clauses else "1"
        return where, params


def count_matching(
    filters: StructuredFilters,
    *,
    conn: sqlite3.Connection | None = None,
    db_path: Path | str = DEFAULT_DB,
) -> int:
    """Number of companies satisfying the mandatory structured filters."""
    owns = conn is None
    conn = conn or connect(db_path)
    try:
        where, params = filters.to_sql(alias="c")
        row = conn.execute(
            f"SELECT COUNT(*) AS n FROM companies c WHERE {where}", params
        ).fetchone()
        return int(row["n"])
    finally:
        if owns:
            conn.close()

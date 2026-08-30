"""Pydantic contracts for the retrieval tools (C2).

Every tool has an explicit input and output model. These are *storage / transport*
models — the narrow schemas the LLM is asked to produce live separately (C3), so
that an internal model is never reused as an LLM response schema.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app import config
from app.revenue import RevenueRange, buckets_matching

# Hard ceilings — enforced here as well as by BudgetGuard (C5).
MAX_LIMIT = 100


class Exclusions(BaseModel):
    """Negative constraints. ``industries`` is a structured gate; ``keywords`` is a
    deterministic substring gate over name + description. Semantic exclusion (a
    category the text only implies) is left to the validator (C4)."""

    model_config = ConfigDict(frozen=True)

    industries: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.industries and not self.keywords


class StructuredFilters(BaseModel):
    """Mandatory structured constraints only. Every field is ANDed; list fields are
    set membership (IN). Preferences never appear here — they are ranking boosts."""

    model_config = ConfigDict(frozen=True)

    countries: list[str] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)
    industries: list[str] = Field(default_factory=list)
    founded_year_gte: int | None = None
    founded_year_lte: int | None = None
    employee_count_gte: int | None = None
    employee_count_lte: int | None = None
    revenue: RevenueRange = Field(default_factory=RevenueRange)

    def resolved_locations(self) -> list[str]:
        """countries + regions collapsed to a concrete, sorted country set."""
        locations: set[str] = set(self.countries)
        for region in self.regions:
            locations.update(config.resolve_region(region).countries)
        return sorted(locations)

    def to_sql(self, *, alias: str = "c") -> tuple[str, list]:
        """Render to a ``WHERE`` fragment (no ``WHERE`` keyword) + bind params."""
        clauses: list[str] = []
        params: list = []

        if self.industries:
            clauses.append(
                f"{alias}.industry IN ({', '.join('?' for _ in self.industries)})"
            )
            params.extend(self.industries)

        locations = self.resolved_locations()
        if locations:
            clauses.append(
                f"{alias}.location IN ({', '.join('?' for _ in locations)})"
            )
            params.extend(locations)

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
                clauses.append("0")

        return (" AND ".join(clauses) if clauses else "1"), params


class Candidate(BaseModel):
    """One row from ``search_companies`` — enough to rank and to hand to the
    validator, not the full record (that comes from ``get_by_ids``)."""

    id: int
    name: str
    description: str
    industry: str | None
    location: str | None
    founded_year: int | None
    employee_count: int | None
    revenue_range: str | None
    bm25_score: float | None = None          # None when no topic query was run
    matched_topics: list[str] = Field(default_factory=list)
    rank: int = 0                             # 1-based position in this result set


class SearchResult(BaseModel):
    """Return type of ``search_companies``.

    Deviation from the PLAN sketch (``-> list[Candidate]``): the node and the
    response metadata need the exclusion bookkeeping (Q3 must report "exclusion
    matched no candidates"; S3 must report how many it removed), so the tool
    returns the list *plus* those counters rather than losing them.
    """

    candidates: list[Candidate] = Field(default_factory=list)
    matched_filters: int = 0            # rows passing the mandatory structured gate
    excluded: int = 0                   # of those, how many the exclusions removed
    fts_query: str | None = None        # the FTS5 MATCH string actually used
    truncated: bool = False             # more than `limit` matched


class Company(BaseModel):
    """Full hydrated record from ``get_by_ids`` — the source of truth for
    span-grounding and for the fields returned in the final response."""

    id: int
    name: str
    description: str
    industry: str | None
    location: str | None
    founded_year: int | None
    employee_count: int | None
    revenue_range: str | None
    revenue_min_eur: int | None
    revenue_max_eur: int | None
    regions: list[str] = Field(default_factory=list)

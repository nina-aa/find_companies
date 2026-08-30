"""Retrieval tool contracts — filter correctness, FTS ranking, exclusions.
All against the shared real-shaped fixture index. Zero LLM involvement."""

import pytest

from app.schemas import MAX_LIMIT, Candidate, Company, Exclusions, StructuredFilters
from app.tools import (
    count_matching,
    fts_match_string,
    get_by_ids,
    search_companies,
)


# --------------------------------------------------------------------------- #
# structured gate
# --------------------------------------------------------------------------- #
def test_structured_gate_is_a_hard_filter(fixture_db):
    res = search_companies(
        StructuredFilters(countries=["Finland"], industries=["Fintech"]),
        db_path=fixture_db,
    )
    assert {c.id for c in res.candidates} == {1, 2, 3, 4}
    assert all(c.industry == "Fintech" and c.location == "Finland" for c in res.candidates)
    assert res.matched_filters == 4
    assert res.fts_query is None


def test_no_topic_terms_orders_by_stable_key(fixture_db):
    res = search_companies(
        StructuredFilters(industries=["Energy"]), db_path=fixture_db
    )
    ids = [c.id for c in res.candidates]
    assert ids == sorted(ids)
    assert all(c.bm25_score is None for c in res.candidates)


def test_region_filter_expands_deterministically(fixture_db):
    res = search_companies(
        StructuredFilters(regions=["Nordic"], industries=["Energy"]),
        db_path=fixture_db,
    )
    assert {c.id for c in res.candidates} == {5, 6}  # Norway + Sweden, not Germany


def test_mandatory_numeric_bounds(fixture_db):
    res = search_companies(
        StructuredFilters(industries=["Fintech"], employee_count_lte=200),
        db_path=fixture_db,
    )
    assert {c.id for c in res.candidates} == {1, 3}  # 120 and 45 employees


# --------------------------------------------------------------------------- #
# FTS5 topic ranking
# --------------------------------------------------------------------------- #
def test_topic_terms_rank_by_bm25_and_record_matches(fixture_db):
    res = search_companies(
        StructuredFilters(countries=["Finland"], industries=["Fintech"]),
        topic_terms=["fraud detection"],
        db_path=fixture_db,
    )
    ids = [c.id for c in res.candidates]
    assert set(ids) == {1, 3}                      # only the two fraud-detection rows
    assert all("fraud detection" in c.matched_topics for c in res.candidates)
    assert all(c.bm25_score is not None for c in res.candidates)
    assert res.fts_query == '"fraud detection"'


def test_topic_terms_are_ORed(fixture_db):
    res = search_companies(
        StructuredFilters(countries=["Finland"], industries=["Fintech"]),
        topic_terms=["fraud detection", "banking analytics"],
        db_path=fixture_db,
    )
    # row 1 has both, row 3 has fraud detection only -> both returned
    assert {c.id for c in res.candidates} == {1, 3}
    row1 = next(c for c in res.candidates if c.id == 1)
    assert set(row1.matched_topics) == {"fraud detection", "banking analytics"}
    assert res.fts_query == '"fraud detection" OR "banking analytics"'


def test_topic_terms_do_not_widen_beyond_structured_gate(fixture_db):
    res = search_companies(
        StructuredFilters(countries=["Finland"], industries=["Fintech"]),
        topic_terms=["energy forecasting"],
        db_path=fixture_db,
    )
    assert res.candidates == []


# --------------------------------------------------------------------------- #
# exclusions
# --------------------------------------------------------------------------- #
def test_keyword_exclusion_removes_and_counts(fixture_db):
    res = search_companies(
        StructuredFilters(countries=["Germany"], industries=["Energy"]),
        exclusions=Exclusions(keywords=["smart grid"]),
        db_path=fixture_db,
    )
    # Germany Energy = {8 Berlin GridSense, 9 Hamburg Volt}; 8 mentions smart grid
    assert {c.id for c in res.candidates} == {9}
    assert res.excluded == 1


def test_industry_exclusion(fixture_db):
    res = search_companies(
        StructuredFilters(countries=["Germany"]),
        exclusions=Exclusions(industries=["Energy"]),
        db_path=fixture_db,
    )
    assert all(c.industry != "Energy" for c in res.candidates)
    assert {c.id for c in res.candidates} == {7, 10}


def test_vacuous_exclusion_reports_zero_removed(fixture_db):
    """Q3 shape: the exclusion category matches nothing -> excluded == 0."""
    res = search_companies(
        StructuredFilters(countries=["Finland"], industries=["Fintech"]),
        topic_terms=["fraud detection"],
        exclusions=Exclusions(keywords=["blockchain gaming"]),
        db_path=fixture_db,
    )
    assert {c.id for c in res.candidates} == {1, 3}
    assert res.excluded == 0


def test_count_matching_ignores_exclusions(fixture_db):
    f = StructuredFilters(countries=["Germany"], industries=["Energy"])
    assert count_matching(f, db_path=fixture_db) == 2  # feasibility is mandate-only


# --------------------------------------------------------------------------- #
# limits / bookkeeping
# --------------------------------------------------------------------------- #
def test_limit_is_clamped_and_truncation_flagged(fixture_db):
    res = search_companies(
        StructuredFilters(), limit=2, db_path=fixture_db
    )
    assert len(res.candidates) == 2
    assert res.truncated is True
    assert [c.rank for c in res.candidates] == [1, 2]


def test_limit_upper_bound(fixture_db):
    res = search_companies(StructuredFilters(), limit=10_000, db_path=fixture_db)
    assert len(res.candidates) == 12          # limit clamped to MAX_LIMIT, all rows fit
    assert res.truncated is False
    assert MAX_LIMIT == 100


def test_semantic_query_is_ignored(fixture_db):
    a = search_companies(StructuredFilters(industries=["Biotech"]), db_path=fixture_db)
    b = search_companies(
        StructuredFilters(industries=["Biotech"]),
        semantic_query="cutting edge genomics",
        db_path=fixture_db,
    )
    assert [c.id for c in a.candidates] == [c.id for c in b.candidates]


# --------------------------------------------------------------------------- #
# get_by_ids
# --------------------------------------------------------------------------- #
def test_get_by_ids_preserves_order_and_hydrates_regions(fixture_db):
    companies = get_by_ids([5, 1, 999], db_path=fixture_db)
    assert [c.id for c in companies] == [5, 1]      # unknown id skipped, order kept
    bergen = companies[0]
    assert isinstance(bergen, Company)
    assert bergen.revenue_min_eur == 1_000_000
    assert "nordic" in bergen.regions and "europe" in bergen.regions


def test_get_by_ids_empty(fixture_db):
    assert get_by_ids([], db_path=fixture_db) == []


# --------------------------------------------------------------------------- #
# FTS match-string builder (pure)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "terms, expected",
    [
        (["fraud detection"], '"fraud detection"'),
        (["a", "b"], '"a" OR "b"'),
        ([], None),
        (["  "], None),
        (['drug "discovery"'], '"drug discovery"'),        # injected quote stripped
        (["fraud OR (evil)"], '"fraud OR evil"'),           # operators neutralised
    ],
)
def test_fts_match_string(terms, expected):
    assert fts_match_string(terms) == expected

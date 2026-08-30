"""Ingestion + StructuredFilters SQL, against the shared fixture index."""

from app.db import connect
from app.revenue import RevenueRange
from app.schemas import StructuredFilters
from app.tools import count_matching


def test_typed_columns_and_derived_revenue(fixture_db):
    with connect(fixture_db) as conn:
        row = conn.execute("SELECT * FROM companies WHERE id = 5").fetchone()
    assert row["industry"] == "Energy"
    assert row["revenue_min_eur"] == 1_000_000
    assert row["revenue_max_eur"] == 10_000_000


def test_region_lookup_table_populated(fixture_db):
    with connect(fixture_db) as conn:
        nordic = {r["company_id"] for r in conn.execute(
            "SELECT company_id FROM company_regions WHERE region = 'nordic'"
        )}
        europe = {r["company_id"] for r in conn.execute(
            "SELECT company_id FROM company_regions WHERE region = 'europe'"
        )}
    assert nordic == {1, 2, 3, 4, 5, 6}      # Finland x4 + Norway + Sweden
    assert 11 not in europe                  # USA


def test_fts5_bm25_ranks_topic_match(fixture_db):
    with connect(fixture_db) as conn:
        ids = [r["rowid"] for r in conn.execute(
            "SELECT rowid FROM companies_fts WHERE companies_fts MATCH ? "
            "ORDER BY bm25(companies_fts)",
            ('"drug discovery"',),
        )]
    assert ids == [7]


def test_structured_filters_country_and_industry(fixture_db):
    f = StructuredFilters(countries=["Finland"], industries=["Fintech"])
    assert count_matching(f, db_path=fixture_db) == 4


def test_structured_filters_region_expands_to_countries(fixture_db):
    f = StructuredFilters(regions=["Nordic"], industries=["Energy"])
    assert count_matching(f, db_path=fixture_db) == 2  # Bergen Grid + Stockholm Watts


def test_mandatory_year_bound_filters(fixture_db):
    f = StructuredFilters(industries=["Fintech"], founded_year_gte=2020)
    assert count_matching(f, db_path=fixture_db) == 2  # Espoo (2021) + Tampere (2023)


def test_revenue_constraint_below_10m(fixture_db):
    f = StructuredFilters(revenue=RevenueRange(max_eur=10_000_000))
    with connect(fixture_db) as conn:
        where, params = f.to_sql(alias="c")
        ids = {r["id"] for r in conn.execute(
            f"SELECT id FROM companies c WHERE {where}", params
        )}
    assert ids == {3, 4, 5}  # 1M-10M / 0-1M buckets only


def test_employee_gt_5000_is_infeasible(fixture_db):
    f = StructuredFilters(countries=["Finland"], industries=["Fintech"],
                          employee_count_gte=5001)
    assert count_matching(f, db_path=fixture_db) == 0

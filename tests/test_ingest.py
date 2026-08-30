"""Ingestion + StructuredFilters SQL, against a small hand-built fixture index."""

import json

import pytest

from app import ingest
from app.db import StructuredFilters, connect, count_matching
from app.revenue import RevenueRange

FIXTURE_ROWS = [
    # id, name, description, industry, location, founded, employees, revenue
    (1, "Helsinki Fraud Labs", "AI-powered platform for fraud detection and banking analytics.",
     "Fintech", "Finland", 2019, 120, "10M-50M"),
    (2, "Oulu Ledger", "cloud-native engine for payments and lending.",
     "Fintech", "Finland", 2012, 600, "100M-500M"),
    (3, "Bergen Grid", "data-driven software for energy forecasting and smart grid.",
     "Energy", "Norway", 2020, 80, "1M-10M"),
    (4, "Munich BioWorks", "AI-powered platform for drug discovery and molecular analysis.",
     "Biotech", "Germany", 2021, 300, "50M-100M"),
    (5, "Berlin GridSense", "data-driven engine for smart grid optimization.",
     "Energy", "Germany", 2017, 210, "500M+"),
    (6, "Texas Telco", "cloud-native software for 5G analytics and network optimization.",
     "Telecom", "USA", 2005, 5000, "500M+"),
]


@pytest.fixture(scope="module")
def fixture_db(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("index")
    src = tmp / "companies.json"
    keys = ["id", "name", "description", "industry", "location",
            "founded_year", "employee_count", "revenue_range"]
    src.write_text(json.dumps([dict(zip(keys, r)) for r in FIXTURE_ROWS]), encoding="utf-8")
    db = tmp / "companies.db"
    manifest = ingest.build(src, db, tmp / "manifest.json")
    assert manifest["row_count"] == len(FIXTURE_ROWS)
    return db


def test_typed_columns_and_derived_revenue(fixture_db):
    with connect(fixture_db) as conn:
        row = conn.execute(
            "SELECT * FROM companies WHERE id = 3"
        ).fetchone()
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
    assert nordic == {1, 2, 3}          # Finland x2 + Norway
    assert 6 not in europe              # USA


def test_fts5_bm25_ranks_topic_match(fixture_db):
    with connect(fixture_db) as conn:
        ids = [r["rowid"] for r in conn.execute(
            "SELECT rowid FROM companies_fts WHERE companies_fts MATCH ? "
            "ORDER BY bm25(companies_fts)",
            ('"drug discovery"',),
        )]
    assert ids == [4]


def test_structured_filters_country_and_industry(fixture_db):
    f = StructuredFilters(countries=["Finland"], industries=["Fintech"])
    assert count_matching(f, db_path=fixture_db) == 2


def test_structured_filters_region_expands_to_countries(fixture_db):
    f = StructuredFilters(regions=["Nordic"], industries=["Energy"])
    assert count_matching(f, db_path=fixture_db) == 1  # Bergen Grid only


def test_preferences_are_not_in_structured_filters(fixture_db):
    # founded_year / employee_count as *mandatory* bounds still work when asked for
    f = StructuredFilters(industries=["Fintech"], founded_year_gte=2015)
    assert count_matching(f, db_path=fixture_db) == 1  # Helsinki Fraud Labs (2019)


def test_revenue_constraint_below_10m(fixture_db):
    f = StructuredFilters(revenue=RevenueRange(max_eur=10_000_000))
    with connect(fixture_db) as conn:
        where, params = f.to_sql(alias="c")
        ids = {r["id"] for r in conn.execute(
            f"SELECT id FROM companies c WHERE {where}", params
        )}
    assert ids == {3}  # only Bergen Grid is in 0-1M / 1M-10M


def test_employee_gt_5000_is_infeasible(fixture_db):
    f = StructuredFilters(countries=["Finland"], industries=["Fintech"],
                          employee_count_gte=5001)
    assert count_matching(f, db_path=fixture_db) == 0

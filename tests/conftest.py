"""Shared fixture: a small SQLite index built the real way (via ``app.ingest``)
from a hand-written slice that mirrors companies.json's shape and quirks."""

import json

import pytest

from app import ingest

_KEYS = ["id", "name", "description", "industry", "location",
         "founded_year", "employee_count", "revenue_range"]

FIXTURE_ROWS = [
    (1, "Helsinki Fraud Labs", "AI-powered platform for fraud detection and banking analytics.",
     "Fintech", "Finland", 2019, 120, "10M-50M"),
    (2, "Oulu Ledger", "cloud-native engine for payments and lending.",
     "Fintech", "Finland", 2012, 600, "100M-500M"),
    (3, "Espoo RiskScore", "data-driven software for fraud detection and risk assessment.",
     "Fintech", "Finland", 2021, 45, "1M-10M"),
    (4, "Tampere PayFlow", "cloud-native platform for payments.",
     "Fintech", "Finland", 2023, 5000, "0-1M"),
    (5, "Bergen Grid", "data-driven software for energy forecasting and smart grid.",
     "Energy", "Norway", 2020, 80, "1M-10M"),
    (6, "Stockholm Watts", "AI-powered engine for energy forecasting.",
     "Energy", "Sweden", 2018, 150, "50M-100M"),
    (7, "Munich BioWorks", "AI-powered platform for drug discovery and molecular analysis.",
     "Biotech", "Germany", 2021, 300, "50M-100M"),
    (8, "Berlin GridSense", "data-driven engine for smart grid optimization.",
     "Energy", "Germany", 2017, 210, "500M+"),
    (9, "Hamburg Volt", "cloud-native software for renewable energy.",
     "Energy", "Germany", 2022, 60, "10M-50M"),
    (10, "Cologne CyberAudit", "data-driven platform for cybersecurity compliance for banks.",
     "Technology", "Germany", 2016, 500, "100M-500M"),
    (11, "Texas Telco", "cloud-native software for 5G analytics and network optimization.",
     "Telecom", "USA", 2005, 4800, "500M+"),
    (12, "Lyon Diagnostics", "machine learning software for diagnostics and patient monitoring.",
     "Healthcare", "France", 2019, 240, "10M-50M"),
]


@pytest.fixture(scope="session")
def fixture_db(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("index")
    src = tmp / "companies.json"
    src.write_text(
        json.dumps([dict(zip(_KEYS, r)) for r in FIXTURE_ROWS]), encoding="utf-8"
    )
    db = tmp / "companies.db"
    manifest = ingest.build(src, db, tmp / "manifest.json")
    assert manifest["row_count"] == len(FIXTURE_ROWS)
    return db

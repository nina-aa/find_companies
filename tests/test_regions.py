"""Region / country resolution + industry-enum validation."""

import pytest

from app import config


@pytest.mark.parametrize(
    "term, expected",
    [
        ("Nordic", {"Finland", "Norway", "Sweden"}),
        ("nordics", {"Finland", "Norway", "Sweden"}),
        ("Scandinavia", {"Norway", "Sweden"}),
        ("Europe", {"Finland", "Germany", "France", "Norway", "Sweden", "Netherlands", "UK"}),
        ("European", {"Finland", "Germany", "France", "Norway", "Sweden", "Netherlands", "UK"}),
        ("DACH", {"Germany"}),
        ("Benelux", {"Netherlands"}),
        ("North America", {"USA"}),
    ],
)
def test_resolve_region_multi_country(term, expected):
    res = config.resolve_region(term)
    assert res.known is True
    assert set(res.countries) == expected
    assert res.ambiguous is False


@pytest.mark.parametrize(
    "term, country",
    [
        ("Finland", "Finland"),
        ("finnish", "Finland"),
        ("German", "Germany"),
        ("the Netherlands", "Netherlands"),
        ("US", "USA"),
        ("United Kingdom", "UK"),
        ("British", "UK"),
    ],
)
def test_resolve_country_and_aliases(term, country):
    res = config.resolve_region(term)
    assert res.known is True
    assert res.countries == (country,)


def test_baltics_is_known_but_empty():
    res = config.resolve_region("Baltics")
    assert res.known is True
    assert res.empty_region is True
    assert res.countries == ()


@pytest.mark.parametrize("term", ["Atlantis", "APAC", "Middle East", "Denmark", "Switzerland"])
def test_unknown_region_is_ambiguous_not_guessed(term):
    res = config.resolve_region(term)
    assert res.known is False
    assert res.ambiguous is True
    assert res.countries == ()


@pytest.mark.parametrize(
    "value, expected",
    [
        ("fintech", "Fintech"),
        ("FINTECH", "Fintech"),
        ("  Biotech ", "Biotech"),
        ("Healthcare", "Healthcare"),
    ],
)
def test_canonical_industry(value, expected):
    assert config.canonical_industry(value) == expected


@pytest.mark.parametrize("value", ["Cybersecurity", "Manufacturing", "AI", ""])
def test_canonical_industry_rejects_unknown(value):
    assert config.canonical_industry(value) is None


def test_industry_enum_is_exactly_ten():
    assert len(config.INDUSTRIES) == 10

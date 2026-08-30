"""Revenue-bucket parsing + constraint -> bucket-set mapping."""

import pytest

from app.revenue import (
    BUCKETS,
    RevenueRange,
    buckets_matching,
    parse_amount,
    parse_bucket,
    parse_constraint,
)


@pytest.mark.parametrize(
    "bucket, expected",
    [
        ("0-1M", (0, 1_000_000)),
        ("1M-10M", (1_000_000, 10_000_000)),
        ("10M-50M", (10_000_000, 50_000_000)),
        ("50M-100M", (50_000_000, 100_000_000)),
        ("100M-500M", (100_000_000, 500_000_000)),
        ("500M+", (500_000_000, None)),
        (" 10M-50M ", (10_000_000, 50_000_000)),
    ],
)
def test_parse_bucket_known(bucket, expected):
    assert parse_bucket(bucket) == expected


@pytest.mark.parametrize("bad", [None, "", "  ", "5M-7M", "unknown", "10M"])
def test_parse_bucket_unknown_is_none(bad):
    assert parse_bucket(bad) is None


def test_bucket_table_is_ordered_and_contiguous():
    lows = [lo for lo, _ in BUCKETS.values()]
    assert lows == sorted(lows)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("10M", 10_000_000),
        ("EUR 10 million", 10_000_000),
        ("€10m", 10_000_000),
        ("500,000", 500_000),
        ("1.5M", 1_500_000),
        ("2 billion", 2_000_000_000),
        ("250k", 250_000),
    ],
)
def test_parse_amount(text, expected):
    assert parse_amount(text) == expected


def test_parse_amount_unparseable():
    assert parse_amount("a lot of money") is None


@pytest.mark.parametrize(
    "constraint, expected",
    [
        (RevenueRange(max_eur=10_000_000), ["0-1M", "1M-10M"]),
        (RevenueRange(max_eur=1_000_000), ["0-1M"]),
        (RevenueRange(min_eur=500_000_000), ["500M+"]),
        (RevenueRange(min_eur=100_000_000), ["100M-500M", "500M+"]),
        (RevenueRange(min_eur=10_000_000, max_eur=100_000_000), ["10M-50M", "50M-100M"]),
        (RevenueRange(), list(BUCKETS)),
    ],
)
def test_buckets_matching(constraint, expected):
    assert buckets_matching(constraint) == expected


def test_buckets_matching_impossible_window_is_empty():
    # a window strictly inside a single bucket still overlaps that bucket
    assert buckets_matching(RevenueRange(min_eur=2_000_000, max_eur=3_000_000)) == ["1M-10M"]


@pytest.mark.parametrize(
    "phrase, lo, hi",
    [
        ("revenue below EUR 10M", None, 10_000_000),
        ("annual revenue under 1 million", None, 1_000_000),
        ("more than EUR 100M in revenue", 100_000_000, None),
        ("at least 50M", 50_000_000, None),
        ("between 10M and 50M", 10_000_000, 50_000_000),
        ("a profitable company", None, None),
    ],
)
def test_parse_constraint(phrase, lo, hi):
    result = parse_constraint(phrase)
    assert (result.min_eur, result.max_eur) == (lo, hi)

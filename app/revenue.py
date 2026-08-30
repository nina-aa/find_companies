"""Revenue-bucket parsing.

``companies.json`` stores revenue as one of six bucket strings. Two jobs here:

1. ``parse_bucket`` — turn a stored bucket string into a concrete
   ``(min_eur, max_eur)`` pair (used at ingestion time to fill the
   ``revenue_min_eur`` / ``revenue_max_eur`` columns).
2. ``buckets_matching`` — turn a mandate constraint ("revenue below EUR 10M")
   into the set of bucket strings that satisfy it, so retrieval can filter on the
   stored ``revenue_range`` column directly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Canonical order, smallest first. ``None`` upper bound = open-ended.
BUCKETS: dict[str, tuple[int, int | None]] = {
    "0-1M": (0, 1_000_000),
    "1M-10M": (1_000_000, 10_000_000),
    "10M-50M": (10_000_000, 50_000_000),
    "50M-100M": (50_000_000, 100_000_000),
    "100M-500M": (100_000_000, 500_000_000),
    "500M+": (500_000_000, None),
}

_UNIT = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000, "bn": 1_000_000_000}
_AMOUNT_RE = re.compile(r"(\d[\d.,]*)\s*(k|m|bn|b)?", re.IGNORECASE)
_THOUSANDS_RE = re.compile(r"^\d{1,3}(,\d{3})+$")


@dataclass(frozen=True)
class RevenueRange:
    """A mandate's revenue constraint in euros. ``None`` = unbounded on that side."""

    min_eur: int | None = None
    max_eur: int | None = None

    def is_empty(self) -> bool:
        return self.min_eur is None and self.max_eur is None


def parse_bucket(bucket: str | None) -> tuple[int, int | None] | None:
    """Stored bucket string -> (min_eur, max_eur). Unknown/empty -> ``None``."""
    if not bucket:
        return None
    return BUCKETS.get(bucket.strip())


def parse_amount(text: str) -> int | None:
    """'10M' / 'EUR 10 million' / '500,000' -> integer euros. ``None`` if unparseable."""
    lowered = text.lower().replace("euro", "").replace("eur", "").replace("€", "")
    lowered = lowered.replace("million", "m").replace("billion", "bn").replace("thousand", "k")
    match = _AMOUNT_RE.search(lowered)
    if not match:
        return None
    raw = match.group(1).rstrip(".,")
    if _THOUSANDS_RE.match(raw):
        number = float(raw.replace(",", ""))          # 500,000 -> 500000
    else:
        number = float(raw.replace(",", "."))          # 1,5 -> 1.5 (European decimal)
    unit = (match.group(2) or "").lower()
    return int(round(number * _UNIT.get(unit, 1)))


def buckets_matching(constraint: RevenueRange) -> list[str]:
    """Bucket strings whose euro interval overlaps the constraint interval.

    A bucket is kept when it could contain a company satisfying the constraint,
    i.e. the two ``[min, max]`` intervals intersect. "below EUR 10M" therefore
    keeps ``0-1M`` and ``1M-10M`` (and not ``10M-50M``, whose min is 10M — no
    overlap with the open interval below 10M except the shared endpoint).
    """
    if constraint.is_empty():
        return list(BUCKETS)

    lo = constraint.min_eur if constraint.min_eur is not None else 0
    hi = constraint.max_eur  # may be None -> +inf

    out: list[str] = []
    for name, (b_lo, b_hi) in BUCKETS.items():
        # bucket interval [b_lo, b_hi); constraint interval [lo, hi]
        if b_hi is not None and b_hi <= lo:
            continue  # bucket entirely below the constraint floor
        if hi is not None and b_lo >= hi:
            continue  # bucket entirely above the constraint ceiling
        out.append(name)
    return out


# Phrasing -> which side of the range the amount bounds.
_BELOW = ("below", "under", "less than", "at most", "up to", "no more than", "<", "<=")
_ABOVE = ("above", "over", "more than", "at least", "greater than", "minimum", ">", ">=")


def parse_constraint(text: str) -> RevenueRange:
    """Free-text revenue phrase -> ``RevenueRange``. Best-effort; used as a fallback
    when the LLM does not already emit a structured ``revenue_eur`` bound."""
    lowered = text.lower()
    amount = parse_amount(lowered)
    if amount is None:
        return RevenueRange()

    if any(k in lowered for k in _BELOW):
        return RevenueRange(max_eur=amount)
    if any(k in lowered for k in _ABOVE):
        return RevenueRange(min_eur=amount)

    # "between X and Y"
    between = re.search(r"between\s+(.+?)\s+and\s+(.+)", lowered)
    if between:
        lo = parse_amount(between.group(1))
        hi = parse_amount(between.group(2))
        return RevenueRange(min_eur=lo, max_eur=hi)

    return RevenueRange()

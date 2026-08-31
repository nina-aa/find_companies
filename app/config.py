"""Loaders for the static configuration that codifies the dataset's taxonomy.

Three YAML files, loaded once and cached:

* ``schema_map.yaml`` — raw field -> canonical field mapping (ingestion).
* ``regions.yaml``    — region term -> country list, plus country aliases.
* ``lexicon.yaml``    — domain phrase -> {industry, topics} safety net.

Everything here is deterministic. The LLM never sees these tables; it produces
structured criteria and this module validates / resolves them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent

# The 10 industry labels that actually occur in companies.json. build_search_plan
# validates every LLM-proposed industry against this set.
INDUSTRIES: frozenset[str] = frozenset({
    "Fintech", "Energy", "Biotech", "Healthcare", "Telecom",
    "Technology", "Logistics", "Education", "Retail", "Automotive",
})

_INDUSTRY_BY_LOWER = {name.lower(): name for name in INDUSTRIES}


def _load_yaml(name: str) -> dict:
    with (APP_DIR / name).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_env(path: Path | None = None) -> None:
    """Minimal ``.env`` loader (no dependency). ``KEY=value`` lines, ``#`` comments;
    never overrides a variable already set in the real environment."""
    import os

    path = path or (REPO_ROOT / ".env")
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if value and value[0] in "\"'":                 # quoted -> take up to the closing quote
            quote = value[0]
            end = value.find(quote, 1)
            value = value[1:end] if end != -1 else value[1:]
        else:                                            # unquoted -> strip inline comment
            value = value.split(" #", 1)[0].split("\t#", 1)[0].strip()
        os.environ.setdefault(key.strip(), value)


@lru_cache(maxsize=1)
def schema_map() -> dict:
    return _load_yaml("schema_map.yaml")


@lru_cache(maxsize=1)
def _regions_raw() -> dict:
    return _load_yaml("regions.yaml")


@lru_cache(maxsize=1)
def _lexicon_raw() -> dict:
    return _load_yaml("lexicon.yaml")


@lru_cache(maxsize=1)
def countries() -> frozenset[str]:
    return frozenset(_regions_raw()["countries"])


@dataclass(frozen=True)
class RegionResolution:
    """Result of resolving one geographic term from a mandate."""

    term: str
    countries: tuple[str, ...] = ()
    known: bool = False          # was the term in regions.yaml / a country / an alias?
    empty_region: bool = False   # known region concept, but no country in the data

    @property
    def ambiguous(self) -> bool:
        return not self.known


def resolve_region(term: str) -> RegionResolution:
    """Resolve a country / demonym / region word to concrete country names.

    Unknown terms come back with ``known=False`` so the caller can record an
    ambiguity instead of guessing.
    """
    raw = term.strip()
    key = raw.lower()
    data = _regions_raw()

    # direct country name
    for country in data["countries"]:
        if country.lower() == key:
            return RegionResolution(raw, (country,), known=True)

    # country alias / demonym
    alias = {k.lower(): v for k, v in (data.get("country_aliases") or {}).items()}
    if key in alias:
        return RegionResolution(raw, (alias[key],), known=True)

    # multi-country region
    regions = {k.lower(): v for k, v in (data.get("regions") or {}).items()}
    if key in regions:
        resolved = tuple(regions[key])
        return RegionResolution(raw, resolved, known=True, empty_region=not resolved)

    return RegionResolution(raw, (), known=False)


def canonical_industry(value: str) -> str | None:
    """Case-insensitive match of a proposed industry to the 10-label enum."""
    return _INDUSTRY_BY_LOWER.get(value.strip().lower())


@lru_cache(maxsize=1)
def core_topics() -> tuple[str, ...]:
    """The capability phrases that actually occur in companies.json (>= 50 rows
    each). Written by ``python -m app.profile_dataset`` to
    ``data/dataset_vocab.json``; falls back to the distinct topics declared in
    ``lexicon.yaml`` if that artifact is absent (every core topic is in the
    lexicon, so the fallback is complete)."""
    path = REPO_ROOT / "data" / "dataset_vocab.json"
    try:
        topics = json.loads(path.read_text(encoding="utf-8"))["core_topics"]
        if topics:
            return tuple(topics)
    except (OSError, KeyError, ValueError):
        pass
    seen: dict[str, None] = {}
    for hit in lexicon().values():
        for t in hit.topics:
            seen.setdefault(t, None)
    return tuple(seen)


import re as _re

# British -> American spelling for the -ise/-yse families only (safe: no false
# positives on real words). Covers "optimisation", "personalised", "analyse", …
_SPELLING_SUBS = [
    (_re.compile(r"isation\b"), "ization"),
    (_re.compile(r"isations\b"), "izations"),
    (_re.compile(r"ise\b"), "ize"),
    (_re.compile(r"ised\b"), "ized"),
    (_re.compile(r"ising\b"), "izing"),
    (_re.compile(r"iser\b"), "izer"),
    (_re.compile(r"yse\b"), "yze"),
    (_re.compile(r"ysed\b"), "yzed"),
    (_re.compile(r"ysing\b"), "yzing"),
]


def normalise_spelling(text: str) -> str:
    """Normalise British -ise/-yse spellings to American so lexicon and FTS
    lookups match ("supply chain optimisation" -> "...optimization")."""
    out = text
    for pattern, repl in _SPELLING_SUBS:
        out = pattern.sub(repl, out)
    return out


@dataclass(frozen=True)
class LexiconHit:
    phrase: str
    industry: str | None
    topics: tuple[str, ...] = ()


@lru_cache(maxsize=1)
def lexicon() -> dict[str, LexiconHit]:
    out: dict[str, LexiconHit] = {}
    for phrase, spec in (_lexicon_raw().get("phrases") or {}).items():
        industry = spec.get("industry")
        if industry is not None:
            industry = canonical_industry(industry) or industry
        out[phrase.lower()] = LexiconHit(
            phrase=phrase,
            industry=industry,
            topics=tuple(spec.get("topics") or ()),
        )
    return out


def lookup_phrases(text: str) -> list[LexiconHit]:
    """Every lexicon phrase that occurs as a substring of ``text`` (longest first).
    Spelling and hyphenation are normalised first ("fraud-detection" -> "fraud
    detection") so British and compound forms still match."""
    lowered = normalise_spelling(text.lower()).replace("-", " ")
    hits = [hit for key, hit in lexicon().items() if key in lowered]
    hits.sort(key=lambda h: len(h.phrase), reverse=True)
    return hits

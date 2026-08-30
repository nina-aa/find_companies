"""Domain-phrase lexicon: phrase -> {industry, topics} safety net."""

import pytest

from app import config


@pytest.mark.parametrize(
    "phrase, industry, topic",
    [
        ("drug discovery", "Biotech", "drug discovery"),
        ("fraud detection", "Fintech", "fraud detection"),
        ("energy forecasting", "Energy", "energy forecasting"),
        ("smart grid", "Energy", "smart grid"),
        ("autonomous driving", "Automotive", "autonomous driving"),
        ("personalized learning", "Education", "personalized learning"),
        ("5G analytics", "Telecom", "5G analytics"),
    ],
)
def test_phrase_maps_to_industry_and_topic(phrase, industry, topic):
    hits = config.lookup_phrases(f"companies working on {phrase} in Europe")
    match = next(h for h in hits if h.phrase.lower() == phrase.lower())
    assert match.industry == industry
    assert topic in match.topics


def test_every_lexicon_industry_is_in_the_enum():
    for hit in config.lexicon().values():
        assert hit.industry is None or hit.industry in config.INDUSTRIES


def test_cybersecurity_has_no_industry():
    hits = {h.phrase.lower(): h for h in config.lookup_phrases("exclude cybersecurity consultancies")}
    assert hits["cybersecurity"].industry is None
    assert "cybersecurity" in hits["cybersecurity"].topics


def test_lookup_prefers_longer_phrases_first():
    hits = config.lookup_phrases("supply chain visibility platforms")
    # "supply chain visibility" (23 chars) must rank before "supply chain" (12)
    phrases = [h.phrase for h in hits]
    assert phrases.index("supply chain visibility") < phrases.index("supply chain")


def test_no_match_returns_empty():
    assert config.lookup_phrases("a completely unrelated sentence about weather") == []


@pytest.mark.parametrize(
    "british, american",
    [
        ("supply chain optimisation", "supply chain optimization"),
        ("Personalised learning", "personalized learning"),
        ("analyse the data", "analyze the data"),
        ("customised platform", "customized platform"),
    ],
)
def test_normalise_spelling(british, american):
    assert config.normalise_spelling(british.lower()) == american


@pytest.mark.parametrize(
    "phrase, industry",
    [
        ("Personalised learning", "Education"),          # british spelling
        ("supply chain optimisation", "Logistics"),      # british + generic tail
        ("cybercrime prevention", "Fintech"),            # synonym -> fraud detection
        ("precision medicine", "Biotech"),               # synonym
    ],
)
def test_lexicon_handles_spelling_and_synonyms(phrase, industry):
    hits = config.lookup_phrases(phrase)
    assert any(h.industry == industry for h in hits), (phrase, [h.phrase for h in hits])

"""Prompt builders for the two LLM calls. Kept as pure functions returning the
``messages`` list so they are trivial to unit-test (assert what the model is
actually told)."""

from __future__ import annotations

import json

from app import config
from app.schemas import Candidate
from app.state import MandateConstraints, MandateCriteria, MandateExclusions, SearchPlan

_INDUSTRIES = ", ".join(sorted(config.INDUSTRIES))
_REGIONS = ", ".join(sorted((config._regions_raw().get("regions") or {}).keys()))


# --------------------------------------------------------------------------- #
# interpret_mandate
# --------------------------------------------------------------------------- #
INTERPRET_SYSTEM = f"""\
You convert a natural-language company-search mandate into structured criteria.

Rules:
- Separate MANDATORY requirements from soft PREFERENCES from EXCLUSIONS, and put
  each constraint on ONLY ONE side.
  - MANDATORY: "must", "only", "required", or a bare stated fact
    ("Finnish fintech companies", "in the energy sector").
  - PREFERENCE: anything hedged with "prefer", "preferably", "ideally",
    "lean towards", "nice to have", "if possible". A whole sentence like
    "Prefer companies founded after 2015 with fewer than 250 employees" puts
    BOTH the founding-year bound AND the employee bound in `preferences`, never
    in `mandatory`.
  - EXCLUSION: "exclude", "not", "avoid", "without".
- Resolve relative terms to concrete values:
  - "founded after 2015" -> founded_year_gte: 2016 ; "since 2010" -> gte: 2010
  - "fewer than 250 employees" -> employee_count_lte: 249
  - A single country name goes in `countries` ("Finland", "Germany", "USA").
    `regions` is ONLY for multi-country groupings you must not expand yourself
    (e.g. "Nordic", "Europe", "Benelux"). Known region words: {_REGIONS}.
- `industries` MUST be drawn from this fixed list (map domain terms onto it):
  {_INDUSTRIES}.
  A domain phrase maps to BOTH an industry AND a capability, e.g.
  "drug-discovery companies" -> industries:["Biotech"], capabilities_all:["drug discovery"];
  "fintech working on fraud detection" -> industries:["Fintech"], capabilities_all:["fraud detection"].
  If a term has no matching industry (e.g. "cybersecurity"), leave `industries`
  empty and put it in capabilities or exclusions as appropriate.
- "X or Y" over a list field -> both go in capabilities_any (or the list); it is
  set membership, never a contradiction. "X and Y" -> capabilities_all.
- Put revenue bounds in revenue_eur_gte / revenue_eur_lte as integer euros
  ("below EUR 10M" -> revenue_eur_lte: 10000000).
- Any numeric bound that the mandate does NOT specify must be null, never 0.
- The word "prefer" / "preferably" / "ideally" makes EVERY constraint in that
  clause a preference, even constraints stated as "after 2018" or "more than 100".
  Example: "preferably founded after 2018" -> preferences.founded_year_gte: 2019,
  and mandatory.founded_year_gte stays null.
- `serves` holds customer/market phrases that are not structured fields
  ("serves European banks" -> serves:["European banks"]).
- `semantic_focus`: a short phrase capturing the overall intent.
- `ambiguities`: list anything genuinely unclear (vague size terms, unknown
  regions, whether a constraint is mandatory). Do not invent constraints to
  resolve them.
"""

# Worked examples — gpt-4o-mini follows the mandatory/preference split far more
# reliably from demonstrations than from rules alone. Built from the real models so
# every field (and its null) is present. Deliberately not evaluation queries.
_EXAMPLES: list[tuple[str, MandateCriteria]] = [
    (
        "Logistics companies in Germany or France focused on route optimization. "
        "Ideally founded after 2012 and with more than 100 employees.",
        MandateCriteria(
            mandatory=MandateConstraints(
                countries=["Germany", "France"], industries=["Logistics"],
                capabilities_any=["route optimization"],
            ),
            preferences=MandateConstraints(founded_year_gte=2013, employee_count_gte=101),
            semantic_focus="logistics route-optimization companies in Germany or France",
        ),
    ),
    (
        "Nordic biotech firms, must work on gene editing, preferably founded after 2020.",
        MandateCriteria(
            mandatory=MandateConstraints(
                regions=["Nordic"], industries=["Biotech"],
                capabilities_all=["gene editing"],
            ),
            preferences=MandateConstraints(founded_year_gte=2021),
            semantic_focus="Nordic biotech companies working on gene editing",
        ),
    ),
]


def interpret_messages(query: str) -> list[dict]:
    messages = [{"role": "system", "content": INTERPRET_SYSTEM}]
    for example_q, example_a in _EXAMPLES:
        messages.append({"role": "user", "content": f"Mandate:\n{example_q}"})
        messages.append({"role": "assistant",
                         "content": example_a.model_dump_json()})
    messages.append({"role": "user", "content": f"Mandate:\n{query.strip()}"})
    return messages


# --------------------------------------------------------------------------- #
# validate_and_rank
# --------------------------------------------------------------------------- #
VALIDATE_SYSTEM = """\
For each candidate company, judge whether the record supports specific
requirements, and extract evidence. You do NOT decide the overall verdict — that
is computed from your findings.

Every candidate has ALREADY passed the structured filters (country, industry,
numeric bounds); the `already_verified_do_not_recheck` block lists them. Ignore
those. Answer only what is asked:

- `capabilities_to_check`: for EACH phrase, add one entry to `capability_findings`
  with `requirement` set to that exact phrase and `supported` true/false.
- `serves_to_check`: for EACH phrase, add one entry to `serves_findings`. Thin
  descriptions often will not state the customer — then `supported` is false with
  a short `note`.
- `preferences_to_check`: for each, add a `preference_signals` entry.
- If a list is empty, return an empty findings list for it. Never invent a
  requirement that was not given to you.

For every finding:
- `quote`: a VERBATIM substring copied from `source_field` ("description" or
  "name"). If you cannot quote it, leave `quote` empty and use `note`. Never
  paraphrase inside `quote`.

relevance_score: 0.0-1.0 — overall strength of fit including preferences.
rationale: one sentence.
"""


def _candidate_block(cand: Candidate, plan: SearchPlan) -> dict:
    """What the model is told about one candidate — the record plus the
    deterministic signals we already hold (lesson 10)."""
    return {
        "candidate_id": cand.id,
        "record": {
            "name": cand.name,
            "description": cand.description,
            "industry": cand.industry,
            "location": cand.location,
            "founded_year": cand.founded_year,
            "employee_count": cand.employee_count,
            "revenue_range": cand.revenue_range,
        },
        "already_verified_do_not_recheck": {
            "industry": plan.filters.industries,
            "location": plan.filters.resolved_locations(),
            "founded_year": [plan.filters.founded_year_gte, plan.filters.founded_year_lte],
            "employee_count": [plan.filters.employee_count_gte, plan.filters.employee_count_lte],
        },
        "capabilities_to_check": plan.topic_terms,
        "capability_match_rule": (
            "at_least_one" if plan.topic_mode == "any" else "all"
        ),
        "serves_to_check": plan.serves,
        "capability_terms_already_found_in_text": cand.matched_topics,
        "preferences_to_check": plan.preferences.model_dump(exclude_defaults=True),
    }


def validate_messages(
    query: str, criteria: MandateCriteria, plan: SearchPlan, candidates: list[Candidate]
) -> list[dict]:
    payload = {
        "mandate": query.strip(),
        "semantic_focus": criteria.semantic_focus,
        "exclusions": criteria.exclusions.model_dump(exclude_defaults=True),
        "candidates": [_candidate_block(c, plan) for c in candidates],
    }
    return [
        {"role": "system", "content": VALIDATE_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]

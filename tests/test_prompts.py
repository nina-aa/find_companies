"""The interpret prompt is a contract with the model — assert the rules that keep
the parse honest are actually stated."""

from app.prompts import INTERPRET_SYSTEM, VALIDATE_SYSTEM, interpret_messages, validate_messages
from app.schemas import Candidate
from app.state import MandateConstraints, MandateCriteria, SearchPlan


def test_interpret_prompt_has_the_serves_rule():
    s = INTERPRET_SYSTEM.lower()
    assert "serves" in s
    # a customer phrase must not become a country filter
    assert "never copy the customer" in s or "not where it" in s


def test_interpret_prompt_forbids_fabricated_thresholds_for_vague_words():
    s = INTERPRET_SYSTEM.lower()
    for word in ("innovative", "startup", "fast-growing"):
        assert word in s
    assert "never invent a threshold" in s
    assert "ambiguities" in s


def test_interpret_prompt_keeps_topic_as_capability_alongside_industry():
    s = INTERPRET_SYSTEM.lower()
    assert "renewable-energy companies" in s
    assert "capabilities_any" in s


def test_interpret_prompt_carries_the_dataset_topic_vocabulary():
    from app import config

    topics = config.core_topics()
    assert len(topics) >= 27
    for t in ("drug discovery", "fraud detection", "gene editing",
              "route optimization", "molecular analysis"):
        assert t in topics
        assert f'"{t}"' in INTERPRET_SYSTEM          # rendered into the prompt
    assert "cancer research" in INTERPRET_SYSTEM     # the worked synonym example


def test_interpret_messages_carry_examples_and_the_query():
    msgs = interpret_messages("Find fintech companies in Finland")
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["content"].endswith("Find fintech companies in Finland")
    # few-shot examples are present (>=4 worked answers)
    assert sum(1 for m in msgs if m["role"] == "assistant") >= 4


def test_validate_prompt_hands_over_already_verified_signals():
    plan = SearchPlan()
    plan.filters = plan.filters.model_copy(update={"industries": ["Fintech"]})
    plan.topic_terms = ["fraud detection"]
    plan.serves = ["European banks"]
    cand = Candidate(id=1, name="X", description="d", industry="Fintech", location="UK",
                     founded_year=2019, employee_count=10, revenue_range="1M-10M")
    msgs = validate_messages("q", MandateCriteria(), plan, [cand])
    body = msgs[-1]["content"]
    assert "already_verified_do_not_recheck" in body
    assert "serves_to_check" in body and "European banks" in body
    assert "You do NOT decide the ranking" in VALIDATE_SYSTEM

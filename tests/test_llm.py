"""LLM layer: fabrication rule, fake provider, repair retry, provider-namespaced cache."""

import pytest
from pydantic import BaseModel, Field

from app.llm import (
    FakeProvider,
    LLMClient,
    LLMError,
    RawUsage,
    ResponseCache,
    estimate_cost,
    fabricate,
)
from app.state import MandateCriteria, ValidationBatch


class Sample(BaseModel):
    required_str: str
    required_int: int
    with_default: str = "kept"
    with_factory: list[str] = Field(default_factory=lambda: ["kept"])
    optional: int | None = None


def test_fabricate_only_fills_genuinely_required_fields():
    obj = fabricate(Sample)
    assert obj.required_str == "fake-required_str"
    assert obj.required_int == 0
    assert obj.with_default == "kept"          # default not clobbered (lesson 1)
    assert obj.with_factory == ["kept"]        # factory not clobbered
    assert obj.optional is None


def test_fabricate_nested_and_lists():
    vb = fabricate(ValidationBatch)
    assert vb.judgements == []
    mc = fabricate(MandateCriteria)
    assert mc.model_dump(exclude_defaults=True) == {}


def test_fake_provider_returns_schema_valid_instance():
    provider = FakeProvider()
    client = LLMClient(provider)
    result = client.complete([{"role": "user", "content": "x"}], MandateCriteria)
    assert isinstance(result.parsed, MandateCriteria)
    assert result.usage.est_cost_usd == 0.0
    assert result.provider == "fake"


def test_fake_provider_canned_response():
    canned = MandateCriteria(semantic_focus="canned")
    client = LLMClient(FakeProvider(responses={MandateCriteria: canned}))
    result = client.complete([{"role": "user", "content": "x"}], MandateCriteria)
    assert result.parsed.semantic_focus == "canned"


def test_repair_retry_on_validation_failure():
    calls = {"n": 0}

    def handler(messages, model):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"judgements": "not-a-list"}          # invalid -> triggers repair
        # second call sees the error fed back
        assert "failed schema validation" in messages[-1]["content"]
        return ValidationBatch(judgements=[])

    client = LLMClient(FakeProvider(handler=handler))
    result = client.complete([{"role": "user", "content": "x"}], ValidationBatch)
    assert calls["n"] == 2
    assert result.repaired is True
    assert result.attempts == 2


def test_repair_retry_gives_up_after_two_attempts():
    def handler(messages, model):
        return {"judgements": "still-bad"}

    client = LLMClient(FakeProvider(handler=handler))
    with pytest.raises(LLMError) as exc:
        client.complete([{"role": "user", "content": "x"}], ValidationBatch)
    assert exc.value.kind == "schema"
    assert exc.value.attempts == 2


def test_estimate_cost():
    cost = estimate_cost("gpt-4o-mini", RawUsage(prompt_tokens=1_000_000, completion_tokens=500_000))
    assert cost == pytest.approx(0.15 + 0.30)
    assert estimate_cost("unknown-model", RawUsage(prompt_tokens=999)) == 0.0


def test_cache_round_trip(tmp_path):
    cache = ResponseCache(tmp_path / "c.json")
    msgs = [{"role": "user", "content": "hello"}]
    key = ResponseCache.key("fake", MandateCriteria, msgs)
    cache.put(key, MandateCriteria(semantic_focus="v"), RawUsage(prompt_tokens=3))
    parsed, raw = cache.get(key, MandateCriteria)
    assert parsed.semantic_focus == "v"
    assert raw.prompt_tokens == 3
    # survives a reload
    assert ResponseCache(tmp_path / "c.json").get(key, MandateCriteria)[0].semantic_focus == "v"


def test_cache_key_is_namespaced_by_provider(tmp_path):
    """A value cached under the fake provider must never be served for openai."""
    msgs = [{"role": "user", "content": "same message"}]
    fake_key = ResponseCache.key("fake", MandateCriteria, msgs)
    real_key = ResponseCache.key("openai:gpt-4o-mini", MandateCriteria, msgs)
    assert fake_key != real_key

    cache = ResponseCache(tmp_path / "c.json")
    cache.put(fake_key, MandateCriteria(semantic_focus="from-fake"), RawUsage())
    assert cache.get(real_key, MandateCriteria) is None


def test_client_uses_cache_and_flags_hit(tmp_path):
    cache = ResponseCache(tmp_path / "c.json")
    seen = {"n": 0}

    def handler(messages, model):
        seen["n"] += 1
        return MandateCriteria(semantic_focus="fresh")

    client = LLMClient(FakeProvider(handler=handler), cache=cache)
    msgs = [{"role": "user", "content": "q"}]
    first = client.complete(msgs, MandateCriteria)
    second = client.complete(msgs, MandateCriteria)
    assert seen["n"] == 1              # second call served from cache
    assert first.cached is False and second.cached is True

"""LLM layer — the single seam between the workflow and a language model.

* ``LLMClient.complete(messages, response_model)`` is the only entry point the
  nodes use. It returns a parsed Pydantic instance plus usage / cost / lineage.
* ``Provider`` is the swap point: ``OpenAIProvider`` (hosted ``gpt-4o-mini``) or
  ``FakeProvider`` (schema-valid dummies, zero tokens — the default in tests/CI).
* One **repair retry** on a schema-validation failure, then a structured error.
  A 429 gets one ``Retry-After`` backoff, then fails the run gracefully.
* The response cache is namespaced by provider identity, so a ``fake`` answer is
  never served for a real call (and vice versa).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ValidationError

from app import config

# USD per 1,000,000 tokens (input, output). Verified against OpenAI pricing
# (Aug 2026): gpt-4o-mini $0.15 / $0.60, stable since release. Cached-input
# discount ($0.075) is ignored here — noted in the README.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}

MAX_ATTEMPTS_PER_CALL = 2   # 1 initial + 1 repair retry


class LLMError(RuntimeError):
    """Raised when a logical LLM call cannot produce a valid result. Carries a
    stage-friendly payload so the node can degrade instead of throwing upward."""

    def __init__(self, message: str, *, kind: str, attempts: int = 0):
        super().__init__(message)
        self.kind = kind          # "schema" | "rate_limit" | "provider" | "refusal"
        self.attempts = attempts


class RawUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    est_cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def estimate_cost(model: str, usage: RawUsage) -> float:
    in_price, out_price = PRICES.get(model, (0.0, 0.0))
    return round(
        usage.prompt_tokens / 1e6 * in_price
        + usage.completion_tokens / 1e6 * out_price,
        6,
    )


@dataclass
class LLMResult:
    parsed: BaseModel
    usage: Usage
    model: str
    provider: str
    attempts: int = 1
    repaired: bool = False
    cached: bool = False


# --------------------------------------------------------------------------- #
# providers
# --------------------------------------------------------------------------- #
class Provider(Protocol):
    identity: str          # e.g. "fake" or "openai:gpt-4o-mini"
    model: str

    def complete(
        self, messages: list[dict], response_model: type[BaseModel]
    ) -> tuple[BaseModel, RawUsage]:
        ...


class OpenAIProvider:
    """Thin wrapper over ``client.chat.completions.parse`` — strict json-schema
    structured output, one Retry-After backoff on 429."""

    def __init__(self, model: str = "gpt-4o-mini", *, api_key: str | None = None,
                 temperature: float = 0.0, timeout: float = 40.0):
        import openai

        self.model = model
        self.identity = f"openai:{model}"
        self.temperature = temperature
        self._openai = openai
        self._client = openai.OpenAI(api_key=api_key, timeout=timeout)

    def complete(self, messages, response_model):
        for attempt in (1, 2):
            try:
                completion = self._client.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                    response_format=response_model,
                    temperature=self.temperature,
                )
                break
            except self._openai.RateLimitError as exc:
                if attempt == 2:
                    raise LLMError(str(exc), kind="rate_limit")
                delay = _retry_after_seconds(exc) or 2.0
                time.sleep(min(delay, 10.0))
            except self._openai.APIError as exc:
                raise LLMError(str(exc), kind="provider")

        message = completion.choices[0].message
        if getattr(message, "refusal", None):
            raise LLMError(message.refusal, kind="refusal")
        if message.parsed is None:
            raise LLMError("model returned no parseable content", kind="schema")

        u = completion.usage
        return message.parsed, RawUsage(
            prompt_tokens=getattr(u, "prompt_tokens", 0),
            completion_tokens=getattr(u, "completion_tokens", 0),
        )


def _retry_after_seconds(exc) -> float | None:
    try:
        value = exc.response.headers.get("retry-after")
        return float(value) if value is not None else None
    except Exception:
        return None


class FakeProvider:
    """Returns a schema-valid dummy instance of ``response_model``. Optionally
    seeded with canned responses keyed by model class (``responses``) or a
    callable ``handler(messages, response_model) -> BaseModel``."""

    identity = "fake"
    model = "fake"

    def __init__(self, *, responses: dict | None = None, handler=None):
        self._responses = dict(responses or {})
        self._handler = handler

    def complete(self, messages, response_model):
        if self._handler is not None:
            return self._handler(messages, response_model), RawUsage()
        if response_model in self._responses:
            queued = self._responses[response_model]
            value = queued.pop(0) if isinstance(queued, list) else queued
            return value, RawUsage()
        return fabricate(response_model), RawUsage()


def fabricate(model: type[BaseModel]):
    """Build a minimal schema-valid instance.

    Rule (lesson 1): only supply a value for a field that is *genuinely required*
    (no default and no default_factory). Never overwrite a field that has a
    default — let Pydantic apply it.
    """
    values: dict = {}
    for name, fld in model.model_fields.items():
        if not fld.is_required():
            continue
        values[name] = _dummy_for(fld.annotation, name)
    return model(**values)


def _dummy_for(annotation, name: str):
    import typing

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if origin in (list, set, tuple):
        return []
    if origin is dict:
        return {}
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return fabricate(annotation)
    if annotation is bool:
        return False
    if annotation in (int, float):
        return 0
    if annotation is str:
        return f"fake-{name}"
    # Optional / Union: pick the first non-None arg, else None
    for arg in args:
        if arg is not type(None):
            return _dummy_for(arg, name)
    return None


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #
class ResponseCache:
    """File-backed JSON cache. Every key is prefixed with the provider identity,
    so switching providers can never surface a stale cross-provider answer."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._data: dict = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}

    @staticmethod
    def key(provider_identity: str, response_model: type[BaseModel],
            messages: list[dict]) -> str:
        import hashlib

        blob = json.dumps(
            [provider_identity, response_model.__name__, messages],
            sort_keys=True, ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str, response_model: type[BaseModel]):
        entry = self._data.get(key)
        if entry is None:
            return None
        return response_model.model_validate(entry["parsed"]), RawUsage(**entry["usage"])

    def put(self, key: str, parsed: BaseModel, usage: RawUsage) -> None:
        self._data[key] = {
            "parsed": parsed.model_dump(mode="json"),
            "usage": usage.model_dump(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8"
        )


# --------------------------------------------------------------------------- #
# client
# --------------------------------------------------------------------------- #
class LLMClient:
    def __init__(self, provider: Provider, *, cache: ResponseCache | None = None):
        self.provider = provider
        self.cache = cache

    def complete(
        self, messages: list[dict], response_model: type[BaseModel]
    ) -> LLMResult:
        cache_key = None
        if self.cache is not None:
            cache_key = ResponseCache.key(self.provider.identity, response_model, messages)
            hit = self.cache.get(cache_key, response_model)
            if hit is not None:
                parsed, raw = hit
                return LLMResult(
                    parsed=parsed,
                    usage=Usage(prompt_tokens=raw.prompt_tokens,
                               completion_tokens=raw.completion_tokens,
                               est_cost_usd=estimate_cost(self.provider.model, raw)),
                    model=self.provider.model,
                    provider=self.provider.identity,
                    attempts=0,
                    cached=True,
                )

        attempts = 0
        last_error: ValidationError | None = None
        work_messages = list(messages)

        while attempts < MAX_ATTEMPTS_PER_CALL:
            attempts += 1
            raw_parsed, raw = self.provider.complete(work_messages, response_model)
            try:
                # provider may return an instance or a raw dict — validate either
                parsed = response_model.model_validate(raw_parsed)
            except ValidationError as exc:
                last_error = exc
                work_messages = messages + [{
                    "role": "user",
                    "content": (
                        "Your previous response failed schema validation with:\n"
                        f"{exc}\nReturn a corrected response that matches the schema."
                    ),
                }]
                continue

            usage = Usage(
                prompt_tokens=raw.prompt_tokens,
                completion_tokens=raw.completion_tokens,
                est_cost_usd=estimate_cost(self.provider.model, raw),
            )
            if self.cache is not None and cache_key is not None:
                self.cache.put(cache_key, parsed, raw)
            return LLMResult(
                parsed=parsed, usage=usage,
                model=self.provider.model, provider=self.provider.identity,
                attempts=attempts, repaired=attempts > 1,
            )

        raise LLMError(
            f"schema validation failed after {attempts} attempts: {last_error}",
            kind="schema", attempts=attempts,
        )


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #
DEFAULT_CACHE_PATH = config.REPO_ROOT / "data" / "cache" / "llm_cache.json"


def build_client(
    provider: str = "fake",
    *,
    model: str = "gpt-4o-mini",
    use_cache: bool = True,
    api_key: str | None = None,
    cache_path: Path | str = DEFAULT_CACHE_PATH,
) -> LLMClient:
    if provider == "openai":
        impl: Provider = OpenAIProvider(model=model, api_key=api_key)
        # The cache only matters for the paid provider — it makes eval/dev re-runs
        # fast and near-free. The fake provider is already instant and free.
        cache = ResponseCache(cache_path) if use_cache else None
    elif provider == "fake":
        impl = FakeProvider()
        cache = None
    else:
        raise ValueError(f"unknown provider {provider!r}")
    return LLMClient(impl, cache=cache)

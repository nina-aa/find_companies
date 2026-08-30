"""HTTP API — thin wrapper behaviour, auth guard, response shape. Fake provider."""

import os

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(monkeypatch, fixture_db):
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.delenv("AGENT_API_KEY", raising=False)
    # point the workflow at the fixture index
    import app.workflow as wf
    from app.db import DEFAULT_DB
    orig = wf.run_workflow

    def patched(query, cfg=None, **kw):
        kw.setdefault("db_path", fixture_db)
        return orig(query, cfg, **kw)

    monkeypatch.setattr(wf, "run_workflow", patched)
    import app.api as api
    monkeypatch.setattr(api, "run_workflow", patched)
    return TestClient(api.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code in (200, 503)


def test_search_returns_agent_response_shape(client):
    r = client.post("/agent/search", json={"query": "fintech in Finland fraud detection"})
    assert r.status_code == 200
    body = r.json()
    for key in ("run_id", "query", "interpreted_mandate", "search_plan", "results",
                "revision", "metadata"):
        assert key in body
    assert isinstance(body["results"], list)
    assert "llm_calls" in body["metadata"]


def test_api_key_guard(client, monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", "secret")
    assert client.post("/agent/search", json={"query": "x"}).status_code == 401
    ok = client.post("/agent/search", json={"query": "x"}, headers={"X-API-Key": "secret"})
    assert ok.status_code == 200


def test_rejects_empty_query(client):
    assert client.post("/agent/search", json={"query": ""}).status_code == 422

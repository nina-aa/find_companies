"""FastAPI wrapper — a thin HTTP surface over ``run_workflow``.

``POST /agent/search`` runs the full workflow synchronously and returns the same
``AgentResponse`` the CLI produces. An ``X-API-Key`` shared secret protects the
deployed URL (every call spends real OpenAI budget); it is wallet-protection, not
a security boundary.

BudgetGuard's per-run deadline (``AGENT_DEADLINE_S``, ~90s) returns a partial body
with ``timed_out: true`` rather than hanging; clients should set their own request
timeout above that. No server-side HTTP request timeout is configured (uvicorn
default), so the deadline is the effective bound.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from app import config
from app.state import AgentResponse, RunConfig
from app.workflow import run_workflow

config.load_env()

app = FastAPI(title="Agentic Company Search", version="0.1.0")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    min_results: int = Field(default=3, ge=1, le=10)
    limit: int = Field(default=10, ge=1, le=50)


def _check_key(x_api_key: str | None) -> None:
    expected = os.environ.get("AGENT_API_KEY")
    if not expected:
        return  # unset -> open (local dev)
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


@app.get("/")
def root() -> dict:
    return {
        "service": "agentic company search",
        "endpoints": {"POST /agent/search": "run a mandate", "GET /health": "index status",
                      "GET /docs": "OpenAPI UI"},
    }


@app.get("/health")
def health() -> dict:
    from app.db import load_manifest

    try:
        manifest = load_manifest()
    except Exception as exc:  # index not built
        raise HTTPException(status_code=503, detail=f"index unavailable: {exc}")
    return {"status": "ok", "rows": manifest.get("row_count"),
            "embedding_model": manifest.get("embedding_model")}


@app.post("/agent/search", response_model=AgentResponse)
def agent_search(req: SearchRequest, x_api_key: str | None = Header(default=None)) -> AgentResponse:
    _check_key(x_api_key)
    provider = os.environ.get("LLM_PROVIDER", "openai")
    try:
        deadline_s = float(os.environ.get("AGENT_DEADLINE_S", "90"))
    except ValueError:
        deadline_s = 90.0

    cfg = RunConfig(
        provider="openai" if provider == "openai" else "fake",
        min_results=req.min_results,
        result_limit=req.limit,
        deadline_s=deadline_s,
    )
    try:
        response, _ = run_workflow(req.query, cfg)
    except Exception as exc:  # never leak a stack trace
        raise HTTPException(status_code=500, detail=f"workflow error: {exc}")
    return response

"""FastAPI wrapper — a thin HTTP surface over ``run_workflow``.

``POST /agent/search`` runs the full workflow synchronously and returns the same
``AgentResponse`` the CLI produces. An ``X-API-Key`` shared secret protects the
deployed URL (every call spends real OpenAI budget); it is wallet-protection, not
a security boundary.

Two nested timeouts: BudgetGuard's per-run deadline (``AGENT_DEADLINE_S``, ~90s)
returns a partial body with ``timed_out: true``; the outer HTTP timeout is the
last-resort ceiling. Deadline < HTTP timeout, so the client always gets JSON.
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
    min_results: int = 3
    engine: str = "driver"


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
        deadline_s=deadline_s,
        engine="graph" if req.engine == "graph" else "driver",
    )
    try:
        response, _ = run_workflow(req.query, cfg)
    except Exception as exc:  # never leak a stack trace
        raise HTTPException(status_code=500, detail=f"workflow error: {exc}")
    return response

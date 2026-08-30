# Single self-contained image: builds the SQLite/FTS5 index at build time and
# serves the FastAPI app. Targets Hugging Face Spaces (Docker SDK, port 7860)
# but runs anywhere.

FROM python:3.12-slim

# HF Spaces runs the container as a non-root user (uid 1000).
RUN useradd -m -u 1000 appuser
WORKDIR /app

COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser eval/ ./eval/
COPY --chown=appuser:appuser data/companies.json ./data/companies.json

USER appuser

# Build the retrieval index into the image so the container is self-contained
# and starts instantly.
RUN python -m app.ingest --input data/companies.json \
 && python -c "import json; m=json.load(open('data/index/manifest.json')); assert m['row_count']==50000"

ENV LLM_PROVIDER=openai \
    AGENT_DEADLINE_S=90 \
    LOG_LEVEL=INFO \
    PORT=7860

EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-7860}"]

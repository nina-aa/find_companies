# Single self-contained image: builds the SQLite/FTS5 index at build time and
# serves the FastAPI app. Runs anywhere that takes a container (Render, Cloud Run,
# a plain docker host); the process listens on $PORT (default 7860).

FROM python:3.12-slim

# Run as a non-root user (uid 1000).
RUN useradd -m -u 1000 appuser
WORKDIR /app

COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser eval/ ./eval/
COPY --chown=appuser:appuser data/companies.json ./data/companies.json

USER appuser

# Build the retrieval index + regenerate the dataset vocab into the image, so the
# container is self-contained and starts instantly.
RUN python -m app.ingest --input data/companies.json \
 && python -m app.profile_dataset \
 && python -c "import json; m=json.load(open('data/index/manifest.json')); assert m['row_count']==50000"

ENV LLM_PROVIDER=openai \
    AGENT_DEADLINE_S=90 \
    LOG_LEVEL=INFO \
    PORT=7860

EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-7860}"]

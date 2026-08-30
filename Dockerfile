# Placeholder — fleshed out in M5 (C9). Kept here so the repo skeleton is complete.
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY data/companies.json ./data/companies.json
COPY eval/ ./eval/

# Build the index at image-build time so the container is self-contained.
RUN python -m app.ingest --input data/companies.json

EXPOSE 7860
# CMD is added in M4 once the HTTP API exists.
CMD ["python", "-m", "app.cli", "--help"]

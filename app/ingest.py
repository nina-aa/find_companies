"""Build the retrieval index from the raw dataset.

``python -m app.ingest --input data/companies.json`` produces a self-contained,
reproducible SQLite database at ``data/index/companies.db``:

* ``companies``      — one typed row per company, plus derived revenue-in-euros columns.
* ``company_regions``— (company_id, region) lookup so region filters are a plain JOIN.
* ``companies_fts``  — FTS5 (bm25) virtual table over name + description.

and a manifest at ``data/index/manifest.json`` recording row count, the source
file hash, the build timestamp and the embedding model (``none`` for now). The
API/CLI can refuse to run if the manifest disagrees with the runtime config.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from app import config
from app.revenue import parse_bucket

REPO_ROOT = config.REPO_ROOT
DEFAULT_INPUT = REPO_ROOT / "data" / "companies.json"
DEFAULT_DB = REPO_ROOT / "data" / "index" / "companies.db"
DEFAULT_MANIFEST = REPO_ROOT / "data" / "index" / "manifest.json"

SCHEMA_VERSION = 1

_STRUCTURED_COLUMNS = [
    ("id", "INTEGER PRIMARY KEY"),
    ("name", "TEXT NOT NULL"),
    ("description", "TEXT NOT NULL DEFAULT ''"),
    ("industry", "TEXT"),
    ("location", "TEXT"),
    ("founded_year", "INTEGER"),
    ("employee_count", "INTEGER"),
    ("revenue_range", "TEXT"),
    ("revenue_min_eur", "INTEGER"),
    ("revenue_max_eur", "INTEGER"),
]


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
def _coerce(value, kind: str):
    if value is None:
        return None
    if kind == "int":
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    if kind == "str":
        return str(value)
    return value


def normalise_row(raw: dict, fields_spec: dict) -> dict:
    """Apply schema_map.yaml to one raw record -> canonical dict."""
    out: dict = {}
    for canonical, spec in fields_spec.items():
        value = _coerce(raw.get(spec["from"]), spec.get("type", "str"))
        if value is None and "default" in spec:
            value = spec["default"]
        if value is None and spec.get("required"):
            raise ValueError(f"row {raw!r} missing required field {canonical!r}")
        out[canonical] = value

    lo_hi = parse_bucket(out.get("revenue_range"))
    out["revenue_min_eur"] = lo_hi[0] if lo_hi else None
    out["revenue_max_eur"] = lo_hi[1] if lo_hi else None
    return out


def _regions_for(country: str | None) -> list[str]:
    """Every region name in regions.yaml whose country list contains ``country``."""
    if not country:
        return []
    raw = config._regions_raw()
    return sorted(
        name for name, members in (raw.get("regions") or {}).items()
        if country in members
    )


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def _load_source(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(input_path: Path, db_path: Path, manifest_path: Path) -> dict:
    smap = config.schema_map()
    fields_spec = smap["fields"]

    raw_rows = _load_source(input_path)
    rows = [normalise_row(r, fields_spec) for r in raw_rows]

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    try:
        col_ddl = ", ".join(f"{name} {decl}" for name, decl in _STRUCTURED_COLUMNS)
        conn.execute(f"CREATE TABLE companies ({col_ddl})")
        conn.execute(
            "CREATE TABLE company_regions ("
            "  company_id INTEGER NOT NULL REFERENCES companies(id),"
            "  region TEXT NOT NULL,"
            "  PRIMARY KEY (company_id, region)"
            ")"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE companies_fts USING fts5("
            "  name, description, content='', tokenize='porter unicode61'"
            ")"
        )

        col_names = [name for name, _ in _STRUCTURED_COLUMNS]
        placeholders = ", ".join("?" for _ in col_names)
        conn.executemany(
            f"INSERT INTO companies ({', '.join(col_names)}) VALUES ({placeholders})",
            [tuple(r[c] for c in col_names) for r in rows],
        )
        conn.executemany(
            "INSERT INTO companies_fts (rowid, name, description) VALUES (?, ?, ?)",
            [(r["id"], r["name"], r["description"]) for r in rows],
        )

        region_pairs: list[tuple[int, str]] = []
        for r in rows:
            for region in _regions_for(r.get("location")):
                region_pairs.append((r["id"], region))
        conn.executemany(
            "INSERT INTO company_regions (company_id, region) VALUES (?, ?)",
            region_pairs,
        )

        for col in ("industry", "location", "founded_year", "employee_count", "revenue_range"):
            conn.execute(f"CREATE INDEX idx_companies_{col} ON companies({col})")
        conn.execute("CREATE INDEX idx_company_regions_region ON company_regions(region)")
        conn.commit()
    finally:
        conn.close()

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "row_count": len(rows),
        "source_file": str(input_path.relative_to(REPO_ROOT)) if input_path.is_relative_to(REPO_ROOT) else str(input_path),
        "source_sha256": _sha256(input_path),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedding_model": "none",
        "industries": sorted(config.INDUSTRIES),
        "countries": sorted(config.countries()),
        "regions": sorted((config._regions_raw().get("regions") or {}).keys()),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the SQLite + FTS5 retrieval index.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_DB)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    manifest = build(args.input, args.output, args.manifest)
    elapsed = time.perf_counter() - started
    print(
        f"built {args.output} — {manifest['row_count']:,} rows, "
        f"embedding_model={manifest['embedding_model']}, {elapsed:.1f}s"
    )
    print(f"manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

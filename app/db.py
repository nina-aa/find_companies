"""Low-level SQLite access over the built index.

Just the plumbing: opening a read-only connection and reading the manifest. The
retrieval tools that use it live in ``app/tools.py``.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from app import config

DEFAULT_DB = config.REPO_ROOT / "data" / "index" / "companies.db"
DEFAULT_MANIFEST = config.REPO_ROOT / "data" / "index" / "manifest.json"


class IndexMissingError(RuntimeError):
    pass


def connect(db_path: Path | str = DEFAULT_DB, *, read_only: bool = True) -> sqlite3.Connection:
    path = Path(db_path)
    if not path.exists():
        raise IndexMissingError(
            f"no index at {path} — run `python -m app.cli ingest` first"
        )
    if read_only:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def load_manifest(manifest_path: Path | str = DEFAULT_MANIFEST) -> dict:
    path = Path(manifest_path)
    if not path.exists():
        raise IndexMissingError(f"no manifest at {path} — run `python -m app.cli ingest`")
    return json.loads(path.read_text(encoding="utf-8"))

"""SQLite-backed SOP document registry — tracks upload/ingestion status so the
management UI can list, retry, and remove knowledge-base documents.

Stdlib sqlite3, same pattern as registry.py (patient queue); one connection
per call is plenty at document-management rates.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "registry.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sop_documents (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    ext TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    status TEXT NOT NULL,
    chunk_count INTEGER DEFAULT 0,
    error_message TEXT,
    uploaded_at TEXT NOT NULL,
    ingested_at TEXT
);
"""


# Columns added after the first release; applied idempotently on connect (same
# ALTER-if-missing pattern as registry.py).
#   content_hash — fingerprint of the extracted text; identical files share it
#   topic_json   — generated topic card (name, kind, summary, example_questions)
_MIGRATIONS = {
    "content_hash": "ALTER TABLE sop_documents ADD COLUMN content_hash TEXT",
    "topic_json": "ALTER TABLE sop_documents ADD COLUMN topic_json TEXT",
    "topic_at": "ALTER TABLE sop_documents ADD COLUMN topic_at TEXT",
}


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    existing = {r["name"] for r in conn.execute("PRAGMA table_info(sop_documents)")}
    for column, ddl in _MIGRATIONS.items():
        if column not in existing:
            conn.execute(ddl)
    return conn


def add(document_id: str, name: str, file_path: str, ext: str, size_bytes: int,
       status: str = "uploaded") -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO sop_documents "
            "(id, name, file_path, ext, size_bytes, status, chunk_count, error_message, "
            "uploaded_at, ingested_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 0, NULL, ?, NULL)",
            (document_id, name, file_path, ext, size_bytes, status,
             datetime.now(timezone.utc).isoformat()),
        )


def update_status(document_id: str, status: str, *,
                  chunk_count: int | None = None, error_message: str | None = None) -> None:
    ingested_at = datetime.now(timezone.utc).isoformat() if status == "ingested" else None
    with _connect() as conn:
        conn.execute(
            "UPDATE sop_documents SET status = ?, chunk_count = COALESCE(?, chunk_count), "
            "error_message = ?, ingested_at = COALESCE(?, ingested_at) WHERE id = ?",
            (status, chunk_count, error_message, ingested_at, document_id),
        )


def get(document_id: str) -> sqlite3.Row | None:
    with _connect() as conn:
        return conn.execute(
            "SELECT * FROM sop_documents WHERE id = ?", (document_id,)
        ).fetchone()


def list_all() -> list[sqlite3.Row]:
    """Newest first."""
    with _connect() as conn:
        return conn.execute(
            "SELECT * FROM sop_documents ORDER BY uploaded_at DESC"
        ).fetchall()


def delete(document_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM sop_documents WHERE id = ?", (document_id,))


def set_content_hash(document_id: str, content_hash: str) -> None:
    with _connect() as conn:
        conn.execute("UPDATE sop_documents SET content_hash = ? WHERE id = ?",
                     (content_hash, document_id))


def clear_topic(document_id: str) -> None:
    """After a reingest the content may differ: recompute fingerprint and topic."""
    with _connect() as conn:
        conn.execute("UPDATE sop_documents SET content_hash = NULL, topic_json = NULL, "
                     "topic_at = NULL WHERE id = ?", (document_id,))


def set_topic(document_ids: list[str], topic_json: str) -> None:
    """Store one topic card on every document of a duplicate group."""
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.executemany(
            "UPDATE sop_documents SET topic_json = ?, topic_at = ? WHERE id = ?",
            [(topic_json, now, d) for d in document_ids],
        )

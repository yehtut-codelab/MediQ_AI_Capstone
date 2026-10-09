"""Thumbs up/down feedback on SOP Q&A answers, stored in registry.db.

One row per answer (session_id + message_id from the browser); rating again
overwrites, clearing the rating deletes the row. Keeps the question, answer and
persona alongside, so answer quality can be reviewed without the browser's
chat history (e.g. for the evaluation phase).
"""

import sqlite3
from datetime import datetime, timezone

from app.services.document_registry import DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS qa_feedback (
    session_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    rating TEXT NOT NULL CHECK (rating IN ('up', 'down')),
    reason TEXT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    persona TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (session_id, message_id)
);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def upsert(session_id: str, message_id: str, rating: str, reason: str | None,
           question: str, answer: str, persona: str | None) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO qa_feedback "
            "(session_id, message_id, rating, reason, question, answer, persona, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, message_id, rating, reason, question, answer, persona,
             datetime.now(timezone.utc).isoformat()),
        )


def clear(session_id: str, message_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM qa_feedback WHERE session_id = ? AND message_id = ?",
                     (session_id, message_id))


def summary() -> dict:
    """Counts overall and per persona, plus the most recent thumbs-down reasons."""
    with _connect() as conn:
        totals = {r["rating"]: r["n"] for r in conn.execute(
            "SELECT rating, COUNT(*) AS n FROM qa_feedback GROUP BY rating")}
        by_persona = [dict(r) for r in conn.execute(
            "SELECT COALESCE(persona, 'default') AS persona, "
            "SUM(rating = 'up') AS up, SUM(rating = 'down') AS down "
            "FROM qa_feedback GROUP BY persona ORDER BY persona")]
        recent_down = [dict(r) for r in conn.execute(
            "SELECT question, reason, persona, created_at FROM qa_feedback "
            "WHERE rating = 'down' ORDER BY created_at DESC LIMIT 20")]
    return {"up": totals.get("up", 0), "down": totals.get("down", 0),
            "by_persona": by_persona, "recent_down": recent_down}

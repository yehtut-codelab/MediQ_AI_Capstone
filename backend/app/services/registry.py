"""SQLite-backed patient arrivals registry — queue survives restarts.

Stdlib sqlite3; one connection per call is plenty at clinic-registration rates.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.schemas import RegisteredPatient

DB_PATH = Path(__file__).resolve().parents[2] / "registry.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS registrations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL,
    display_name TEXT,
    clinic TEXT NOT NULL,
    service_type TEXT NOT NULL,
    diagnosis TEXT,
    medical_history TEXT,
    current_issue TEXT,
    priority INTEGER DEFAULT 3,
    priority_label TEXT,
    triage_reasons TEXT,
    pathway_label TEXT,
    pathway TEXT,
    total_visit_min REAL,
    arrival_datetime TEXT NOT NULL,
    appointment_datetime TEXT,
    queue_position INTEGER NOT NULL,
    estimated_wait_min REAL NOT NULL,
    range_lo REAL NOT NULL,
    range_hi REAL NOT NULL,
    status_band TEXT NOT NULL,
    explanation TEXT NOT NULL,
    registered_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reg_queue
    ON registrations (clinic, service_type, arrival_datetime);
"""


# Columns added after the table first shipped; CREATE TABLE IF NOT EXISTS
# won't extend an existing DB, so ALTER idempotently on connect.
_MIGRATION_COLUMNS = [
    ("diagnosis", "TEXT"),
    ("medical_history", "TEXT"),
    ("current_issue", "TEXT"),
    ("priority", "INTEGER DEFAULT 3"),
    ("priority_label", "TEXT"),
    ("triage_reasons", "TEXT"),
    ("pathway_label", "TEXT"),
    ("pathway", "TEXT"),
    ("total_visit_min", "REAL"),
    ("model_category", "TEXT"),
    ("model_estimate_min", "REAL"),
    ("model_mae_min", "REAL"),
    ("model_queue_depth", "INTEGER"),
    ("model_congestion_state", "TEXT"),
    ("model_congestion_mean_wait_min", "REAL"),
    ("model_forecast", "TEXT"),
    ("model_forecast_trend", "TEXT"),
    ("model_replayed_at", "TEXT"),
    ("model_is_analog_replay", "INTEGER DEFAULT 0"),
    ("model_replay_note", "TEXT"),
    ("model_unavailable_reason", "TEXT"),
]


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    existing = {r["name"] for r in conn.execute("PRAGMA table_info(registrations)")}
    for col, sqltype in _MIGRATION_COLUMNS:
        if col not in existing:
            conn.execute(f"ALTER TABLE registrations ADD COLUMN {col} {sqltype}")
    return conn


def next_queue_position(clinic: str, service_type: str, arrival_date: str) -> int:
    """1 + count of same-day registrations for the same clinic + service."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM registrations "
            "WHERE clinic = ? AND service_type = ? AND date(arrival_datetime) = ?",
            (clinic, service_type, arrival_date),
        ).fetchone()
    return row["n"] + 1


def add(patient: RegisteredPatient) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO registrations (patient_id, display_name, clinic, service_type, "
            "diagnosis, medical_history, current_issue, "
            "priority, priority_label, triage_reasons, pathway_label, pathway, total_visit_min, "
            "arrival_datetime, appointment_datetime, queue_position, estimated_wait_min, "
            "range_lo, range_hi, status_band, explanation, registered_at, "
            "model_category, model_estimate_min, model_mae_min, model_queue_depth, "
            "model_congestion_state, model_congestion_mean_wait_min, model_forecast, "
            "model_forecast_trend, model_replayed_at, model_is_analog_replay, "
            "model_replay_note, model_unavailable_reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
            "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                patient.patient_id,
                patient.display_name,
                patient.clinic,
                patient.service_type,
                patient.diagnosis,
                patient.medical_history,
                patient.current_issue,
                patient.priority,
                patient.priority_label,
                json.dumps(patient.triage_reasons),
                patient.pathway_label,
                json.dumps([s.model_dump() for s in patient.pathway]),
                patient.total_visit_min,
                patient.arrival_datetime.isoformat(),
                patient.appointment_datetime.isoformat() if patient.appointment_datetime else None,
                patient.queue_position,
                patient.estimated_wait_min,
                patient.range_min[0],
                patient.range_min[1],
                patient.status_band,
                patient.explanation,
                patient.registered_at.isoformat(),
                patient.model_category,
                patient.model_estimate_min,
                patient.model_mae_min,
                patient.model_queue_depth,
                patient.model_congestion_state,
                patient.model_congestion_mean_wait_min,
                json.dumps([f.model_dump() for f in patient.model_forecast]),
                patient.model_forecast_trend,
                patient.model_replayed_at.isoformat() if patient.model_replayed_at else None,
                int(patient.model_is_analog_replay),
                patient.model_replay_note,
                patient.model_unavailable_reason,
            ),
        )


def counts_since(since: datetime) -> dict[tuple[str, str], int]:
    """{(clinic, service_type): registrations made at or after `since`}.
    Same "registered today" definition as list_all(since=...)."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT clinic, service_type, COUNT(*) AS n FROM registrations "
            "WHERE registered_at >= ? GROUP BY clinic, service_type",
            (since.astimezone(timezone.utc).isoformat(),),
        ).fetchall()
    return {(r["clinic"], r["service_type"]): r["n"] for r in rows}


def list_all(limit: int = 200, since: datetime | None = None) -> list[RegisteredPatient]:
    """Newest first. `since` (timezone-aware) keeps only rows registered at or
    after that instant; registered_at is stored as a UTC ISO string, so the
    comparison is done on UTC text."""
    with _connect() as conn:
        if since is None:
            rows = conn.execute(
                "SELECT * FROM registrations ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM registrations WHERE registered_at >= ? ORDER BY id DESC LIMIT ?",
                (since.astimezone(timezone.utc).isoformat(), limit),
            ).fetchall()
    return [
        RegisteredPatient(
            patient_id=r["patient_id"],
            display_name=r["display_name"],
            clinic=r["clinic"],
            service_type=r["service_type"],
            diagnosis=r["diagnosis"],
            medical_history=r["medical_history"],
            current_issue=r["current_issue"],
            priority=r["priority"] if r["priority"] is not None else 3,
            priority_label=r["priority_label"] or "P3 — Routine",
            triage_reasons=json.loads(r["triage_reasons"]) if r["triage_reasons"] else [],
            pathway_label=r["pathway_label"],
            pathway=json.loads(r["pathway"]) if r["pathway"] else [],
            total_visit_min=r["total_visit_min"],
            arrival_datetime=datetime.fromisoformat(r["arrival_datetime"]),
            appointment_datetime=(
                datetime.fromisoformat(r["appointment_datetime"])
                if r["appointment_datetime"] else None
            ),
            queue_position=r["queue_position"],
            estimated_wait_min=r["estimated_wait_min"],
            range_min=(r["range_lo"], r["range_hi"]),
            status_band=r["status_band"],
            explanation=r["explanation"],
            registered_at=datetime.fromisoformat(r["registered_at"]),
            model_category=r["model_category"],
            model_estimate_min=r["model_estimate_min"],
            model_mae_min=r["model_mae_min"],
            model_queue_depth=r["model_queue_depth"],
            model_congestion_state=r["model_congestion_state"],
            model_congestion_mean_wait_min=r["model_congestion_mean_wait_min"],
            model_forecast=json.loads(r["model_forecast"]) if r["model_forecast"] else [],
            model_forecast_trend=r["model_forecast_trend"],
            model_replayed_at=(
                datetime.fromisoformat(r["model_replayed_at"])
                if r["model_replayed_at"] else None
            ),
            model_is_analog_replay=bool(r["model_is_analog_replay"]),
            model_replay_note=r["model_replay_note"],
            model_unavailable_reason=r["model_unavailable_reason"],
        )
        for r in rows
    ]

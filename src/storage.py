import json
import os
import sqlite3
from datetime import UTC, datetime
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS prediction_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    input_json TEXT NOT NULL,
    n_records INTEGER NOT NULL,
    output_json TEXT,
    latency_ms REAL,
    status TEXT NOT NULL,
    error_detail TEXT
);
"""


def _db_path(db_path: str | None) -> str:
    return db_path or os.environ.get("LOGS_DB_PATH", "logs.db")


def init_db(db_path: str | None = None) -> None:
    with sqlite3.connect(_db_path(db_path)) as conn:
        conn.execute(_SCHEMA)


def log_prediction_call(
    records: list[dict[str, Any]],
    status: str,
    output: dict[str, Any] | None = None,
    latency_ms: float | None = None,
    error_detail: str | None = None,
    db_path: str | None = None,
) -> None:
    """Enregistre un appel /predict (succes ou echec) : input, output, latence.

    Donnee clé pour l'analyse en aval (dérive des données, taux d'erreur,
    latence anormale) --> voir src/monitoring.py.
    """
    path = _db_path(db_path)
    with sqlite3.connect(path) as conn:
        conn.execute(_SCHEMA)  # idempotent, evite un init_db() explicite obligatoire
        conn.execute(
            """
            INSERT INTO prediction_logs
                (timestamp, input_json, n_records, output_json, latency_ms, status, error_detail)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(UTC).isoformat(),
                json.dumps(records),
                len(records),
                json.dumps(output) if output is not None else None,
                latency_ms,
                status,
                error_detail,
            ),
        )

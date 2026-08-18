import json
import os
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import psycopg
from psycopg.rows import dict_row

_SCHEMA = """
CREATE TABLE IF NOT EXISTS prediction_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    input_json TEXT NOT NULL,
    n_records INTEGER NOT NULL,
    output_json TEXT,
    latency_ms REAL,
    status TEXT NOT NULL,
    error_detail TEXT
);
"""


def _database_url(database_url: str | None) -> str:
    return database_url or os.environ.get(
        "DATABASE_URL", "postgresql://mlops:mlops@localhost:5432/mlops"
    )


def init_db(database_url: str | None = None) -> None:
    with psycopg.connect(_database_url(database_url)) as conn:
        conn.execute(_SCHEMA)
        conn.commit()


def log_prediction_call(
    records: list[dict[str, Any]],
    status: str,
    output: dict[str, Any] | None = None,
    latency_ms: float | None = None,
    error_detail: str | None = None,
    database_url: str | None = None,
) -> None:
    """Enregistre un appel /predict (succes ou echec) : input, output, latence.
    """
    url = _database_url(database_url)
    with psycopg.connect(url) as conn:
        conn.execute(_SCHEMA)  # idempotent, evite un init_db() explicite obligatoire
        conn.execute(
            """
            INSERT INTO prediction_logs
                (timestamp, input_json, n_records, output_json, latency_ms, status, error_detail)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                datetime.now(UTC),
                json.dumps(records),
                len(records),
                json.dumps(output) if output is not None else None,
                latency_ms,
                status,
                error_detail,
            ),
        )
        conn.commit()


def load_calls_df(database_url: str | None = None, limit: int = 500) -> pd.DataFrame:
    """Charge les derniers appels enregistres sous forme de DataFrame"""
    url = _database_url(database_url)
    with psycopg.connect(url, row_factory=dict_row) as conn:
        conn.execute(_SCHEMA)
        conn.commit()
        cur = conn.execute(
            "SELECT * FROM prediction_logs ORDER BY id DESC LIMIT %s",
            (limit,),
        )
        df = pd.DataFrame(cur.fetchall())
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_rows(database_url: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    """Charge les derniers appels sous forme de liste de dicts"""
    url = _database_url(database_url)
    with psycopg.connect(url, row_factory=dict_row) as conn:
        conn.execute(_SCHEMA)
        conn.commit()
        cur = conn.execute(
            "SELECT input_json, status, latency_ms FROM prediction_logs ORDER BY id DESC LIMIT %s",
            (limit,),
        )
        return cur.fetchall()


def load_successful_inputs(database_url: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    """Charge les input_json des derniers appels réussi"""
    url = _database_url(database_url)
    with psycopg.connect(url) as conn:
        conn.execute(_SCHEMA)
        conn.commit()
        cur = conn.execute(
            "SELECT input_json FROM prediction_logs WHERE status = 'success' ORDER BY id DESC LIMIT %s",
            (limit,),
        )
        return [row[0] for row in cur.fetchall()]

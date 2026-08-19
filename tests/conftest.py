import os

import psycopg
import pytest

os.environ.setdefault("API_KEY", "test-secret-key")

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql://mlops:mlops@localhost:5433/mlops_test"
)


def _ensure_test_database_exists() -> None:
    """Créé la base de test si besoin (en se connectant à la base 'mlops' et pas à la base de test elle-même)"""
    admin_url, db_name = TEST_DATABASE_URL.rsplit("/", 1)
    with psycopg.connect(f"{admin_url}/mlops", autocommit=True) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (db_name,)
        ).fetchone()
        if not exists:
            conn.execute(f'CREATE DATABASE "{db_name}"')


@pytest.fixture
def test_db() -> str:
    """Base Postgres de test, remise à 0 avant chaque test"""
    from storage import init_db

    _ensure_test_database_exists()
    init_db(TEST_DATABASE_URL)
    with psycopg.connect(TEST_DATABASE_URL) as conn:
        conn.execute("TRUNCATE TABLE prediction_logs RESTART IDENTITY")
        conn.commit()
    return TEST_DATABASE_URL

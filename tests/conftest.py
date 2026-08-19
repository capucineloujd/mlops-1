import os

import psycopg
import pytest

os.environ.setdefault("API_KEY", "test-secret-key")

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql://mlops:mlops@localhost:5433/mlops"
)


@pytest.fixture
def test_db() -> str:
    """Base Postgres de test, remise a 0 avant chaque test"""
    from storage import init_db

    init_db(TEST_DATABASE_URL)
    with psycopg.connect(TEST_DATABASE_URL) as conn:
        conn.execute("TRUNCATE TABLE prediction_logs RESTART IDENTITY")
        conn.commit()
    return TEST_DATABASE_URL

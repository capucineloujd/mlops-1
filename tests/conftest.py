import os

import psycopg
import pytest

# Doit etre defini AVANT le premier import de `api` (qui lit API_KEY au chargement
# du module), donc dans conftest.py : pytest le charge avant de collecter les tests.
os.environ.setdefault("API_KEY", "test-secret-key")

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql://mlops:mlops@localhost:5433/mlops"
)


@pytest.fixture
def test_db() -> str:
    """Base Postgres de test, remise a zero avant chaque test (isolation :
    plus de fichier sqlite jetable par test, une vraie base Postgres partagee
    dont on vide la table entre chaque test)."""
    from storage import init_db

    init_db(TEST_DATABASE_URL)
    with psycopg.connect(TEST_DATABASE_URL) as conn:
        conn.execute("TRUNCATE TABLE prediction_logs RESTART IDENTITY")
        conn.commit()
    return TEST_DATABASE_URL

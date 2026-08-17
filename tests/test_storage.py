import json
import sqlite3

from storage import init_db, load_calls_df, log_prediction_call


def test_init_db_cree_la_table(tmp_path):
    db_path = str(tmp_path / "logs.db")

    init_db(db_path)

    with sqlite3.connect(db_path) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='prediction_logs'"
        ).fetchall()
    assert tables


def test_log_prediction_call_succes(tmp_path):
    db_path = str(tmp_path / "logs.db")
    records = [{"AMT_INCOME_TOTAL": 50000}]
    output = {"probabilities": [0.3], "decisions": ["ACCORDE"], "threshold": 0.499}

    log_prediction_call(records, status="success", output=output, latency_ms=12.5, db_path=db_path)

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT input_json, n_records, output_json, latency_ms, status, error_detail FROM prediction_logs"
        ).fetchone()

    assert json.loads(row[0]) == records
    assert row[1] == 1
    assert json.loads(row[2]) == output
    assert row[3] == 12.5
    assert row[4] == "success"
    assert row[5] is None


def test_log_prediction_call_erreur(tmp_path):
    db_path = str(tmp_path / "logs.db")

    log_prediction_call(
        [{"AMT_INCOME_TOTAL": 0}],
        status="error",
        latency_ms=3.1,
        error_detail="revenu hors plage",
        db_path=db_path,
    )

    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT output_json, status, error_detail FROM prediction_logs").fetchone()

    assert row[0] is None
    assert row[1] == "error"
    assert row[2] == "revenu hors plage"


def test_plusieurs_appels_sont_tous_enregistres(tmp_path):
    db_path = str(tmp_path / "logs.db")

    for _ in range(3):
        log_prediction_call([{"x": 1}], status="success", output={}, latency_ms=1.0, db_path=db_path)

    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM prediction_logs").fetchone()[0]

    assert count == 3


def test_log_prediction_call_sans_init_db_prealable(tmp_path):
    # log_prediction_call doit créer la table toute seule si besoin
    db_path = str(tmp_path / "fresh.db")

    log_prediction_call([{"x": 1}], status="success", output={}, latency_ms=1.0, db_path=db_path)

    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM prediction_logs").fetchone()[0]

    assert count == 1


class TestLoadCallsDf:
    def test_renvoie_un_dataframe_vide_sans_appels(self, tmp_path):
        db_path = str(tmp_path / "logs.db")
        init_db(db_path)

        df = load_calls_df(db_path=db_path)

        assert df.empty

    def test_renvoie_les_appels_avec_timestamp_type_datetime(self, tmp_path):
        db_path = str(tmp_path / "logs.db")
        log_prediction_call([{"x": 1}], status="success", output={}, latency_ms=5.0, db_path=db_path)

        df = load_calls_df(db_path=db_path)

        assert len(df) == 1
        assert df.iloc[0]["status"] == "success"
        assert df.iloc[0]["latency_ms"] == 5.0
        assert str(df["timestamp"].dtype).startswith("datetime64")

    def test_respecte_la_limite(self, tmp_path):
        db_path = str(tmp_path / "logs.db")
        for _ in range(10):
            log_prediction_call([{"x": 1}], status="success", output={}, latency_ms=1.0, db_path=db_path)

        df = load_calls_df(db_path=db_path, limit=3)

        assert len(df) == 3

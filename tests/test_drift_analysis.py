import pandas as pd
import pytest

from drift_analysis import (
    load_current_from_logs,
    run_drift_report,
    summarize,
)
from storage import init_db, log_prediction_call


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "logs.db")
    init_db(path)
    return path


@pytest.fixture
def reference_df():
    return pd.DataFrame({"AMT_INCOME_TOTAL": [50000 + i * 100 for i in range(500)]})


class TestLoadCurrentFromLogs:
    def test_ne_recupere_que_les_appels_reussis(self, db_path):
        log_prediction_call([{"AMT_INCOME_TOTAL": 1}], status="success", output={}, latency_ms=1.0, db_path=db_path)
        log_prediction_call(
            [{"AMT_INCOME_TOTAL": 2}], status="error", latency_ms=1.0, error_detail="x", db_path=db_path
        )

        df = load_current_from_logs(db_path=db_path, window=100)

        assert len(df) == 1
        assert df.iloc[0]["AMT_INCOME_TOTAL"] == 1

    def test_filtre_sur_les_colonnes_demandees(self, db_path):
        log_prediction_call(
            [{"AMT_INCOME_TOTAL": 1, "AUTRE_CHAMP": 99}],
            status="success",
            output={},
            latency_ms=1.0,
            db_path=db_path,
        )

        df = load_current_from_logs(db_path=db_path, window=100, columns=["AMT_INCOME_TOTAL"])

        assert list(df.columns) == ["AMT_INCOME_TOTAL"]

    def test_aucun_appel_renvoie_un_dataframe_vide(self, db_path):
        df = load_current_from_logs(db_path=db_path, window=100)

        assert df.empty


class TestRunDriftReportEtSummarize:
    def test_pas_de_derive_sur_des_donnees_identiques(self, reference_df):
        current = reference_df.copy()

        result = run_drift_report(reference_df, current)
        summary = summarize(result)

        assert summary["drifted_columns"] == []
        assert summary["n_drifted_columns"] == 0

    def test_derive_evidente_est_detectee(self, reference_df):
        current = reference_df.copy() * 100  # revenus 100x plus eleves

        result = run_drift_report(reference_df, current)
        summary = summarize(result)

        # la valeur brute depend de la methode choisie par evidently (distance
        # ou p-value, dont le sens de comparaison au seuil s'inverse) : seul
        # le flag agrege drifted_columns est teste ici, pas la valeur brute
        assert "AMT_INCOME_TOTAL" in summary["drifted_columns"]
        assert summary["n_drifted_columns"] == 1

    def test_ne_compare_que_les_colonnes_communes(self, reference_df):
        current = pd.DataFrame({"AMT_INCOME_TOTAL": reference_df["AMT_INCOME_TOTAL"], "AUTRE": range(500)})

        result = run_drift_report(reference_df, current)
        summary = summarize(result)

        assert set(summary["column_drift"].keys()) == {"AMT_INCOME_TOTAL"}

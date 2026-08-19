import json
import pytest
from monitoring import analyze
from storage import log_prediction_call


@pytest.fixture
def reference_path(tmp_path):
    path = str(tmp_path / "reference.json")
    with open(path, "w") as fh:
        json.dump({"AMT_INCOME_TOTAL": [50000 + i * 100 for i in range(500)]}, fh)
    return path


def _log_n_calls(database_url, n, status="success", income=50000):
    for _ in range(n):
        log_prediction_call(
            [{"AMT_INCOME_TOTAL": income}],
            status=status,
            output={"probabilities": [0.1]} if status == "success" else None,
            latency_ms=10.0,
            error_detail=None if status == "success" else "erreur",
            database_url=database_url,
        )


class TestTauxDErreur:
    def test_taux_erreur_bas_ne_declenche_pas_d_alerte(self, test_db, reference_path):
        _log_n_calls(test_db, 9, status="success")
        _log_n_calls(test_db, 1, status="error")

        report = analyze(database_url=test_db, reference_path=reference_path, window=100)

        assert report.error_rate == pytest.approx(0.1)
        assert not any(a.check == "error_rate" for a in report.anomalies)

    def test_taux_erreur_eleve_declenche_une_alerte(self, test_db, reference_path):
        _log_n_calls(test_db, 5, status="success")
        _log_n_calls(test_db, 5, status="error")

        report = analyze(database_url=test_db, reference_path=reference_path, window=100)

        assert report.error_rate == pytest.approx(0.5)
        assert any(a.check == "error_rate" and a.severity == "critical" for a in report.anomalies)


class TestLatence:
    def test_latence_normale_ne_declenche_pas_d_alerte(self, test_db, reference_path):
        for _ in range(5):
            log_prediction_call([{"x": 1}], status="success", output={}, latency_ms=50.0, database_url=test_db)

        report = analyze(database_url=test_db, reference_path=reference_path, window=100)

        assert not any(a.check == "latency" for a in report.anomalies)

    def test_latence_anormale_declenche_une_alerte(self, test_db, reference_path):
        for _ in range(5):
            log_prediction_call([{"x": 1}], status="success", output={}, latency_ms=2000.0, database_url=test_db)

        report = analyze(database_url=test_db, reference_path=reference_path, window=100)

        assert report.latency_p95_ms == 2000.0
        assert any(a.check == "latency" and a.severity == "warning" for a in report.anomalies)


class TestDataDrift:
    def test_donnees_similaires_a_la_reference_ne_declenchent_pas_d_alerte(self, test_db, reference_path):
        for i in range(40):
            log_prediction_call(
                [{"AMT_INCOME_TOTAL": 50000 + i * 1250}],
                status="success",
                output={},
                latency_ms=10.0,
                database_url=test_db,
            )

        report = analyze(database_url=test_db, reference_path=reference_path, window=100)

        assert "AMT_INCOME_TOTAL" in report.drift
        assert not any(a.check == "data_drift" for a in report.anomalies)

    def test_donnees_tres_differentes_de_la_reference_declenchent_une_alerte(self, test_db, reference_path):
        # revenus 100x plus élevés que toute la référence ==> drift obvious
        for _ in range(40):
            log_prediction_call(
                [{"AMT_INCOME_TOTAL": 5_000_000}],
                status="success",
                output={},
                latency_ms=10.0,
                database_url=test_db,
            )

        report = analyze(database_url=test_db, reference_path=reference_path, window=100)

        assert any(a.check == "data_drift" and "AMT_INCOME_TOTAL" in a.message for a in report.anomalies)

    def test_pas_assez_d_echantillons_ne_declenche_pas_le_test(self, test_db, reference_path):
        _log_n_calls(test_db, 5, status="success", income=5_000_000)  # < DRIFT_MIN_SAMPLES

        report = analyze(database_url=test_db, reference_path=reference_path, window=100)

        assert "AMT_INCOME_TOTAL" not in report.drift

    def test_reference_absente_ne_plante_pas(self, test_db, tmp_path):
        _log_n_calls(test_db, 5, status="success")

        report = analyze(database_url=test_db, reference_path=str(tmp_path / "nexiste_pas.json"), window=100)

        assert report.drift == {}


class TestAnalyzeGeneral:
    def test_aucun_appel_ne_plante_pas(self, test_db, reference_path):
        report = analyze(database_url=test_db, reference_path=reference_path, window=100)

        assert report.n_calls_analyzed == 0
        assert report.anomalies == []

    def test_window_limite_le_nombre_d_appels_analyses(self, test_db, reference_path):
        _log_n_calls(test_db, 20, status="success")

        report = analyze(database_url=test_db, reference_path=reference_path, window=5)

        assert report.n_calls_analyzed == 5

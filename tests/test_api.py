import sqlite3
from typing import ClassVar

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import API_KEY, DECISION_THRESHOLD, app, get_model

AUTH_HEADERS = {"X-API-Key": API_KEY}


class FakeModel:
    """Modele factice. Reproduit l'interface reellement utilisee par
    src/api.py depuis le bypass de pandas/sklearn (feature_name_ +
    booster_.predict sur un tableau numpy), cf. profiling."""

    feature_name_: ClassVar[list[str]] = ["EXT_SOURCE_2"]

    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.booster_ = self

    def predict(self, arr):
        assert len(arr) == len(self.probabilities)
        return np.array(self.probabilities)


@pytest.fixture
def client():
    with TestClient(app, headers=AUTH_HEADERS) as c:
        yield c
    app.dependency_overrides.clear()


def override_model(probabilities):
    app.dependency_overrides[get_model] = lambda: FakeModel(probabilities)


class TestStockageDesAppels:
    """Verifie que chaque appel /predict (succes et echec) est bien
    persisté dans logs.db, cf. src/storage.py."""

    @pytest.fixture
    def logged_client(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "logs.db")
        monkeypatch.setenv("LOGS_DB_PATH", db_path)

        with TestClient(app, headers=AUTH_HEADERS) as c:
            yield c, db_path
        app.dependency_overrides.clear()

    def test_appel_reussi_est_enregistre(self, logged_client):
        client, db_path = logged_client
        override_model([0.2])

        client.post("/predict", json={"records": [{"EXT_SOURCE_2": 0.5}]})

        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT status, latency_ms, output_json FROM prediction_logs"
            ).fetchone()

        assert row[0] == "success"
        assert row[1] is not None and row[1] >= 0
        assert row[2] is not None

    def test_appel_en_erreur_est_enregistre(self, logged_client):
        client, db_path = logged_client
        override_model([0.2])

        client.post("/predict", json={"records": [{"AMT_INCOME_TOTAL": 0}]})

        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT status, output_json, error_detail FROM prediction_logs"
            ).fetchone()

        assert row[0] == "error"
        assert row[1] is None
        assert "AMT_INCOME_TOTAL" in row[2]


class TestHealth:
    def test_health_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_health_ne_necessite_pas_de_cle_api(self):
        # /health doit rester accessible sans authentification
        with TestClient(app) as unauth_client:
            response = unauth_client.get("/health")
        assert response.status_code == 200


class TestAuthentification:
    def test_predict_sans_cle_api_renvoie_401(self):
        override_model([0.5])
        with TestClient(app) as unauth_client:
            response = unauth_client.post("/predict", json={"records": [{"EXT_SOURCE_2": 0.5}]})
        app.dependency_overrides.clear()

        assert response.status_code == 401

    def test_predict_avec_mauvaise_cle_api_renvoie_401(self):
        override_model([0.5])
        with TestClient(app, headers={"X-API-Key": "mauvaise-cle"}) as bad_client:
            response = bad_client.post("/predict", json={"records": [{"EXT_SOURCE_2": 0.5}]})
        app.dependency_overrides.clear()

        assert response.status_code == 401

    def test_predict_avec_bonne_cle_api_passe(self, client):
        override_model([0.5])

        response = client.post("/predict", json={"records": [{"EXT_SOURCE_2": 0.5}]})

        assert response.status_code == 200


class TestPredict:
    def test_predict_renvoie_une_proba_par_client(self, client):
        override_model([0.1, 0.9])

        response = client.post(
            "/predict",
            json={"records": [{"EXT_SOURCE_2": 0.5}, {"EXT_SOURCE_2": 0.1}]},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["probabilities"] == [0.1, 0.9]
        assert len(body["decisions"]) == 2

    def test_decision_accorde_sous_le_seuil(self, client):
        override_model([DECISION_THRESHOLD - 0.1])

        response = client.post("/predict", json={"records": [{"EXT_SOURCE_2": 0.5}]})

        assert response.json()["decisions"] == ["ACCORDE"]

    def test_decision_refuse_au_dessus_du_seuil(self, client):
        override_model([DECISION_THRESHOLD + 0.1])

        response = client.post("/predict", json={"records": [{"EXT_SOURCE_2": 0.5}]})

        assert response.json()["decisions"] == ["REFUSE"]

    def test_seuil_expose_dans_la_reponse(self, client):
        override_model([0.5])

        response = client.post("/predict", json={"records": [{"EXT_SOURCE_2": 0.5}]})

        assert response.json()["threshold"] == DECISION_THRESHOLD

    def test_liste_vide_renvoie_liste_vide(self, client):
        override_model([])

        response = client.post("/predict", json={"records": []})

        assert response.status_code == 200
        assert response.json()["probabilities"] == []

    def test_records_manquant_renvoie_422(self, client):
        override_model([])

        response = client.post("/predict", json={})

        assert response.status_code == 422


class TestValidationMetier:
    """Cas critiques demandés par la consigne"""

    def test_revenu_nul_rejete(self, client):
        override_model([0.5])

        response = client.post(
            "/predict", json={"records": [{"AMT_INCOME_TOTAL": 0}]}
        )

        assert response.status_code == 422
        assert "AMT_INCOME_TOTAL" in response.json()["detail"]

    def test_revenu_negatif_rejete(self, client):
        override_model([0.5])

        response = client.post(
            "/predict", json={"records": [{"AMT_INCOME_TOTAL": -1000}]}
        )

        assert response.status_code == 422

    def test_credit_negatif_rejete(self, client):
        override_model([0.5])

        response = client.post("/predict", json={"records": [{"AMT_CREDIT": -500}]})

        assert response.status_code == 422

    def test_age_impossible_rejete(self, client):
        # DAYS_BIRTH doit etre negatif (convention Home Credit)
        override_model([0.5])

        response = client.post("/predict", json={"records": [{"DAYS_BIRTH": 5}]})

        assert response.status_code == 422
        assert "DAYS_BIRTH" in response.json()["detail"]

    def test_nombre_enfants_negatif_rejete(self, client):
        override_model([0.5])

        response = client.post("/predict", json={"records": [{"CNT_CHILDREN": -3}]})

        assert response.status_code == 422

    def test_texte_a_la_place_d_un_chiffre_rejete(self, client):
        override_model([0.5])

        response = client.post(
            "/predict", json={"records": [{"AMT_INCOME_TOTAL": "beaucoup"}]}
        )

        assert response.status_code == 422
        assert "AMT_INCOME_TOTAL" in response.json()["detail"]

    def test_champ_critique_absent_ne_bloque_pas(self, client):
        # les règles metier ne s'appliquent que si le champ est présent
        override_model([0.5])

        response = client.post("/predict", json={"records": [{"EXT_SOURCE_2": 0.5}]})

        assert response.status_code == 200

    def test_deuxieme_enregistrement_invalide_est_detecte(self, client):
        # la validation doit parcourir tous les records, pas seulement le premier
        override_model([0.5, 0.5])

        response = client.post(
            "/predict",
            json={
                "records": [
                    {"AMT_INCOME_TOTAL": 50000},
                    {"AMT_INCOME_TOTAL": 0},
                ]
            },
        )

        assert response.status_code == 422
        assert "Enregistrement 1" in response.json()["detail"]


class TestPredictAvecLeVraiModele:
    """Test d'integration bout-en-bout avec le vrai modele chargé depuis
    Model Registry MLflow."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_model(self):
        try:
            get_model()
        except Exception as exc:
            pytest.skip(f"Modele MLflow indisponible : {exc}")

    def test_predict_avec_un_seul_client(self, client):
        model = get_model()
        record = {name: 0.0 for name in model.feature_name_}
        # 0.0 partout n'est pas metier-plausible pour ces champs: on les corrige pour un client factice mais valide
        record.update(
            {
                "AMT_INCOME_TOTAL": 50000.0,
                "AMT_CREDIT": 100000.0,
                "DAYS_BIRTH": -12000.0,
                "CNT_CHILDREN": 0.0,
            }
        )

        response = client.post("/predict", json={"records": [record]})

        assert response.status_code == 200
        body = response.json()
        assert 0.0 <= body["probabilities"][0] <= 1.0
        assert body["decisions"][0] in {"ACCORDE", "REFUSE"}

    def test_champs_obligatoires_manquants_rejete_par_le_vrai_schema(self, client):
        response = client.post("/predict", json={"records": [{"EXT_SOURCE_2": 0.5}]})

        assert response.status_code == 422

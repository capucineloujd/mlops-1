import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import DECISION_THRESHOLD, app, get_model


class FakeModel:
    """Modele factice : renvoie une proba fixe par ligne, pour tester l'API
    (routing, validation, format de reponse) independamment du vrai modele MLflow."""

    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict(self, df):
        assert len(df) == len(self.probabilities)
        return np.array(self.probabilities)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def override_model(probabilities):
    app.dependency_overrides[get_model] = lambda: FakeModel(probabilities)


class TestHealth:
    def test_health_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


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
        # override du modele : ce test verifie la validation du body,
        # pas le chargement du modele (FastAPI resout les Depends meme
        # quand le body est invalide, donc get_model() serait quand meme
        # appele si on ne le mockait pas)
        override_model([])

        response = client.post("/predict", json={})

        assert response.status_code == 422


class TestPredictAvecLeVraiModele:
    """Test d'integration bout-en-bout avec le vrai modele charge depuis
    Model Registry MLflow."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_model(self):
        try:
            get_model()
        except Exception as exc:
            pytest.skip(f"Modele MLflow indisponible : {exc}")

    def test_predict_avec_un_seul_client(self, client):
        model = get_model()
        schema = model.metadata.get_input_schema()
        if schema is None:
            pytest.skip("Le modele enregistre n'expose pas de signature d'entree")

        record = {
            col.name: False if str(col.type) == "DataType.boolean" else 0.0
            for col in schema.inputs
        }

        response = client.post("/predict", json={"records": [record]})

        assert response.status_code == 200
        body = response.json()
        assert 0.0 <= body["probabilities"][0] <= 1.0
        assert body["decisions"][0] in {"ACCORDE", "REFUSE"}

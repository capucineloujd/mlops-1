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


class TestValidationMetier:
    """Cas critiques demandes par la consigne : champs obligatoires manquants,
    valeurs hors plage, types incorrects. Utilise le FakeModel pour tester la
    validation elle-meme, independamment du vrai modele MLflow."""

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
        # DAYS_BIRTH doit etre negatif (convention Home Credit : nombre de
        # jours avant la demande de credit). Une valeur positive ou nulle
        # correspondrait a un client pas encore ne.
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
        # les regles metier ne s'appliquent que si le champ est present :
        # un record qui ne contient pas ces champs n'est pas rejete a ce
        # niveau (le schema complet reste valide par MLflow au moment du
        # predict, hors scope de ce test avec le FakeModel)
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
        # 0.0 partout n'est pas metier-plausible pour ces champs (cf.
        # TestValidationMetier) : on les corrige pour un client factice mais valide
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
        # contre le vrai schema de production (710 colonnes attendues) : un
        # record quasiment vide doit etre rejete par l'enforcement de schema
        # MLflow, pas silencieusement accepte avec des colonnes manquantes
        response = client.post("/predict", json={"records": [{"EXT_SOURCE_2": 0.5}]})

        assert response.status_code == 422

import os
from functools import lru_cache
from typing import Any

import mlflow.pyfunc
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from mlflow.exceptions import MlflowException
from pydantic import BaseModel

MODEL_URI = os.environ.get("MODEL_URI", "models:/lightgbm-credit-scoring-serving@gagnant")
DECISION_THRESHOLD = float(os.environ.get("DECISION_THRESHOLD", "0.499"))

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

app = FastAPI(
    title="Home Credit - Scoring API",
    description="Expose le modele de scoring credit (LightGBM, optimise Optuna) via une API REST.",
    version="1.0.0",
)


class ErrorResponse(BaseModel):
    detail: str


@lru_cache
def _load_model() -> mlflow.pyfunc.PyFuncModel:
    return mlflow.pyfunc.load_model(MODEL_URI)


def get_model() -> mlflow.pyfunc.PyFuncModel:
    """Charge le modele pyfunc depuis le Model Registry MLflow (une seule fois, mis en cache)."""
    try:
        return _load_model()
    except MlflowException as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Modele indisponible dans le Model Registry MLflow : {exc.message}",
        ) from exc


# Regles de plausibilite metier pour les champs les plus critiques, verifiees
# uniquement si le champ est present dans le record (le schema complet, lui,
# est valide par MLflow au moment de l'inference). Objectif : rejeter tot les
# entrees absurdes (age negatif, revenu nul, texte a la place d'un chiffre)
# plutot que de laisser le modele produire une prediction sur du n'importe quoi.
_BUSINESS_RULES: dict[str, tuple[str, Any]] = {
    "AMT_INCOME_TOTAL": (">", 0),  # revenu strictement positif
    "AMT_CREDIT": (">", 0),  # montant du credit strictement positif
    "DAYS_BIRTH": ("<", 0),  # convention Home Credit : jours negatifs (age > 0)
    "CNT_CHILDREN": (">=", 0),  # nombre d'enfants positif ou nul
}


def _validate_business_rules(records: list[dict[str, Any]]) -> None:
    for i, record in enumerate(records):
        for field, (op, bound) in _BUSINESS_RULES.items():
            if field not in record:
                continue
            value = record[field]

            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise HTTPException(
                    status_code=422,
                    detail=f"Enregistrement {i} : '{field}' doit etre numerique, recu {value!r}",
                )

            valid = {
                ">": value > bound,
                ">=": value >= bound,
                "<": value < bound,
            }[op]

            if not valid:
                raise HTTPException(
                    status_code=422,
                    detail=f"Enregistrement {i} : '{field}'={value} hors de la plage attendue (doit etre {op} {bound})",
                )


class PredictRequest(BaseModel):
    records: list[dict[str, Any]]


class PredictResponse(BaseModel):
    probabilities: list[float]
    decisions: list[str]
    threshold: float


@app.get("/health", summary="Verifie que l'API est en ligne")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Calcule la probabilite de defaut d'un ou plusieurs clients",
    responses={
        422: {"model": ErrorResponse, "description": "Donnees d'entree invalides (colonnes manquantes ou mal typees)"},
        503: {"model": ErrorResponse, "description": "Modele indisponible (Model Registry MLflow inaccessible)"},
    },
)
def predict(
    request: PredictRequest, model: mlflow.pyfunc.PyFuncModel = Depends(get_model)
) -> PredictResponse:
    _validate_business_rules(request.records)

    df = pd.DataFrame(request.records)

    try:
        probabilities = list(model.predict(df))
    except MlflowException as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Donnees d'entree invalides pour le modele : {exc.message}",
        ) from exc

    decisions = ["REFUSE" if p > DECISION_THRESHOLD else "ACCORDE" for p in probabilities]

    return PredictResponse(
        probabilities=[float(p) for p in probabilities],
        decisions=decisions,
        threshold=DECISION_THRESHOLD,
    )

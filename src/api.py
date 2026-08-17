import json
import logging
import os
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any

import mlflow.lightgbm
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from lightgbm import LGBMClassifier
from lightgbm.basic import LightGBMError
from mlflow.exceptions import MlflowException
from pydantic import BaseModel

from storage import init_db, log_prediction_call

# Modele LightGBM natif charge directement (mlflow.lightgbm), pas via le
# wrapper mlflow.pyfunc : profiling (src/profile_inference.py) a montre que
# le wrapper pyfunc ajoutait ~77% de temps de predict en pur overhead
# (enforcement de schema, indirection Python) par rapport au calcul reel du
# modele. Voir la section "Optimisation des performances" du README.
MODEL_URI = os.environ.get("MODEL_URI", "models:/lightgbm-credit-scoring@gagnant")
DECISION_THRESHOLD = float(os.environ.get("DECISION_THRESHOLD", "0.499"))

logger = logging.getLogger("scoring_api")
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)


def _log(event: str, **fields: Any) -> None:
    logger.info(json.dumps({"event": event, **fields}))


API_KEY = os.environ.get("API_KEY")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(provided_key: str | None = Security(_api_key_header)) -> None:
    if not API_KEY:
        raise HTTPException(
            status_code=500,
            detail="API_KEY n'est pas configuree cote serveur : l'API ne peut pas etre securisee.",
        )
    if provided_key != API_KEY:
        # jamais la cle recue dans les logs, seulement le fait qu'elle est invalide
        _log("auth_failed", reason="missing_or_invalid_key")
        raise HTTPException(status_code=401, detail="Cle API manquante ou invalide (header X-API-Key)")

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))


@lru_cache
def _load_model() -> LGBMClassifier:
    return mlflow.lightgbm.load_model(MODEL_URI)


@asynccontextmanager
async def lifespan(_: FastAPI):

    init_db()
    try:
        _load_model()
        _log("model_loaded", model_uri=MODEL_URI)
    except MlflowException as exc:
        _log("model_load_failed", model_uri=MODEL_URI, reason=str(exc))
    yield


app = FastAPI(
    title="Home Credit - Scoring API",
    description="Expose le modele de scoring credit (LightGBM, optimise Optuna) via une API REST.",
    version="1.0.0",
    lifespan=lifespan,
)


class ErrorResponse(BaseModel):
    detail: str


def get_model() -> LGBMClassifier:
    """Charge le modele LightGBM natif depuis le Model Registry MLflow (une seule fois, mis en cache)."""
    try:
        return _load_model()
    except MlflowException as exc:
        _log("model_unavailable", model_uri=MODEL_URI, reason=str(exc))
        raise HTTPException(
            status_code=503,
            detail=f"Modele indisponible dans le Model Registry MLflow : {exc.message}",
        ) from exc


_BUSINESS_RULES: dict[str, tuple[str, Any]] = {
    "AMT_INCOME_TOTAL": (">", 0),  # revenu strictement positif
    "AMT_CREDIT": (">", 0),  # montant du credit strictement positif
    "DAYS_BIRTH": ("<", 0),  # convention Home Credit : jours negatifs
    "CNT_CHILDREN": (">=", 0),  # nombre d'enfants positif ou nul
}


def _validate_business_rules(records: list[dict[str, Any]]) -> None:
    for i, record in enumerate(records):
        for field, (op, bound) in _BUSINESS_RULES.items():
            if field not in record:
                continue
            value = record[field]

            if isinstance(value, bool) or not isinstance(value, (int, float)):
                _log("validation_rejected", field=field, index=i, reason="wrong_type")
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
                _log("validation_rejected", field=field, index=i, reason="out_of_range")
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
    dependencies=[Depends(require_api_key)],
    responses={
        401: {"model": ErrorResponse, "description": "Cle API manquante ou invalide"},
        422: {"model": ErrorResponse, "description": "Donnees d'entree invalides (colonnes manquantes ou mal typees)"},
        503: {"model": ErrorResponse, "description": "Modele indisponible (Model Registry MLflow inaccessible)"},
    },
)
def predict(
    request: PredictRequest, model: LGBMClassifier = Depends(get_model)
) -> PredictResponse:
    start = time.perf_counter()

    try:
        _validate_business_rules(request.records)

        df = pd.DataFrame(request.records)

        try:
            probabilities = list(model.predict_proba(df)[:, 1])
        except (LightGBMError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Donnees d'entree invalides pour le modele : {exc}",
            ) from exc

        decisions = ["REFUSE" if p > DECISION_THRESHOLD else "ACCORDE" for p in probabilities]
        response = PredictResponse(
            probabilities=[float(p) for p in probabilities],
            decisions=decisions,
            threshold=DECISION_THRESHOLD,
        )
    except HTTPException as exc:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        _log("prediction_failed", n_records=len(request.records), latency_ms=latency_ms, reason=str(exc.detail))
        log_prediction_call(
            request.records,
            status="error",
            latency_ms=latency_ms,
            error_detail=str(exc.detail),
        )
        raise

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    _log(
        "prediction",
        n_records=len(request.records),
        n_accorde=decisions.count("ACCORDE"),
        n_refuse=decisions.count("REFUSE"),
        latency_ms=latency_ms,
    )
    log_prediction_call(
        request.records,
        status="success",
        output=response.model_dump(),
        latency_ms=latency_ms,
    )
    return response

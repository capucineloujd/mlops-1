import json
import logging
import os
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any

import mlflow.lightgbm
import numpy as np
import psycopg
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from lightgbm import LGBMClassifier
from lightgbm.basic import LightGBMError
from mlflow.exceptions import MlflowException
from pydantic import BaseModel

from storage import init_db, log_prediction_call


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
        # jamais la clef recue dans les logs, seulement le fait qu'elle est invalide
        _log("auth_failed", reason="missing_or_invalid_key")
        raise HTTPException(status_code=401, detail="Cle API manquante ou invalide (header X-API-Key)")

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))


@lru_cache
def _load_model() -> LGBMClassifier:
    return mlflow.lightgbm.load_model(MODEL_URI)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # best-effort : si la base Postgres ou le Model Registry sont indisponibles au demarrage, l'API demarre quand même
    try:
        init_db()
    except psycopg.OperationalError as exc:
        _log("storage_unavailable", reason=str(exc))

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
    """Charge le modèle LightGBM natif depuis le Model Registry MLflow (une seule fois, mis en cache)."""
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


def _safe_log_prediction_call(*args: Any, **kwargs: Any) -> None:
    # une base Postgres momentanément injoignable ne doit pas faire échouer une réponse /predict déjà calculeé avec succès
    try:
        log_prediction_call(*args, **kwargs)
    except psycopg.OperationalError as exc:
        _log("storage_unavailable", reason=str(exc))


def _build_feature_array(records: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    """Construit le tableau numpy attendu par le Booster, dans l'ordre exact des features d'entrainement."""
    if not records:
        return np.empty((0, len(feature_names)), dtype=np.float64)
    try:
        return np.array(
            [[record[name] for name in feature_names] for record in records],
            dtype=np.float64,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=422, detail=f"Champ manquant dans un enregistrement : {exc}"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail=f"Donnees invalides pour le modele : {exc}"
        ) from exc


class PredictRequest(BaseModel):
    records: list[dict[str, Any]]


class PredictResponse(BaseModel):
    probabilities: list[float]
    decisions: list[str]
    threshold: float


@app.get("/health", summary="Vérifie que l'API est en ligne")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Calcule la probabilite de défault d'un ou plusieurs clients",
    dependencies=[Depends(require_api_key)],
    responses={
        401: {"model": ErrorResponse, "description": "Clef API manquante ou invalide"},
        422: {"model": ErrorResponse, "description": "Données d'entrée invalides (colonnes manquantes ou mal typées)"},
        503: {"model": ErrorResponse, "description": "Modele indisponible (Model Registry MLflow inaccessible)"},
    },
)
def predict(
    request: PredictRequest, model: LGBMClassifier = Depends(get_model)
) -> PredictResponse:
    start = time.perf_counter()

    try:
        _validate_business_rules(request.records)

        feature_array = _build_feature_array(request.records, model.feature_name_)

        if len(request.records) == 0:
            probabilities: list[float] = []
        else:
            try:
                probabilities = list(model.booster_.predict(feature_array))
            except LightGBMError as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"Données d'entrée invalides pour le modèle : {exc}",
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
        _safe_log_prediction_call(
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
    _safe_log_prediction_call(
        request.records,
        status="success",
        output=response.model_dump(),
        latency_ms=latency_ms,
    )
    return response

import os
from functools import lru_cache
from typing import Any

import mlflow.pyfunc
import pandas as pd
from fastapi import Depends, FastAPI
from pydantic import BaseModel

MODEL_URI = os.environ.get("MODEL_URI", "models:/lightgbm-credit-scoring-serving@gagnant")
DECISION_THRESHOLD = float(os.environ.get("DECISION_THRESHOLD", "0.499"))

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

app = FastAPI(title="Home Credit - Scoring API")


@lru_cache
def get_model() -> mlflow.pyfunc.PyFuncModel:
    """Charge le modele pyfunc depuis le Model Registry MLflow (une seule fois, mis en cache)."""
    return mlflow.pyfunc.load_model(MODEL_URI)


class PredictRequest(BaseModel):
    records: list[dict[str, Any]]


class PredictResponse(BaseModel):
    probabilities: list[float]
    decisions: list[str]
    threshold: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(
    request: PredictRequest, model: mlflow.pyfunc.PyFuncModel = Depends(get_model)
) -> PredictResponse:
    df = pd.DataFrame(request.records)
    probabilities = list(model.predict(df))

    decisions = ["REFUSE" if p > DECISION_THRESHOLD else "ACCORDE" for p in probabilities]

    return PredictResponse(
        probabilities=[float(p) for p in probabilities],
        decisions=decisions,
        threshold=DECISION_THRESHOLD,
    )

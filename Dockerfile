FROM python:3.11-slim

# uv : gestion des dépendances (voir pyproject.toml / uv.lock)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Dépendances d'abord pour profiter du cache Docker
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Code source (le modèle final est chargé depuis le Model Registry MLflow)
COPY src/ ./src/

ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db
ENV MODEL_URI=models:/lightgbm-credit-scoring-serving@gagnant

EXPOSE 5001

CMD uv run mlflow models serve \
    -m "$MODEL_URI" \
    --host 0.0.0.0 \
    --port 5001 \
    --env-manager local

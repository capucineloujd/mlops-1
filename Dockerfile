FROM python:3.11-slim

# libgomp1 : requis par LightGBM (chargé via le modele pyfunc au runtime)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

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
ENV DECISION_THRESHOLD=0.499

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]

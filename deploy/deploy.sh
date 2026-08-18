#!/usr/bin/env bash

set -euo pipefail

IMAGE_NAME="mlops1-api"
CONTAINER_NAME="mlops1-api-deployed"
PORT="${PORT:-8000}"
HEALTH_URL="http://localhost:${PORT}/health"
MAX_RETRIES=20

if [ -z "${API_KEY:-}" ]; then
  API_KEY="$(openssl rand -hex 16)"
  echo "[info] API_KEY non fournie : cle ephemere generee pour ce deploiement."
fi

cd "$(dirname "$0")/.."

echo "[1/4] Build de l'image ${IMAGE_NAME}"
docker build -t "${IMAGE_NAME}" .

echo "[2/4] Nettoyage d'un deploiement precedent (le cas echeant)"
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

echo "[3/4] Lancement du conteneur sur l'environnement cible (port ${PORT})"
docker run -d \
  --name "${CONTAINER_NAME}" \
  -p "${PORT}:8000" \
  -e "API_KEY=${API_KEY}" \
  "${IMAGE_NAME}"

echo "[4/4] Verification de sante (${HEALTH_URL})"
for i in $(seq 1 "${MAX_RETRIES}"); do
  if curl -sf "${HEALTH_URL}" >/dev/null 2>&1; then
    echo "Deploiement reussi : ${HEALTH_URL} repond."
    exit 0
  fi
  sleep 2
done

echo "Echec du deploiement : ${HEALTH_URL} ne repond pas apres $((MAX_RETRIES * 2))s."
echo "--- Logs du conteneur ---"
docker logs "${CONTAINER_NAME}"
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
exit 1

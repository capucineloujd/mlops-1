import os
import statistics
import time

import mlflow.lightgbm
import requests

API_URL = os.environ.get("API_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY")
MODEL_URI = os.environ.get("MODEL_URI", "models:/lightgbm-credit-scoring@gagnant")
N_RUNS = 100
N_WARMUP = 10

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))


def build_sample_record() -> dict:
    model = mlflow.lightgbm.load_model(MODEL_URI)
    record = {name: 0.0 for name in model.feature_name_}
    record.update(
        {
            "AMT_INCOME_TOTAL": 150000.0,
            "AMT_CREDIT": 500000.0,
            "DAYS_BIRTH": -15000.0,
            "CNT_CHILDREN": 0.0,
        }
    )
    return record


if __name__ == "__main__":
    if not API_KEY:
        raise SystemExit("API_KEY doit etre definie (meme cle que celle utilisee par l'API en cours d'execution)")

    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
    payload = {"records": [build_sample_record()]}

    print(f"Warmup ({N_WARMUP} appels)...")
    for _ in range(N_WARMUP):
        requests.post(f"{API_URL}/predict", headers=headers, json=payload)

    print(f"Benchmark ({N_RUNS} appels)...")
    durations = []
    for _ in range(N_RUNS):
        start = time.perf_counter()
        response = requests.post(f"{API_URL}/predict", headers=headers, json=payload)
        durations.append((time.perf_counter() - start) * 1000)
        response.raise_for_status()

    durations.sort()
    print(f"\nmean   : {statistics.mean(durations):.2f}ms")
    print(f"median : {durations[len(durations) // 2]:.2f}ms")
    print(f"p95    : {durations[int(len(durations) * 0.95)]:.2f}ms")

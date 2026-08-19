import json
import os
import random
import urllib.error
import urllib.request

import mlflow.lightgbm

API_URL = os.environ.get("API_URL", "http://localhost:8000/predict")
API_KEY = os.environ.get("API_KEY")
MODEL_URI = os.environ.get("MODEL_URI", "models:/lightgbm-credit-scoring@gagnant")
N_REQUESTS = int(os.environ.get("N_REQUESTS", "30"))

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))


def build_base_record() -> dict:
    model = mlflow.lightgbm.load_model(MODEL_URI)
    return {name: 0.0 for name in model.feature_name_}


def build_random_record(base: dict, seed: int) -> dict:
    random.seed(seed)
    record = dict(base)
    record["AMT_INCOME_TOTAL"] = round(random.gauss(180000, 60000), 1)
    record["AMT_CREDIT"] = round(random.gauss(600000, 200000), 1)
    record["DAYS_BIRTH"] = -round(random.uniform(8000, 22000))
    record["CNT_CHILDREN"] = random.choice([0, 0, 1, 2, 3])
    record["EXT_SOURCE_2"] = round(random.uniform(0, 1), 4)
    record["EXT_SOURCE_3"] = round(random.uniform(0, 1), 4)
    return record


def send_predict(record: dict) -> None:
    payload = json.dumps({"records": [record]}).encode()
    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        resp.read()


if __name__ == "__main__":
    if not API_KEY:
        raise SystemExit("API_KEY doit etre definie (meme cle que celle utilisee par l'API en cours d'execution)")

    base_record = build_base_record()
    print(f"Envoi de {N_REQUESTS} appels /predict factices vers {API_URL}...")

    n_failed = 0
    for i in range(N_REQUESTS):
        record = build_random_record(base_record, seed=1000 + i)
        try:
            send_predict(record)
        except (urllib.error.URLError, TimeoutError) as exc:
            n_failed += 1
            print(f"  [{i}] echec : {exc}")

    n_ok = N_REQUESTS - n_failed
    print(f"Termine : {n_ok}/{N_REQUESTS} appels reussis.")

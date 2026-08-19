import json
import os
import random
import re
import urllib.error
import urllib.request

import pandas as pd

API_URL = os.environ.get("API_URL", "http://localhost:8000/predict")
API_KEY = os.environ.get("API_KEY")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "test_engineered.csv")

BATCH_PLAN = [1] * 10 + [3] * 4 + [5] * 2 + [10] * 1
random.seed(42)
random.shuffle(BATCH_PLAN)


def _sanitize_column_name(name: str) -> str:
    return re.sub(r"[ ,:]", "_", name)


def load_candidate_records(n: int) -> list[dict]:
    """Tire n observations réelles du jeu de test, prêtes pour /predict."""
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["SK_ID_CURR", "TARGET"], errors="ignore")
    df = df.rename(columns=_sanitize_column_name)
    df = df.fillna(0.0)
    sample = df.sample(n=n, random_state=42)
    return sample.to_dict(orient="records")


def send_predict(records: list[dict]) -> tuple[int, str]:
    payload = json.dumps({"records": records}).encode()
    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


if __name__ == "__main__":
    if not API_KEY:
        raise SystemExit("API_KEY doit être définie (même clé que celle utilisée par l'API en cours d'exécution)")

    total_records = sum(BATCH_PLAN)
    print(f"Plan : {len(BATCH_PLAN)} appels /predict, {total_records} observations réelles au total.")
    print(f"Tailles de batch utilisées : {sorted(set(BATCH_PLAN))}")

    pool = load_candidate_records(total_records)
    cursor = 0

    n_ok, n_failed = 0, 0
    for i, batch_size in enumerate(BATCH_PLAN):
        batch = pool[cursor : cursor + batch_size]
        cursor += batch_size

        status, body = send_predict(batch)
        if status == 200:
            n_ok += 1
            print(f"  [{i}] n_records={batch_size} -> OK ({status})")
        else:
            n_failed += 1
            print(f"  [{i}] n_records={batch_size} -> ECHEC ({status}) : {body[:200]}")

    print(f"Terminéé : {n_ok}/{len(BATCH_PLAN)} appels reussis, {n_failed} echoues.")
    print("Chaque appel a créé une ligne dans prediction_logs via le fonctionnement normal de l'API.")

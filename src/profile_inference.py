"""Profiling de /predict : decoupe le chemin d'inference en etapes
chronometrees separement, pour identifier le vrai goulot d'etranglement
avant de choisir une optimisation.

Usage : uv run python src/profile_inference.py
"""

import json
import os
import sqlite3
import time

import mlflow.lightgbm
import mlflow.pyfunc
import pandas as pd

from monitoring import analyze

MODEL_URI = os.environ.get("MODEL_URI", "models:/lightgbm-credit-scoring-serving@gagnant")
N_RUNS = 200
N_WARMUP = 20

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))


def print_monitoring_baseline() -> None:
    """Point de depart methodologique : que dit le monitoring reel (logs.db)
    avant de se lancer dans le profiling de code ? Sans ca, on profile a
    l'aveugle sur une hypothese non fondee sur des donnees de production."""
    report = analyze(window=200)
    print("=== Baseline monitoring (logs.db, donnees reelles) ===")
    if report.n_calls_analyzed == 0:
        print("Aucun appel enregistre dans logs.db pour l'instant.")
        print("(genere du trafic via l'API avant de lancer ce script pour une baseline reelle)")
    else:
        print(f"Appels analyses : {report.n_calls_analyzed}")
        print(f"Latence moyenne  : {report.latency_mean_ms}ms | p95 : {report.latency_p95_ms}ms")
        if report.anomalies:
            for a in report.anomalies:
                print(f"  [{a.severity.upper()}] {a.check} : {a.message}")
    print()


def load_real_record(db_path: str | None = None) -> dict | None:
    """Recupere l'input JSON d'un vrai appel /predict reussi depuis logs.db,
    pour profiler sur une requete authentique plutot qu'un exemple invente."""
    path = db_path or os.environ.get("LOGS_DB_PATH", "logs.db")
    if not os.path.exists(path):
        return None
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "SELECT input_json FROM prediction_logs WHERE status = 'success' ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if row is None:
        return None
    records = json.loads(row[0])
    return records[0] if records else None


def _time_stage(fn, n_runs: int = N_RUNS, n_warmup: int = N_WARMUP) -> dict[str, float]:
    for _ in range(n_warmup):
        fn()

    durations = []
    for _ in range(n_runs):
        start = time.perf_counter()
        fn()
        durations.append((time.perf_counter() - start) * 1000)

    durations.sort()
    return {
        "mean_ms": round(sum(durations) / len(durations), 4),
        "median_ms": round(durations[len(durations) // 2], 4),
        "p95_ms": round(durations[int(len(durations) * 0.95)], 4),
    }


def build_sample_record(schema) -> dict:
    record = {c.name: (False if str(c.type) == "DataType.boolean" else 0.0) for c in schema.inputs}
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
    print_monitoring_baseline()

    print(f"Chargement du modele pyfunc ({MODEL_URI})...")
    pyfunc_model = mlflow.pyfunc.load_model(MODEL_URI)
    schema = pyfunc_model.metadata.get_input_schema()

    # modele LightGBM natif, sans passer par le wrapper pyfunc/mlflow
    raw_lgbm_model = pyfunc_model.unwrap_python_model().model
    schema_names = [c.name for c in schema.inputs]
    boolean_names = {c.name for c in schema.inputs if str(c.type) == "DataType.boolean"}

    real_record = load_real_record()
    if real_record is not None:
        print("Profiling sur un enregistrement REEL tire de logs.db (dernier appel reussi).\n")
        # le record reel est enregistre avec les noms sanitizes (feature_name_,
        # cf. api.py) ; le schema mlflow.pyfunc garde les noms originaux (espaces).
        # Meme ordre de colonnes des deux cotes (verifie), donc mapping positionnel.
        # L'API stocke tout en float64 (cf. numpy) ; mlflow.pyfunc exige un vrai
        # bool Python pour les colonnes booleennes du schema.
        record_native = real_record
        record_pyfunc = {}
        for schema_name, native_name in zip(schema_names, raw_lgbm_model.feature_name_):
            value = real_record[native_name]
            record_pyfunc[schema_name] = bool(value) if schema_name in boolean_names else value
    else:
        print("Aucun appel reussi trouve dans logs.db : profiling sur un enregistrement synthetique.\n")
        record_pyfunc = build_sample_record(schema)
        record_native = dict(zip(raw_lgbm_model.feature_name_, record_pyfunc.values()))

    df_pyfunc = pd.DataFrame([record_pyfunc])
    df_native = pd.DataFrame([record_native])

    print(f"\nProfiling sur {N_RUNS} appels (apres {N_WARMUP} warmup)...\n")

    results = {
        "Construction du DataFrame (pd.DataFrame(records))": _time_stage(
            lambda: pd.DataFrame([record_native])
        ),
        "Predict via wrapper pyfunc (chemin de prod actuel)": _time_stage(
            lambda: pyfunc_model.predict(df_pyfunc)
        ),
        "Predict LightGBM natif (sans mlflow pyfunc)": _time_stage(
            lambda: raw_lgbm_model.predict_proba(df_native)
        ),
    }

    print(f"{'Etape':<55} {'mean':>10} {'median':>10} {'p95':>10}")
    print("-" * 90)
    for name, stats in results.items():
        print(f"{name:<55} {stats['mean_ms']:>8.3f}ms {stats['median_ms']:>8.3f}ms {stats['p95_ms']:>8.3f}ms")

    pyfunc_mean = results["Predict via wrapper pyfunc (chemin de prod actuel)"]["mean_ms"]
    raw_mean = results["Predict LightGBM natif (sans mlflow pyfunc)"]["mean_ms"]
    overhead_pct = round((pyfunc_mean - raw_mean) / pyfunc_mean * 100, 1)
    print(f"\nOverhead du wrapper mlflow pyfunc : {overhead_pct}% du temps de predict total")

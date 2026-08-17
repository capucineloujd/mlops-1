"""Profiling de /predict : decoupe le chemin d'inference en etapes
chronometrees separement, pour identifier le vrai goulot d'etranglement
avant de choisir une optimisation.

Usage : uv run python src/profile_inference.py
"""

import os
import time

import mlflow.lightgbm
import mlflow.pyfunc
import pandas as pd

MODEL_URI = os.environ.get("MODEL_URI", "models:/lightgbm-credit-scoring-serving@gagnant")
N_RUNS = 200
N_WARMUP = 20

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))


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
    print(f"Chargement du modele pyfunc ({MODEL_URI})...")
    pyfunc_model = mlflow.pyfunc.load_model(MODEL_URI)
    schema = pyfunc_model.metadata.get_input_schema()
    record = build_sample_record(schema)

    # modele LightGBM natif, sans passer par le wrapper pyfunc/mlflow
    raw_lgbm_model = pyfunc_model.unwrap_python_model().model

    df_single_row = pd.DataFrame([record])

    print(f"\nProfiling sur {N_RUNS} appels (apres {N_WARMUP} warmup)...\n")

    results = {
        "Construction du DataFrame (pd.DataFrame(records))": _time_stage(
            lambda: pd.DataFrame([record])
        ),
        "Predict via wrapper pyfunc (chemin de prod actuel)": _time_stage(
            lambda: pyfunc_model.predict(df_single_row)
        ),
        "Predict LightGBM natif (sans mlflow pyfunc)": _time_stage(
            lambda: raw_lgbm_model.predict_proba(df_single_row)
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

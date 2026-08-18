import cProfile
import io
import os
import pstats

import mlflow.pyfunc
import pandas as pd

MODEL_URI_PYFUNC = os.environ.get("MODEL_URI_PYFUNC", "models:/lightgbm-credit-scoring-serving@gagnant")
N_RUNS = 200
TOP_N = 15

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))


def build_sample_record(feature_names: list[str], boolean_names: set[str] | None = None) -> dict:
    boolean_names = boolean_names or set()
    record = {name: (False if name in boolean_names else 0.0) for name in feature_names}
    record.update(
        {
            "AMT_INCOME_TOTAL": 150000.0,
            "AMT_CREDIT": 500000.0,
            "DAYS_BIRTH": -15000.0,
            "CNT_CHILDREN": 0.0,
        }
    )
    return record


def profile_calls(fn, n_runs: int, label: str) -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(n_runs):
        fn()
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(TOP_N)

    print(f"\n{'=' * 90}\n{label} ({n_runs} appels)\n{'=' * 90}")
    print(stream.getvalue())


if __name__ == "__main__":
    print("Chargement du modèle...")
    pyfunc_model = mlflow.pyfunc.load_model(MODEL_URI_PYFUNC)
    native_model = pyfunc_model.unwrap_python_model().model

    schema_inputs = pyfunc_model.metadata.get_input_schema().inputs
    schema_names = [c.name for c in schema_inputs]
    boolean_names = {c.name for c in schema_inputs if str(c.type) == "DataType.boolean"}
    df_pyfunc = pd.DataFrame([build_sample_record(schema_names, boolean_names)])
    df_native = pd.DataFrame([build_sample_record(native_model.feature_name_)])

    # warmup
    for _ in range(10):
        pyfunc_model.predict(df_pyfunc)
        native_model.predict_proba(df_native)

    profile_calls(lambda: pyfunc_model.predict(df_pyfunc), N_RUNS, "Predict via wrapper mlflow.pyfunc")
    profile_calls(lambda: native_model.predict_proba(df_native), N_RUNS, "Predict LightGBM natifs")

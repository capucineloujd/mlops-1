import os
import time

import mlflow.pyfunc
import numpy as np
import onnxruntime as ort
from onnxmltools.convert import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType

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


if __name__ == "__main__":
    print(f"Chargement du modele natif ({MODEL_URI})...")
    pyfunc_model = mlflow.pyfunc.load_model(MODEL_URI)
    native_model = pyfunc_model.unwrap_python_model().model
    n_features = len(native_model.feature_name_)

    print("Conversion ONNX...")
    initial_types = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_lightgbm(native_model, initial_types=initial_types, target_opset=15)
    onnx_bytes = onnx_model.SerializeToString()
    print(f"Modele ONNX : {len(onnx_bytes) / 1024:.1f} Ko")

    session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])

    array_f64 = np.zeros((1, n_features), dtype=np.float64)
    array_f32 = array_f64.astype(np.float32)

    # verification de precision AVANT de comparer la latence 
    proba_native = native_model.booster_.predict(array_f64)[0]
    onnx_output = session.run(None, {"input": array_f32})
    proba_onnx = onnx_output[1][0][1]  # dict {classe: proba} pour la ligne 0
    diff = abs(proba_native - proba_onnx)
    print(f"\nProba native : {proba_native:.6f} | Proba ONNX : {proba_onnx:.6f} | ecart : {diff:.2e}")
    print("(ecart attendu : precision float32 d'ONNX vs float64 natif, pas un bug)\n")

    results = {
        "Predict LightGBM natif (booster_.predict, chemin de prod actuel)": _time_stage(
            lambda: native_model.booster_.predict(array_f64)
        ),
        "Predict ONNX Runtime": _time_stage(
            lambda: session.run(None, {"input": array_f32})
        ),
    }

    print(f"{'Etape':<65} {'mean':>10} {'median':>10} {'p95':>10}")
    print("-" * 100)
    for name, stats in results.items():
        print(f"{name:<65} {stats['mean_ms']:>8.4f}ms {stats['median_ms']:>8.4f}ms {stats['p95_ms']:>8.4f}ms")

    native_mean = results["Predict LightGBM natif (booster_.predict, chemin de prod actuel)"]["mean_ms"]
    onnx_mean = results["Predict ONNX Runtime"]["mean_ms"]
    if onnx_mean < native_mean:
        gain = round((native_mean - onnx_mean) / native_mean * 100, 1)
        print(f"\nONNX est {gain}% plus rapide que le chemin natif actuel.")
    else:
        cost = round((onnx_mean - native_mean) / native_mean * 100, 1)
        print(f"\nONNX est {cost}% PLUS LENT que le chemin natif actuel sur ce modele.")

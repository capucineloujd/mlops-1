import random

import mlflow.pyfunc
import numpy as np
import pandas as pd
import pytest

mlflow.set_tracking_uri("sqlite:///mlflow.db")


@pytest.fixture(scope="module")
def pyfunc_model():
    try:
        return mlflow.pyfunc.load_model("models:/lightgbm-credit-scoring-serving@gagnant")
    except Exception as exc:
        pytest.skip(f"Modele MLflow indisponible : {exc}")


def _build_records(pyfunc_model, values: dict) -> tuple[dict, dict]:
    native_model = pyfunc_model.unwrap_python_model().model
    schema_inputs = pyfunc_model.metadata.get_input_schema().inputs
    boolean_names = {c.name for c in schema_inputs if str(c.type) == "DataType.boolean"}

    record_pyfunc = {c.name: (False if c.name in boolean_names else 0.0) for c in schema_inputs}
    record_pyfunc.update(values)

    record_native = {name: 0.0 for name in native_model.feature_name_}
    record_native.update(values)

    return record_pyfunc, record_native


def _predict_both_paths(pyfunc_model, values: dict) -> tuple[float, float]:
    native_model = pyfunc_model.unwrap_python_model().model
    record_pyfunc, record_native = _build_records(pyfunc_model, values)

    proba_pyfunc = pyfunc_model.predict(pd.DataFrame([record_pyfunc]))[0]

    array_native = np.array([[record_native[n] for n in native_model.feature_name_]], dtype=np.float64)
    proba_native = native_model.booster_.predict(array_native)[0]

    return float(proba_pyfunc), float(proba_native)


class TestEquivalencePredictionAvantApresOptimisation:
    def test_predictions_identiques_sur_un_client_type(self, pyfunc_model):
        values = {
            "AMT_INCOME_TOTAL": 150000.0,
            "AMT_CREDIT": 500000.0,
            "DAYS_BIRTH": -15000.0,
            "CNT_CHILDREN": 0.0,
        }

        proba_pyfunc, proba_native = _predict_both_paths(pyfunc_model, values)

        assert proba_pyfunc == pytest.approx(proba_native, abs=1e-9)

    def test_predictions_identiques_sur_30_echantillons_varies(self, pyfunc_model):
        random.seed(0)

        for _ in range(30):
            values = {
                "AMT_INCOME_TOTAL": random.uniform(30000, 500000),
                "AMT_CREDIT": random.uniform(50000, 1500000),
                "DAYS_BIRTH": random.uniform(-25000, -7000),
                "CNT_CHILDREN": float(random.choice([0, 1, 2, 3])),
                "EXT_SOURCE_2": random.uniform(0.01, 0.99),
                "EXT_SOURCE_3": random.uniform(0.01, 0.99),
            }

            proba_pyfunc, proba_native = _predict_both_paths(pyfunc_model, values)

            assert proba_pyfunc == pytest.approx(proba_native, abs=1e-9)

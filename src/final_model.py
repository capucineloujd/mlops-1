import numpy as np
import pandas as pd

import mlflow
import mlflow.lightgbm
import mlflow.pyfunc
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, recall_score, precision_score, f1_score

MODEL_NAME = "lightgbm-credit-scoring"
SERVING_MODEL_NAME = "lightgbm-credit-scoring-serving"

# Meilleurs hyperparamètres trouves par Optuna 
BEST_PARAMS = {
    "n_estimators": 398,
    "max_depth": 3,
    "num_leaves": 119,
    "learning_rate": 0.12410214049339299,
    "min_child_samples": 45,
    "subsample": 0.6541145684135494,
    "colsample_bytree": 0.7383830850565832,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}


# On charge le jeu de données
def load_data(path: str = "data/train_engineered.csv") -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)

    # LightGBM ne supporte pas les caracteres séeciaux JSON dans les noms de colonnes
    df.columns = df.columns.str.replace(r'[\[\]{},:"\'\\]', "_", regex=True)

    X = df.drop(columns=["TARGET", "SK_ID_CURR"])
    y = df["TARGET"]

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    return X, y


# Déf métrique coût métier 
def cout_metier(y_true, y_pred, cout_fn: int = 10, cout_fp: int = 1) -> float:
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    return cout_fn * fn + cout_fp * fp


# Seuil qui minimise le coût métier 
def seuil_optimal(y_true, y_pred_proba) -> float:
    _, _, thresholds = roc_curve(y_true, y_pred_proba, drop_intermediate=False)
    couts = [cout_metier(y_true, (y_pred_proba > t).astype(int)) for t in thresholds]
    return float(thresholds[np.argmin(couts)])


# Wrapper pyfunc : retourne des probabilités de défault, pour un serving REST agnostique du client 
class LGBMProbaWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        self.model = mlflow.lightgbm.load_model(context.artifacts["lgbm_model"])

    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]


def train_and_register(data_path: str = "data/train_engineered.csv") -> None:
    """Entraîne le modèle final, l'enregistre dans le Model Registry avec l'alias gagnant, puis enregistre le wrapper
    pyfunc de serving associé."""
    X, y = load_data(data_path)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("home-credit-scoring")

    model = LGBMClassifier(**BEST_PARAMS)

    with mlflow.start_run(
        run_name="lightgbm_final",
        tags={"type": "optuna_best", "seuil": "optimise_cout_metier"},
        description="Modèle final : LightGBM + meilleurs hyperparamètres Optuna, seuil optimisé sur le coût métier",
    ):
        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_val)[:, 1]
        threshold = seuil_optimal(y_val, y_pred_proba)
        y_pred = (y_pred_proba > threshold).astype(int)

        cout = cout_metier(y_val, y_pred)
        cout_naif = cout_metier(y_val, np.zeros(len(y_val)).astype(int))

        mlflow.log_params(BEST_PARAMS)
        mlflow.log_param("seuil_optimal", round(threshold, 3))
        mlflow.log_metric("auc", roc_auc_score(y_val, y_pred_proba))
        mlflow.log_metric("recall_minority", recall_score(y_val, y_pred, pos_label=1))
        mlflow.log_metric("precision_minority", precision_score(y_val, y_pred, pos_label=1))
        mlflow.log_metric("f1", f1_score(y_val, y_pred, pos_label=1))
        mlflow.log_metric("cout_metier", cout)
        mlflow.log_metric("ratio_vs_naif", cout / cout_naif)

        mlflow.lightgbm.log_model(model, artifact_path="model", registered_model_name=MODEL_NAME)

    client = MlflowClient()
    latest_version = client.get_registered_model(MODEL_NAME).latest_versions[-1].version
    client.set_registered_model_alias(name=MODEL_NAME, alias="gagnant", version=latest_version)

    _register_serving_wrapper(model, X_val)


def _register_serving_wrapper(model: LGBMClassifier, X_val: pd.DataFrame) -> None:
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "lgbm_model")
        mlflow.lightgbm.save_model(model, model_path)

        with mlflow.start_run(
            run_name="lightgbm_pyfunc_serving",
            tags={"type": "serving"},
            description="Wrapper pyfunc du modèle final pour serving REST (retourne des probabilités)",
        ):
            model_info = mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=LGBMProbaWrapper(),
                artifacts={"lgbm_model": model_path},
                registered_model_name=SERVING_MODEL_NAME,
                signature=infer_signature(X_val, model.predict_proba(X_val.iloc[:5])[:, 1]),
                input_example=X_val.iloc[:5],
            )

    client = MlflowClient()
    serving_version = model_info.registered_model_version
    client.set_registered_model_alias(name=SERVING_MODEL_NAME, alias="gagnant", version=serving_version)


if __name__ == "__main__":
    train_and_register()

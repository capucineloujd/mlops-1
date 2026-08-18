import json
import os

import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

from storage import load_successful_inputs

MIN_CURRENT_SAMPLES = 30

_REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "reference_sample.csv")


def load_reference(reference_path: str | None = None) -> pd.DataFrame:
    return pd.read_csv(reference_path or _REFERENCE_PATH)


def load_current_from_logs(
    database_url: str | None = None, window: int = 200, columns: list[str] | None = None
) -> pd.DataFrame:
    """Reconstruit un DataFrame des inputs recents a partir des appels /predict
    réussis (les rejets métier/schéma n'ont jamais atteint le modèle)."""
    input_jsons = load_successful_inputs(database_url, window)

    records = []
    for input_json in input_jsons:
        records.extend(json.loads(input_json))

    df = pd.DataFrame(records)
    if columns is not None:
        df = df[[c for c in columns if c in df.columns]]
    return df


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> Report:
    shared_columns = [c for c in reference.columns if c in current.columns]
    reference = reference[shared_columns]
    current = current[shared_columns]

    data_definition = DataDefinition()
    reference_ds = Dataset.from_pandas(reference, data_definition=data_definition)
    current_ds = Dataset.from_pandas(current, data_definition=data_definition)

    report = Report(metrics=[DataDriftPreset()])
    return report.run(reference_data=reference_ds, current_data=current_ds)


def summarize(result) -> dict:
    """Extrait un resume exploitable du resultat brut Evidently."""
    metrics = result.dict()["metrics"]

    summary = {"n_drifted_columns": None, "drift_share": None, "drifted_columns": [], "column_drift": {}}

    for m in metrics:
        name = m["metric_name"]
        if name.startswith("DriftedColumnsCount"):
            summary["n_drifted_columns"] = m["value"]["count"]
            summary["drift_share"] = m["value"]["share"]
        elif name.startswith("ValueDrift"):
            column = m["config"]["column"]
            value = m["value"]
            method = m["config"].get("method", "")
            threshold = m["config"].get("threshold", 0.1)
            summary["column_drift"][column] = value
            # methodes en p-value (K-S, chi2...) : derive si value < threshold
            # methodes en distance (Wasserstein, Jensen-Shannon...) : derive si value > threshold
            is_drifted = value < threshold if "p_value" in method else value > threshold
            if is_drifted:
                summary["drifted_columns"].append(column)

    return summary


if __name__ == "__main__":
    reference = load_reference()
    current = load_current_from_logs(window=200, columns=list(reference.columns))

    if len(current) < MIN_CURRENT_SAMPLES:
        print(
            f"Pas assez de donnees recentes pour une analyse de drift fiable "
            f"({len(current)} < {MIN_CURRENT_SAMPLES} echantillons)."
        )
    else:
        result = run_drift_report(reference, current)
        summary = summarize(result)

        print(f"Colonnes analysees : {list(summary['column_drift'].keys())}")
        print(f"Colonnes en derive : {summary['drifted_columns']} ({summary['n_drifted_columns']} au total)")
        print(f"Part de derive     : {summary['drift_share']}")

        result.save_html("drift_report.html")
        print("\nRapport visuel enregistre : drift_report.html")

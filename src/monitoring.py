import json
import os
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from scipy.stats import ks_2samp

from storage import load_rows

ERROR_RATE_THRESHOLD = float(os.environ.get("ERROR_RATE_THRESHOLD", "0.10"))
LATENCY_P95_THRESHOLD_MS = float(os.environ.get("LATENCY_P95_THRESHOLD_MS", "500"))
DRIFT_PVALUE_THRESHOLD = float(os.environ.get("DRIFT_PVALUE_THRESHOLD", "0.05"))
DRIFT_MIN_SAMPLES = 30  # pas assez d'appels recents -> le test KS n'est pas fiable

_REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "reference_distribution.json")


@dataclass
class Anomaly:
    check: str
    severity: str  # "warning" ou bien "critical"
    message: str


@dataclass
class Report:
    n_calls_analyzed: int
    error_rate: float | None = None
    latency_p95_ms: float | None = None
    latency_mean_ms: float | None = None
    drift: dict[str, float] = field(default_factory=dict)  # champ -> p-value
    anomalies: list[Anomaly] = field(default_factory=list)


def _load_recent_calls(database_url: str | None, limit: int) -> list[dict[str, Any]]:
    return load_rows(database_url, limit)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(int(len(values) * p), len(values) - 1)
    return values[idx]


def _check_error_rate(rows: list[dict[str, Any]], report: Report) -> None:
    if not rows:
        return
    n_errors = sum(1 for r in rows if r["status"] == "error")
    report.error_rate = n_errors / len(rows)
    if report.error_rate > ERROR_RATE_THRESHOLD:
        report.anomalies.append(
            Anomaly(
                check="error_rate",
                severity="critical",
                message=(
                    f"Taux d'erreur {report.error_rate:.1%} au-dessus du seuil "
                    f"{ERROR_RATE_THRESHOLD:.0%} ({n_errors}/{len(rows)} appels)"
                ),
            )
        )


def _check_latency(rows: list[dict[str, Any]], report: Report) -> None:
    latencies = [r["latency_ms"] for r in rows if r["status"] == "success" and r["latency_ms"] is not None]
    if not latencies:
        return
    report.latency_mean_ms = round(mean(latencies), 2)
    report.latency_p95_ms = round(_percentile(latencies, 0.95), 2)
    if report.latency_p95_ms > LATENCY_P95_THRESHOLD_MS:
        report.anomalies.append(
            Anomaly(
                check="latency",
                severity="warning",
                message=(
                    f"Latence p95 {report.latency_p95_ms}ms au-dessus du seuil "
                    f"{LATENCY_P95_THRESHOLD_MS}ms"
                ),
            )
        )


def _check_data_drift(rows: list[dict[str, Any]], report: Report, reference_path: str) -> None:
    if not os.path.exists(reference_path):
        return

    with open(reference_path) as fh:
        reference = json.load(fh)

    recent_inputs = []
    for r in rows:
        try:
            records = json.loads(r["input_json"])
        except (TypeError, ValueError):
            continue
        recent_inputs.extend(records)

    for field_name, ref_values in reference.items():
        observed = [
            rec[field_name]
            for rec in recent_inputs
            if field_name in rec and isinstance(rec[field_name], (int, float)) and not isinstance(rec[field_name], bool)
        ]
        if len(observed) < DRIFT_MIN_SAMPLES:
            continue

        _statistic, p_value = ks_2samp(observed, ref_values)
        report.drift[field_name] = round(float(p_value), 4)

        if p_value < DRIFT_PVALUE_THRESHOLD:
            report.anomalies.append(
                Anomaly(
                    check="data_drift",
                    severity="warning",
                    message=(
                        f"Derive detectee sur '{field_name}' (test KS, p-value={p_value:.4f} "
                        f"< {DRIFT_PVALUE_THRESHOLD}) : la distribution recente differe "
                        f"significativement des donnees d'entrainement"
                    ),
                )
            )


def analyze(
    database_url: str | None = None, reference_path: str | None = None, window: int = 200
) -> Report:
    """Analyse les window derniers appels enregistres en base."""
    ref_path = reference_path or _REFERENCE_PATH

    rows = _load_recent_calls(database_url, window)
    report = Report(n_calls_analyzed=len(rows))

    _check_error_rate(rows, report)
    _check_latency(rows, report)
    _check_data_drift(rows, report, ref_path)

    return report


if __name__ == "__main__":
    result = analyze()
    print(f"Appels analyses : {result.n_calls_analyzed}")
    print(f"Taux d'erreur   : {result.error_rate}")
    print(f"Latence moyenne : {result.latency_mean_ms}ms | p95 : {result.latency_p95_ms}ms")
    print(f"Drift (p-values): {result.drift}")
    if result.anomalies:
        print(f"\n{len(result.anomalies)} anomalie(s) detectee(s) :")
        for a in result.anomalies:
            print(f"  [{a.severity.upper()}] {a.check} : {a.message}")
    else:
        print("\nAucune anomalie detectee.")

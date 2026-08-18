import streamlit as st
import streamlit.components.v1 as components

from drift_analysis import MIN_CURRENT_SAMPLES as DRIFT_MIN_SAMPLES
from drift_analysis import load_current_from_logs, load_reference, run_drift_report, summarize
from monitoring import analyze
from storage import load_calls_df

st.set_page_config(page_title="Monitoring -- Scoring API", layout="wide")
st.title("Monitoring de l'API de scoring credit")

window = st.sidebar.slider("Fenêtre d'analyse (nombre d'appels récents)", 10, 500, 200, step=10)

report = analyze(window=window)
calls_df = load_calls_df(limit=window)

st.header("Vue d'ensemble")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Appels analyses", report.n_calls_analyzed)
col2.metric("Taux d'erreur", f"{report.error_rate:.1%}" if report.error_rate is not None else "N/A")
col3.metric("Latence moyenne", f"{report.latency_mean_ms}ms" if report.latency_mean_ms is not None else "N/A")
col4.metric("Latence p95", f"{report.latency_p95_ms}ms" if report.latency_p95_ms is not None else "N/A")

if report.anomalies:
    st.subheader(f"{len(report.anomalies)} anomalie(s) détectée(s)")
    for a in report.anomalies:
        if a.severity == "critical":
            st.error(f"**{a.check}** : {a.message}")
        else:
            st.warning(f"**{a.check}** : {a.message}")
else:
    st.success("Aucune anomalie détectée (taux d'erreur, latence).")

st.header("Series temporelles")

if calls_df.empty:
    st.info("Aucun appel enregistré pour l'instant.")
else:
    chart_df = calls_df.sort_values("timestamp").set_index("timestamp")

    left, right = st.columns(2)
    with left:
        st.caption("Latence dans le temps (appels réussis)")
        success_latency = chart_df[chart_df["status"] == "success"][["latency_ms"]]
        if not success_latency.empty:
            st.line_chart(success_latency)
        else:
            st.info("Pas d'appel réussi dans la fenêtre.")

    with right:
        st.caption("Répartition des statuts")
        st.bar_chart(calls_df["status"].value_counts())

st.header("Dérive des données (data drift)")

reference = load_reference()
current = load_current_from_logs(window=window, columns=list(reference.columns))

if len(current) < DRIFT_MIN_SAMPLES:
    st.info(
        f"Pas assez de données récentes pour une analyse de drift fiable "
        f"({len(current)} / {DRIFT_MIN_SAMPLES} échantillons minimum)."
    )
else:
    result = run_drift_report(reference, current)
    drift_summary = summarize(result)

    st.caption(
        f"{len(drift_summary['drifted_columns'])} colonne(s) en derive sur "
        f"{len(drift_summary['column_drift'])} analysées"
    )
    st.bar_chart(drift_summary["column_drift"])

    if drift_summary["drifted_columns"]:
        st.warning(f"Colonnes en dérive : {', '.join(drift_summary['drifted_columns'])}")

    with st.expander("Rapport Evidently detaille"):
        components.html(result.get_html_str(), height=800, scrolling=True)

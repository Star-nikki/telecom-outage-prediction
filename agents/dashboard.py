"""
Streamlit dashboard -- interactive front-end over the real orchestration
pipeline (orchestrator_v3.py / orchestrator_llm.py). Does not reimplement
any prediction, root-cause, RAG, or recommendation logic -- it imports and
calls the exact same build_graph()/AgentState machinery the CLI scripts
use, so whatever the dashboard shows is guaranteed consistent with what
`python orchestrator_llm.py --tower X` would print in a terminal.

WHERE TO SAVE:
    Same folder as orchestrator_v3.py / orchestrator_llm.py (agents/), since
    it imports both directly.

INSTALL:
    pip install streamlit --break-system-packages

RUN (from the agents/ folder):
    streamlit run dashboard.py

WHAT IT DOES:
    - Tower picker + snapshot controls (hours-before-outage, threshold,
      backend) in the sidebar -- same knobs as the CLI --tower/
      --hours-before/--threshold/--backend flags. No LLM provider choice
      is exposed -- the dashboard always runs orchestrator_v3.py's fully
      deterministic, rule-based pipeline.
    - Runs orchestrator_v3.py's graph via build_graph().invoke(), same as
      its own CLI main().
    - Renders the resulting AgentState: executive summary, confidence
      across all 7 horizons as a bar chart, root cause reasoning,
      historical pattern at this tower, retrieved knowledge base
      references, recommended actions, priority/impact.
    - A second tab runs a fleet-wide sweep (all towers at their most recent
      snapshot) through just the Decision Agent (not the full LLM pipeline,
      for speed) so you can see which towers are currently highest-risk at
      a glance, then click through to a full per-tower analysis.

WHAT IT DOES NOT DO:
    Does not retrain anything, does not modify saved models, does not touch
    calibration or confidence formulas (Priority 4/5 work stays exactly as
    it is -- this only visualizes what the existing pipeline already
    produces).
"""
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decision_agent as da
import orchestrator_v3 as o3
from calibrated_prediction import predict_with_confidence_calibrated
from config import PREDICTION_HORIZONS
import query_agent

st.set_page_config(page_title="Telecom Outage Prediction", layout="wide")


@st.cache_data(show_spinner="Loading telemetry data...")
def load_data():
    df = pd.read_parquet(o3.DATA_PATH)
    return df


@st.cache_data(show_spinner=False)
def get_tower_list(_df):
    return sorted(_df["tower_id"].unique())


def run_pipeline(tower_id, backend, threshold, primary_horizon, hours_before):
    df = load_data()
    tdf = df[df["tower_id"] == tower_id].sort_values("timestamp")
    tdf = da.add_trend_features(tdf)
    row_series = o3.pick_demo_row(tdf, tower_id, hours_before_outage=hours_before)
    row = row_series.to_dict()
    row["timestamp"] = str(row["timestamp"])

    initial_state = {
        "tower_id": tower_id, "row": row, "evidence": {}, "predictions": {},
        "primary_probability": 0.0, "primary_confidence": 0.0, "threshold": threshold,
        "root_cause_category": None, "root_cause_reasoning": None,
        "knowledge_context": None, "historical_context": None, "recommendation": None,
        "final_report": None, "provider_name": "rule_based", "backend": backend,
        "primary_horizon": primary_horizon,
    }

    app = o3.build_graph()
    return app.invoke(initial_state, config={"recursion_limit": 50})


def render_result(state):
    col1, col2, col3 = st.columns(3)
    col1.metric(f"Probability ({state['primary_horizon']})", f"{state['primary_probability']*100:.1f}%")
    col2.metric("Confidence", f"{state['primary_confidence']*100:.1f}%")
    col3.metric("Root cause", state.get("root_cause_category") or "n/a")

    if state.get("root_cause_reasoning"):
        st.subheader("Root cause reasoning")
        st.write(state["root_cause_reasoning"])

    if state.get("historical_context"):
        st.subheader("Historical pattern at this tower")
        st.write(state["historical_context"].get("summary", "No summary available."))

    if state.get("predictions"):
        st.subheader("Probability & confidence across all horizons")
        horizon_order = [label for label, _ in PREDICTION_HORIZONS]
        rows = []
        for label in horizon_order:
            pred = state["predictions"].get(label)
            if pred:
                rows.append({"horizon": label, "probability": pred["probability"],
                             "confidence": pred["confidence"]})
        if rows:
            chart_df = pd.DataFrame(rows).set_index("horizon")
            st.bar_chart(chart_df)

    if state.get("knowledge_context"):
        st.subheader("Knowledge base references")
        for doc in state["knowledge_context"]:
            with st.expander(f"[{doc['id']}] relevance={doc.get('relevance_score', 'n/a')}"):
                st.write(doc["text"])

    if state.get("recommendation"):
        st.subheader("Recommended actions")
        st.write(state["recommendation"])

    if state.get("final_report"):
        with st.expander("Full text report"):
            st.text(state["final_report"])


def render_fleet_sweep(df, backend):
    st.subheader("Fleet-wide snapshot (Decision Agent only -- fast, no LLM/RAG)")
    horizon = st.selectbox("Horizon", [label for label, _ in PREDICTION_HORIZONS],
                            index=3, key="fleet_horizon")
    if st.button("Run fleet sweep"):
        model_path = os.path.join(o3.MODEL_DIR, f"decision_model_{backend}_{horizon}.joblib")
        if not os.path.exists(model_path):
            st.error(f"No saved model at {model_path}. Train it first.")
            return
        import joblib
        models = joblib.load(model_path)
        rows = []
        progress = st.progress(0)
        towers = sorted(df["tower_id"].unique())
        for i, tid in enumerate(towers):
            tdf = df[df["tower_id"] == tid].sort_values("timestamp")
            tdf = da.add_trend_features(tdf)
            latest = tdf.iloc[-1]
            prob, conf, method = predict_with_confidence_calibrated(
                models, latest, backend, horizon, model_dir=o3.MODEL_DIR)
            rows.append({"tower_id": tid, "probability": prob, "confidence": conf,
                         "calibration": method})
            progress.progress((i + 1) / len(towers))
        progress.empty()
        result_df = pd.DataFrame(rows).sort_values("probability", ascending=False)
        st.dataframe(result_df, use_container_width=True)
        st.bar_chart(result_df.set_index("tower_id")["probability"].head(20))


def main():
    st.title("Telecom 5G Outage Prediction Dashboard")

    df = load_data()
    towers = get_tower_list(df)

    provider = "rule_based"  # fixed -- no LLM providers exposed to the user

    st.sidebar.header("Configuration")
    tower_id = st.sidebar.selectbox("Tower", towers)
    backend = st.sidebar.selectbox("Backend", ["lightgbm", "xgboost", "random_forest"])
    threshold = st.sidebar.slider("Alert threshold", 0.0, 1.0, 0.5, 0.05)
    primary_horizon = st.sidebar.selectbox(
        "Primary horizon", [label for label, _ in PREDICTION_HORIZONS], index=3)
    hours_before = st.sidebar.slider(
        "Demo snapshot: hours before recorded outage", 0.5, 24.0, 3.0, 0.5,
        help="Lower = closer to failure (easier). Higher = genuine early-warning test.")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Single tower analysis", "Fleet-wide snapshot", "Event history", "Ask a Question"])

    with tab1:
        if st.button("Run analysis", type="primary"):
            with st.spinner(f"Running orchestration for {tower_id}..."):
                try:
                    state = run_pipeline(tower_id, backend, threshold, primary_horizon, hours_before)
                    render_result(state)
                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    st.exception(e)

    with tab2:
        render_fleet_sweep(df, backend)

    with tab3:
        st.subheader(f"Recorded events for {tower_id}")
        if os.path.exists(o3.EVENTS_PATH):
            events = pd.read_csv(o3.EVENTS_PATH, parse_dates=["ramp_start", "outage_start", "outage_end"])
            tower_events = events[events["tower_id"] == tower_id].sort_values("outage_start")
            if len(tower_events):
                tower_events = tower_events.copy()
                tower_events["lead_time_h"] = (
                    tower_events["outage_start"] - tower_events["ramp_start"]
                ).dt.total_seconds() / 3600
                tower_events["duration_h"] = (
                    tower_events["outage_end"] - tower_events["outage_start"]
                ).dt.total_seconds() / 3600
                st.dataframe(tower_events, use_container_width=True)
            else:
                st.info(f"No recorded events for {tower_id}.")
        else:
            st.warning(f"Events file not found at {o3.EVENTS_PATH}")

    with tab4:
        st.subheader("Ask a question about tower risk")
        st.caption(
            "Natural-language questions routed through query_agent.py to the real "
            "decision/root-cause/historical pipeline -- not a separate chatbot, same "
            "functions the other tabs use."
        )
        with st.expander("Example questions"):
            st.markdown(
                "- *What's the risk at T0000?*\n"
                "- *Is T0005 okay?*\n"
                "- *Why is T0003 failing?*\n"
                "- *Has T0000 failed before?*\n"
                "- *What should I do about T0007?*\n"
                "- *Compare T0000 and T0001*\n"
                "- *Which towers are most at risk?*"
            )

        if "qa_history" not in st.session_state:
            st.session_state.qa_history = []

        for role, text in st.session_state.qa_history:
            with st.chat_message(role):
                st.write(text)

        question = st.chat_input("Ask about tower risk, root cause, history, or recommendations...")
        if question:
            st.session_state.qa_history.append(("user", question))
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                with st.spinner("Routing to the pipeline..."):
                    try:
                        response = query_agent.answer(question, df, backend, provider)
                    except Exception as e:
                        response = f"Error answering that question: {e}"
                st.write(response)
            st.session_state.qa_history.append(("assistant", response))

        if st.session_state.qa_history:
            if st.button("Clear conversation"):
                st.session_state.qa_history = []
                st.rerun()


if __name__ == "__main__":
    main()

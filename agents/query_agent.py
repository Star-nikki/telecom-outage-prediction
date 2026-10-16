"""
Query Agent -- natural-language question -> intent + tower-ID extraction ->
route to EXISTING orchestrator functions. Does not reimplement any
prediction, root-cause, retrieval, or recommendation logic itself; every
intent handler below is a thin wrapper that calls into decision_agent.py /
orchestrator_v3.py / orchestrator_llm.py, the same functions the CLI and
dashboard use.

Intent extraction is deliberately keyword/regex-based, not an LLM call --
this keeps it fast, free, deterministic, and easy to extend, and the
question set for a project like this (risk/status/reason/history/compare)
is small and stable enough that a classifier would be overkill. If you
outgrow this later, swap _extract_intent()'s body for an LLM call; the
rest of the file (routing, handlers) doesn't need to change.

WHERE TO SAVE:
    Same folder as orchestrator_v3.py / orchestrator_llm.py (agents/).

RUN (interactive):
    python query_agent.py

RUN (single question, non-interactive):
    python query_agent.py --question "what's the risk at T0000 in the next 3 hours"

SUPPORTED INTENTS (see _extract_intent() for the exact patterns each one
matches):
    RISK        "what's the risk at T0000", "how likely is an outage at T0012"
    STATUS      "is T0000 okay", "status of T0005"
    ROOT_CAUSE  "why is T0000 failing", "what's causing the problem at T0003"
    HISTORY     "has T0000 failed before", "history of T0000"
    RECOMMEND   "what should I do about T0000", "recommend action for T0007"
    COMPARE     "compare T0000 and T0001"
    FLEET_TOP   "which towers are most at risk", "top risky towers"
    FLEET_SUMMARY   "how many towers are degraded", "how's the network doing overall"
                     -- no tower ID needed, aggregates the real is_outage/is_degraded
                     columns across the whole fleet's latest snapshot.
    REGION      "which towers are in region R2", "how many towers are degraded in R3"
                -- the dataset's real `region` column is R0-R5 (generic codes, not
                real city names). See REGION_CITY_NAMES below if you want to map
                codes to actual place names.
    CAUSE_DISTRIBUTION  "what's the most common cause of outages",
                        "why do towers usually fail" -- aggregates the real,
                        ground-truth `root_cause` column across all recorded
                        outages, not a guess.
    DEFINITION  "what is congestion collapse", "what does capacity creep mean"
                -- answered from a small built-in glossary, no data lookup.
    HELP        anything unmatched -- prints supported intents, does not guess

NOTE ON "towers in Mumbai"-style questions: the dataset has NO city/location
names at all, only generic region codes (R0-R5). If you want "Mumbai" to
actually resolve, fill in REGION_CITY_NAMES below with your own mapping --
until then, region questions only work with the codes themselves (e.g. "R2").
"""
import argparse
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decision_agent as da
import orchestrator_v3 as o3
from config import PREDICTION_HORIZONS

TOWER_PATTERN = re.compile(r"\bT\d{3,5}\b", re.IGNORECASE)
REGION_PATTERN = re.compile(r"\bR[0-5]\b", re.IGNORECASE)

# The real `region` column in telecom_timeseries.parquet only has generic
# codes (R0-R5) -- no actual city/place names exist anywhere in the dataset.
# Fill this in with your own mapping if you want "Mumbai"-style questions to
# resolve to a real region code; until then, only the codes themselves work.
# Example: REGION_CITY_NAMES = {"R0": "Mumbai", "R1": "Delhi", "R2": "Bangalore", ...}
REGION_CITY_NAMES = {}
_CITY_TO_REGION = {v.lower(): k for k, v in REGION_CITY_NAMES.items()}

# Small built-in glossary for conceptual questions that aren't a data lookup
# at all -- answered directly, no model/pipeline call involved.
GLOSSARY = {
    "congestion_collapse": (
        "Congestion collapse: utilization sustained above roughly 80% with rising "
        "packet loss and latency, indicating the tower's capacity is being exhausted "
        "by current traffic -- an acute, recent-onset problem (see SOP-CONG-01)."
    ),
    "capacity_creep": (
        "Capacity creep: a gradual utilization increase over 24-72 hours with no single "
        "triggering event -- organic demand growth outpacing site capacity. This is a "
        "planning issue rather than an emergency, distinguished from congestion collapse "
        "by being slow and sustained rather than sudden (see SOP-CAP-01)."
    ),
    "hardware_power_failure": (
        "Hardware/power failure: battery level and grid voltage dropping together, "
        "typically from battery degradation or a grid supply fault, indicating imminent "
        "backup power exhaustion (see SOP-PWR-01/02)."
    ),
    "weather_induced_degradation": (
        "Weather-induced degradation: signal strength (RSRP) dropping in correlation with "
        "an active storm or heavy rainfall -- usually self-resolving once the weather "
        "clears (see SOP-WTHR-01)."
    ),
    "root_cause": (
        "Root cause here means which of the four failure categories (hardware/power, "
        "congestion collapse, weather-induced degradation, capacity creep) is most likely "
        "driving a predicted or actual outage, decided by deterministic signal scoring, "
        "not guessed by the LLM."
    ),
    "confidence": (
        "Confidence is a separate number from probability: probability is how likely an "
        "outage is; confidence is how much to trust that specific probability estimate. "
        "This project tested several confidence approaches (ensemble agreement, training-"
        "data density, conformal prediction, a stacked model) against real outcomes rather "
        "than assuming any one of them works."
    ),
    "calibration": (
        "Calibration corrects raw model probabilities so that, among all predictions at "
        "probability p, roughly p fraction are actually positive -- raw outputs here were "
        "found to be overconfident, especially at longer horizons, and were fixed with "
        "Platt scaling or Isotonic regression depending on which helped per horizon."
    ),
}
HORIZON_WORDS = {
    "15 min": "15m", "15m": "15m", "15-minute": "15m",
    "30 min": "30m", "30m": "30m", "half hour": "30m",
    "1 hour": "1h", "1h": "1h", "an hour": "1h",
    "3 hour": "3h", "3h": "3h", "three hour": "3h",
    "6 hour": "6h", "6h": "6h", "six hour": "6h",
    "12 hour": "12h", "12h": "12h",
    "24 hour": "24h", "24h": "24h", "a day": "24h", "tomorrow": "24h",
}


def _extract_towers(question: str):
    return [m.upper() for m in TOWER_PATTERN.findall(question)]


def _extract_region(question: str):
    """Returns a region CODE (e.g. 'R2') if the question names a region code
    directly, or a mapped city name via REGION_CITY_NAMES (if you've filled
    that in), or None if neither is present."""
    m = REGION_PATTERN.search(question)
    if m:
        return m.group(0).upper()
    q_lower = question.lower()
    for city, code in _CITY_TO_REGION.items():
        if city in q_lower:
            return code
    return None


def _extract_horizon(question: str, default="3h"):
    q = question.lower()
    for phrase, label in HORIZON_WORDS.items():
        if phrase in q:
            return label
    return default


def _extract_glossary_term(question: str):
    q = question.lower().replace("-", "_").replace(" ", "_")
    for term in GLOSSARY:
        if term in q or term.replace("_", " ") in question.lower():
            return term
    return None


def _extract_intent(question: str) -> str:
    q = question.lower()
    towers = _extract_towers(question)
    region = _extract_region(question)

    if any(w in q for w in ["compare", "vs", "versus"]) and len(towers) >= 2:
        return "COMPARE"
    if _extract_glossary_term(question) and any(
            w in q for w in ["what is", "what's", "what does", "define", "meaning", "mean"]):
        return "DEFINITION"
    if any(w in q for w in ["most common cause", "usually fail", "typical cause", "main cause",
                             "why do towers", "common root cause"]):
        return "CAUSE_DISTRIBUTION"
    if region and not towers:
        return "REGION"
    if not towers and any(w in q for w in [
            "how many towers", "overall", "network doing", "fleet", "how is the network",
            "any towers", "towers are degraded", "towers are down", "towers in outage",
            "total outages", "how's the network"]):
        return "FLEET_SUMMARY"
    if any(w in q for w in ["most at risk", "top risk", "riskiest", "which tower", "worst tower"]):
        return "FLEET_TOP"
    if any(w in q for w in ["why", "cause", "root cause", "reason"]):
        return "ROOT_CAUSE"
    if any(w in q for w in ["history", "before", "past", "previous", "prior", "happened"]):
        return "HISTORY"
    if any(w in q for w in ["recommend", "should i do", "what to do", "action", "fix"]):
        return "RECOMMEND"
    if any(w in q for w in ["risk", "likely", "probability", "chance", "will it fail", "outage"]):
        return "RISK"
    if any(w in q for w in ["status", "okay", "ok?", "healthy", "how is"]):
        return "STATUS"
    return "HELP"


def _load_snapshot(df, tower_id, hours_before=3.0):
    tdf = df[df["tower_id"] == tower_id].sort_values("timestamp")
    if len(tdf) == 0:
        return None
    tdf = da.add_trend_features(tdf)
    row_series = o3.pick_demo_row(tdf, tower_id, hours_before_outage=hours_before)
    return row_series


def _predict(row, backend, horizon):
    model_path = os.path.join(o3.MODEL_DIR, f"decision_model_{backend}_{horizon}.joblib")
    if not os.path.exists(model_path):
        return None, None
    import joblib
    models = joblib.load(model_path)
    return da.predict_with_confidence(models, row)


def handle_risk(df, towers, question, backend="lightgbm"):
    if not towers:
        return "I couldn't find a tower ID (like T0000) in that question."
    tower_id = towers[0]
    horizon = _extract_horizon(question)
    row = _load_snapshot(df, tower_id)
    if row is None:
        return f"No data found for {tower_id}."
    prob, conf = _predict(row, backend, horizon)
    if prob is None:
        return f"No trained model found for horizon {horizon} (backend={backend})."
    return (f"{tower_id}: {prob*100:.1f}% probability of outage within {horizon}, "
            f"model confidence {conf*100:.1f}%.")


def handle_status(df, towers, question, backend="lightgbm"):
    if not towers:
        return "I couldn't find a tower ID in that question."
    tower_id = towers[0]
    row = _load_snapshot(df, tower_id)
    if row is None:
        return f"No data found for {tower_id}."
    prob, _ = _predict(row, backend, "3h")
    if prob is None:
        return f"No trained model found for {tower_id}."
    status = "at risk (probability >= 50%)" if prob >= 0.5 else "currently within normal range"
    return f"{tower_id} is {status} (3h outage probability: {prob*100:.1f}%)."


def handle_root_cause(df, towers, question, backend="lightgbm", provider="rule_based"):
    if not towers:
        return "I couldn't find a tower ID in that question."
    tower_id = towers[0]
    return _run_full_pipeline(df, tower_id, backend, provider, only="root_cause")


def handle_history(df, towers, question):
    if not towers:
        return "I couldn't find a tower ID in that question."
    tower_id = towers[0]
    state = {"tower_id": tower_id, "root_cause_category": "general"}
    result = o3.historical_events_agent(state)
    return result["historical_context"]["summary"]


def handle_recommend(df, towers, question, backend="lightgbm", provider="rule_based"):
    if not towers:
        return "I couldn't find a tower ID in that question."
    tower_id = towers[0]
    return _run_full_pipeline(df, tower_id, backend, provider, only="recommendation")


def handle_compare(df, towers, question, backend="lightgbm"):
    horizon = _extract_horizon(question)
    lines = []
    for tower_id in towers[:2]:
        row = _load_snapshot(df, tower_id)
        if row is None:
            lines.append(f"{tower_id}: no data found.")
            continue
        prob, conf = _predict(row, backend, horizon)
        if prob is None:
            lines.append(f"{tower_id}: no trained model for {horizon}.")
            continue
        lines.append(f"{tower_id}: {prob*100:.1f}% probability within {horizon} (confidence {conf*100:.1f}%)")
    return "\n".join(lines) if lines else "Need at least two tower IDs to compare."


def handle_fleet_top(df, question, backend="lightgbm", top_n=5):
    horizon = _extract_horizon(question)
    model_path = os.path.join(o3.MODEL_DIR, f"decision_model_{backend}_{horizon}.joblib")
    if not os.path.exists(model_path):
        return f"No trained model found for horizon {horizon} (backend={backend})."
    import joblib
    models = joblib.load(model_path)
    rows = []
    for tower_id in sorted(df["tower_id"].unique()):
        tdf = df[df["tower_id"] == tower_id].sort_values("timestamp")
        tdf = da.add_trend_features(tdf)
        latest = tdf.iloc[-1]
        prob, conf = da.predict_with_confidence(models, latest)
        rows.append((tower_id, prob, conf))
    rows.sort(key=lambda r: r[1], reverse=True)
    lines = [f"Top {top_n} highest-risk towers at their latest snapshot ({horizon} horizon):"]
    for tower_id, prob, conf in rows[:top_n]:
        lines.append(f"  {tower_id}: {prob*100:.1f}% probability, {conf*100:.1f}% confidence")
    return "\n".join(lines)


def handle_definition(question: str):
    term = _extract_glossary_term(question)
    if term is None:
        return "I don't have a definition for that -- try asking about congestion_collapse, capacity_creep, hardware_power_failure, weather_induced_degradation, root cause, confidence, or calibration."
    return GLOSSARY[term]


def _latest_snapshot_per_tower(df):
    """One row per tower: its most recent recorded snapshot. Real columns
    (is_outage, is_degraded, region, root_cause) only, nothing derived."""
    return df.sort_values("timestamp").groupby("tower_id", as_index=False).tail(1)


def handle_fleet_summary(df):
    latest = _latest_snapshot_per_tower(df)
    n_total = len(latest)
    n_outage = int(latest["is_outage"].sum()) if "is_outage" in latest.columns else None
    n_degraded = int(latest["is_degraded"].sum()) if "is_degraded" in latest.columns else None

    lines = [f"Fleet snapshot across {n_total} towers (most recent recorded reading each):"]
    if n_outage is not None:
        lines.append(f"  Currently in outage: {n_outage}")
    if n_degraded is not None:
        lines.append(f"  Currently degraded (not yet full outage): {n_degraded}")
    healthy = n_total - (n_outage or 0) - (n_degraded or 0)
    lines.append(f"  Operating normally: {max(healthy, 0)}")

    if n_outage:
        outage_towers = latest[latest["is_outage"] == 1]["tower_id"].tolist()
        lines.append(f"  Tower(s) in outage: {', '.join(outage_towers[:15])}"
                      f"{' ...' if len(outage_towers) > 15 else ''}")
    return "\n".join(lines)


def handle_region(df, region_code, question):
    region_towers = sorted(df[df["region"] == region_code]["tower_id"].unique())
    if not region_towers:
        return f"No towers found with region code {region_code}. Known region codes: {sorted(df['region'].unique())}."

    latest = _latest_snapshot_per_tower(df)
    region_latest = latest[latest["tower_id"].isin(region_towers)]
    n_outage = int(region_latest["is_outage"].sum()) if "is_outage" in region_latest.columns else 0
    n_degraded = int(region_latest["is_degraded"].sum()) if "is_degraded" in region_latest.columns else 0

    city_note = ""
    for city, code in _CITY_TO_REGION.items():
        if code == region_code:
            city_note = f" ({city.title()})"
            break

    lines = [
        f"Region {region_code}{city_note}: {len(region_towers)} tower(s) -- {', '.join(region_towers)}",
        f"  Currently in outage: {n_outage}",
        f"  Currently degraded: {n_degraded}",
    ]
    return "\n".join(lines)


def handle_cause_distribution(df):
    if "root_cause" not in df.columns:
        return "No root_cause column available in this dataset."
    outages = df[df["root_cause"] != "none"]
    if len(outages) == 0:
        return "No recorded outages with a root cause in this dataset."
    counts = outages["root_cause"].value_counts()
    total = int(counts.sum())
    lines = [f"Root cause distribution across {total:,} recorded outage readings (ground truth from the simulator):"]
    for cause, n in counts.items():
        lines.append(f"  {cause}: {n:,} ({n/total*100:.1f}%)")
    return "\n".join(lines)


def _run_full_pipeline(df, tower_id, backend, provider, only=None):
    """Routes to the real orchestrator graph (root cause -> RAG ->
    recommendation) rather than reimplementing any of that logic here."""
    tdf = df[df["tower_id"] == tower_id].sort_values("timestamp")
    if len(tdf) == 0:
        return f"No data found for {tower_id}."
    tdf = da.add_trend_features(tdf)
    row_series = o3.pick_demo_row(tdf, tower_id, hours_before_outage=3.0)
    row = row_series.to_dict()
    row["timestamp"] = str(row["timestamp"])

    initial_state = {
        "tower_id": tower_id, "row": row, "evidence": {}, "predictions": {},
        "primary_probability": 0.0, "primary_confidence": 0.0, "threshold": 0.0,
        "root_cause_category": None, "root_cause_reasoning": None,
        "knowledge_context": None, "historical_context": None, "recommendation": None,
        "final_report": None, "provider_name": provider, "backend": backend,
        "primary_horizon": "3h",
    }
    app = o3.build_graph()
    final_state = app.invoke(initial_state, config={"recursion_limit": 50})

    if final_state.get("root_cause_category") is None:
        return (f"{tower_id} is below the alert threshold at this snapshot -- "
                f"no root cause or recommendation was generated.")

    if only == "root_cause":
        return (f"{tower_id} root cause: {final_state['root_cause_category']}\n"
                f"{final_state['root_cause_reasoning']}")
    if only == "recommendation":
        return f"{tower_id} recommended actions:\n{final_state['recommendation']}"
    return final_state.get("final_report", "No report generated.")


def answer(question: str, df, backend="lightgbm", provider="rule_based") -> str:
    intent = _extract_intent(question)
    towers = _extract_towers(question)

    if intent == "RISK":
        return handle_risk(df, towers, question, backend)
    if intent == "STATUS":
        return handle_status(df, towers, question, backend)
    if intent == "ROOT_CAUSE":
        return handle_root_cause(df, towers, question, backend, provider)
    if intent == "HISTORY":
        return handle_history(df, towers, question)
    if intent == "RECOMMEND":
        return handle_recommend(df, towers, question, backend, provider)
    if intent == "COMPARE":
        return handle_compare(df, towers, question, backend)
    if intent == "FLEET_TOP":
        return handle_fleet_top(df, question, backend)
    if intent == "FLEET_SUMMARY":
        return handle_fleet_summary(df)
    if intent == "REGION":
        region_code = _extract_region(question)
        return handle_region(df, region_code, question)
    if intent == "CAUSE_DISTRIBUTION":
        return handle_cause_distribution(df)
    if intent == "DEFINITION":
        return handle_definition(question)

    return (
        "I didn't recognize that question. Try things like:\n"
        "  - \"what's the risk at T0000?\"\n"
        "  - \"is T0005 okay?\"\n"
        "  - \"why is T0003 failing?\"\n"
        "  - \"has T0000 failed before?\"\n"
        "  - \"what should I do about T0007?\"\n"
        "  - \"compare T0000 and T0001\"\n"
        "  - \"which towers are most at risk?\"\n"
        "  - \"how many towers are currently degraded?\"\n"
        "  - \"which towers are in region R2?\"\n"
        "  - \"what's the most common cause of outages?\"\n"
        "  - \"what is congestion collapse?\""
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default=None, help="Ask a single question and exit.")
    parser.add_argument("--backend", default="lightgbm", choices=["lightgbm", "xgboost", "random_forest"])
    args = parser.parse_args()
    provider = "rule_based"  # fixed -- no LLM providers exposed to the user

    print(f"Loading {o3.DATA_PATH} ...")
    df = pd.read_parquet(o3.DATA_PATH)

    if args.question:
        print(answer(args.question, df, args.backend, provider))
        return

    print("Query Agent ready. Ask a question about tower risk (Ctrl+C to exit).")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not question:
            continue
        print(answer(question, df, args.backend, provider))


if __name__ == "__main__":
    main()

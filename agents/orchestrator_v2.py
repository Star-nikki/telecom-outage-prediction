"""
Orchestrator v2 -- branching architecture (your recommended design), replacing
the straight-line/requery-loop version in orchestrator.py.

ARCHITECTURE
------------
    Monitoring Agent
          |
          v
    Decision Agent (LightGBM/XGBoost)
          |
   probability > threshold?
      /              \\
    No                Yes
     |                 |
   Finish        Root Cause Agent
                       |
          +------------+------------+
          v                         v
   Knowledge Agent (RAG)   Historical Events Agent
          |                         |
          +------------+------------+
                       v
             Recommendation Agent (LLM)
                       |
                       v
                Report Agent --> END

Why branching instead of a straight pipeline: a linear chain reads as "five
Python functions called in sequence." Branching means the Root Cause Agent's
output genuinely determines what happens next (two independent lookups
happen in parallel, each contributing different context), and the
Recommendation Agent has to synthesize both before acting -- that's a real
multi-agent dependency structure, not just sequential function calls.

WHERE TO SAVE THIS FILE:
    <your project folder>\\telecom_sim_v2\\agents\\orchestrator_v2.py
    (alongside decision_agent.py, llm_providers.py, knowledge_base.py --
    imports all three)

USAGE (run from the agents/ folder, after decision_agent.py --mode train
has produced trained models in agents/models/):

    pip install langgraph scikit-learn --break-system-packages

    python orchestrator_v2.py --tower T0000
    python orchestrator_v2.py --tower T0000 --provider anthropic
    python orchestrator_v2.py --tower T0000 --threshold 0.6 --primary-horizon 3h
"""

import argparse
import os
import re
import sys
from typing import TypedDict, Optional, List

import joblib
import pandas as pd
from langgraph.graph import StateGraph, END

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import PREDICTION_HORIZONS  # noqa: E402
from llm_providers import get_provider  # noqa: E402
import decision_agent as da  # noqa: E402
import knowledge_base as kb  # noqa: E402

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "telecom_timeseries.parquet"
EVENTS_PATH = PROJECT_ROOT / "data" / "failure_events.csv"
MODEL_DIR = PROJECT_ROOT / "agents" / "models"

FAILURE_CATEGORIES = ["hardware_power_failure", "congestion_collapse",
                       "weather_induced_degradation", "capacity_creep"]


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    tower_id: str
    row: dict
    evidence: dict                  # tower_health/traffic/weather/user_behavior -> text
    predictions: dict               # horizon -> {probability, confidence}
    primary_probability: float
    primary_confidence: float
    threshold: float
    root_cause_category: Optional[str]
    root_cause_reasoning: Optional[str]
    knowledge_context: Optional[List[dict]]
    historical_context: Optional[str]
    recommendation: Optional[str]
    final_report: Optional[str]
    provider_name: str
    backend: str
    primary_horizon: str


# ---------------------------------------------------------------------------
# Monitoring Agent -- internally checks all four domains, reports as one node
# ---------------------------------------------------------------------------
def _trend(row: dict, col: str, hours: int):
    return row.get(f"{col}_delta_{hours}h")


def monitoring_agent(state: AgentState) -> dict:
    row = state["row"]
    evidence = {}

    battery, voltage, rsrp = row["battery_level_pct"], row["grid_voltage"], row["rsrp_dbm"]
    evidence["tower_health"] = (
        f"Battery {battery:.0f}%, grid voltage {voltage:.0f}V, RSRP {rsrp:.0f} dBm."
        + (" Power supply degrading." if (battery < 50 or voltage < 210) else " Power normal.")
    )

    util, packet_loss, throughput, latency = (
        row["utilization_pct"], row["packet_loss_pct"], row["throughput_mbps"], row["latency_ms"]
    )
    evidence["traffic"] = (
        f"Utilization {util:.0f}%, packet loss {packet_loss:.1f}%, throughput {throughput:.0f} Mbps, "
        f"latency {latency:.0f} ms."
        + (" Congested." if (util > 80 or packet_loss > 5) else " Normal.")
    )

    storm, rainfall, wind = row["storm_flag"], row["rainfall_mm"], row["wind_kmh"]
    evidence["weather"] = (
        f"Storm={'ACTIVE' if storm else 'clear'}, rainfall {rainfall:.1f}mm, wind {wind:.0f}km/h."
        + (" Storm probability high." if (storm == 1 or rainfall > 5) else "")
    )

    load_mult = row["load_multiplier"]
    evidence["user_behavior"] = (
        f"Load multiplier {load_mult:.2f}."
        + (" Unusual demand spike." if load_mult > 1.5 else " Normal demand pattern.")
    )

    d24_util = _trend(row, "utilization_pct", 24)
    if d24_util is not None:
        evidence["traffic"] += f" 24h utilization trend: {d24_util:+.1f} pts."

    return {"evidence": evidence}


# ---------------------------------------------------------------------------
# Decision Agent -- loads trained models, produces multi-horizon predictions
# ---------------------------------------------------------------------------
def decision_agent_node(state: AgentState) -> dict:
    row = pd.Series(state["row"])
    predictions = {}
    for label, _ in PREDICTION_HORIZONS:
        model_path = os.path.join(MODEL_DIR, f"decision_model_{state['backend']}_{label}.joblib")
        if not os.path.exists(model_path):
            continue
        models = joblib.load(model_path)
        prob, conf = da.predict_with_confidence(models, row)
        predictions[label] = {"probability": prob, "confidence": conf}

    primary = predictions.get(state["primary_horizon"], {"probability": 0.0, "confidence": 0.0})
    return {
        "predictions": predictions,
        "primary_probability": primary["probability"],
        "primary_confidence": primary["confidence"],
    }


def route_after_decision(state: AgentState) -> str:
    return "root_cause" if state["primary_probability"] >= state["threshold"] else "finish"


# ---------------------------------------------------------------------------
# Root Cause Agent
# ---------------------------------------------------------------------------
def root_cause_agent(state: AgentState) -> dict:
    provider_name = state["provider_name"]
    evidence = state["evidence"]

    if provider_name == "rule_based":
        category, reasoning = _rule_based_root_cause(evidence)
    else:
        provider = get_provider(provider_name)
        evidence_block = "\n".join(f"- {k}: {v}" for k, v in evidence.items())
        prompt = f"""Evidence from telecom monitoring agents for tower {state['tower_id']}:
{evidence_block}

Predicted outage probability at {state['primary_horizon']}: {state['primary_probability']*100:.0f}%

Which single category best explains this, and why? Choose exactly one:
hardware_power_failure, congestion_collapse, weather_induced_degradation, capacity_creep

Respond in EXACTLY this format:
CATEGORY: <one of the four categories above>
REASONING: <2-3 sentences explaining which signals support this and which don't>"""
        raw = provider.generate(prompt)
        category, reasoning = _parse_root_cause(raw)

    return {"root_cause_category": category, "root_cause_reasoning": reasoning}


def _rule_based_root_cause(evidence: dict) -> tuple:
    th, tr, we = evidence.get("tower_health", ""), evidence.get("traffic", ""), evidence.get("weather", "")
    if "degrading" in th.lower():
        return "hardware_power_failure", "Battery/voltage evidence directly indicates power supply degradation."
    if "storm probability high" in we.lower() and "congested" not in tr.lower():
        return "weather_induced_degradation", "Active storm conditions with no independent congestion signal."
    if "congested" in tr.lower() and "24h utilization trend" in tr and _extract_trend(tr) > 5:
        return "capacity_creep", "Congestion present alongside a sustained multi-day utilization climb."
    if "congested" in tr.lower():
        return "congestion_collapse", "Acute congestion signal (utilization/packet loss) without a slow multi-day trend."
    return "capacity_creep", "No acute signal stood out; defaulting to the slow-creep category as the mildest explanation."


def _extract_trend(traffic_text: str) -> float:
    m = re.search(r"trend:\s*([+-]?\d+\.?\d*)", traffic_text)
    return float(m.group(1)) if m else 0.0


def _parse_root_cause(text: str) -> tuple:
    cat_match = re.search(r"CATEGORY:\s*(\w+)", text)
    reason_match = re.search(r"REASONING:\s*(.+)", text, re.DOTALL)
    category = cat_match.group(1).strip() if cat_match else "capacity_creep"
    if category not in FAILURE_CATEGORIES:
        category = "capacity_creep"
    reasoning = reason_match.group(1).strip() if reason_match else "(no reasoning parsed)"
    return category, reasoning


# ---------------------------------------------------------------------------
# Knowledge / RAG Agent (parallel branch 1)
# ---------------------------------------------------------------------------
def knowledge_agent(state: AgentState) -> dict:
    query = f"{state['root_cause_category']} {state['root_cause_reasoning']}"
    results = kb.retrieve(query, top_k=2, failure_type=state["root_cause_category"])
    return {"knowledge_context": results}


# ---------------------------------------------------------------------------
# Historical Events Agent (parallel branch 2)
# ---------------------------------------------------------------------------
def historical_events_agent(state: AgentState) -> dict:
    if not os.path.exists(EVENTS_PATH):
        return {"historical_context": "No historical event log found at " + EVENTS_PATH}

    events = pd.read_csv(EVENTS_PATH, parse_dates=["ramp_start", "outage_start", "outage_end"])
    tower_id, category = state["tower_id"], state["root_cause_category"]

    same_tower_same_cause = events[(events["tower_id"] == tower_id) & (events["failure_type"] == category)]
    same_cause_fleet_wide = events[events["failure_type"] == category]

    if len(same_tower_same_cause):
        lead_times = (same_tower_same_cause["outage_start"] - same_tower_same_cause["ramp_start"]).dt.total_seconds() / 3600
        context = (
            f"This tower has {len(same_tower_same_cause)} prior {category} event(s) on record. "
            f"Average lead time from ramp start to outage was {lead_times.mean():.1f}h."
        )
    elif len(same_cause_fleet_wide):
        lead_times = (same_cause_fleet_wide["outage_start"] - same_cause_fleet_wide["ramp_start"]).dt.total_seconds() / 3600
        context = (
            f"No prior {category} events on this specific tower, but {len(same_cause_fleet_wide)} "
            f"occurred fleet-wide, with average lead time {lead_times.mean():.1f}h -- useful as a "
            f"general reference point."
        )
    else:
        context = f"No prior {category} events found anywhere in the historical log -- this would be a first occurrence."

    return {"historical_context": context}


# ---------------------------------------------------------------------------
# Recommendation Agent -- LLM-prompted, not hardcoded if/else
# ---------------------------------------------------------------------------
def recommendation_agent(state: AgentState) -> dict:
    row = state["row"]
    knowledge_text = "\n".join(f"- [{d['id']}] {d['text']}" for d in (state["knowledge_context"] or []))

    context = f"""You are a telecom network operations engineer.

Context:
Battery: {row['battery_level_pct']:.0f}%
Voltage: {row['grid_voltage']:.0f}V
Packet loss: {row['packet_loss_pct']:.1f}%
Predicted outage probability ({state['primary_horizon']}): {state['primary_probability']*100:.0f}%
Model confidence: {state['primary_confidence']*100:.0f}%
Root cause: {state['root_cause_category']} -- {state['root_cause_reasoning']}

Relevant maintenance knowledge:
{knowledge_text}

Historical precedent:
{state['historical_context']}

Generate specific, actionable recommendations for the operations team. Be concrete
about urgency and timing, and reference the knowledge/historical context where relevant."""

    if state["provider_name"] == "rule_based":
        recommendation = _rule_based_recommendation(state)
    else:
        provider = get_provider(state["provider_name"])
        recommendation = provider.generate(context)

    return {"recommendation": recommendation}


def _rule_based_recommendation(state: AgentState) -> str:
    top_doc = state["knowledge_context"][0] if state["knowledge_context"] else None
    urgency = "URGENT (within hours)" if state["primary_probability"] > 0.7 else "Routine (schedule within days)"
    lines = [
        f"Priority: {urgency}",
        f"Root cause: {state['root_cause_category']} -- {state['root_cause_reasoning']}",
    ]
    if top_doc:
        lines.append(f"Per {top_doc['id']}: {top_doc['text']}")
    lines.append(f"Historical context: {state['historical_context']}")
    lines.append("(Rule-based summary -- for richer, context-synthesized recommendations, "
                  "run with --provider anthropic/openai/ollama.)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report Agent
# ---------------------------------------------------------------------------
def report_agent(state: AgentState) -> dict:
    row = state["row"]
    pred_lines = "\n".join(
        f"- {h}: {p['probability']*100:.0f}% probability, {p['confidence']*100:.0f}% confidence"
        for h, p in state["predictions"].items()
    )
    knowledge_lines = "\n".join(f"- [{d['id']}] {d['text']}" for d in (state["knowledge_context"] or []))
    connected_users = row.get("connected_users", 0)
    impact = "High" if connected_users > 500 else ("Medium" if connected_users > 150 else "Low")
    priority = "Urgent" if state["primary_probability"] > 0.7 else ("High" if state["primary_probability"] > 0.4 else "Routine")

    report = f"""EXECUTIVE SUMMARY
Tower {state['tower_id']} shows a {state['primary_probability']*100:.0f}% probability of outage
within {state['primary_horizon']}, confidence {state['primary_confidence']*100:.0f}%. Root cause
assessed as {state['root_cause_category']}.

ROOT CAUSE
{state['root_cause_reasoning']}

TIMELINE
Snapshot analyzed at {row.get('timestamp')}. See failure_events.csv for exact historical windows.

EVIDENCE
{chr(10).join(f'- {k}: {v}' for k, v in state['evidence'].items())}

CONFIDENCE (all horizons)
{pred_lines}

KNOWLEDGE BASE REFERENCES
{knowledge_lines or 'None retrieved.'}

HISTORICAL PRECEDENT
{state['historical_context']}

RECOMMENDED ACTIONS
{state['recommendation']}

ESTIMATED CUSTOMER IMPACT
{impact} ({connected_users} connected users at time of snapshot).

SUGGESTED MAINTENANCE PRIORITY
{priority}"""

    return {"final_report": report}


def finish_agent(state: AgentState) -> dict:
    report = f"""EXECUTIVE SUMMARY
Tower {state['tower_id']} shows a {state['primary_probability']*100:.0f}% probability of outage
within {state['primary_horizon']} -- below the {state['threshold']*100:.0f}% action threshold.
No further analysis triggered.

EVIDENCE (for reference)
{chr(10).join(f'- {k}: {v}' for k, v in state['evidence'].items())}

SUGGESTED MAINTENANCE PRIORITY
Routine monitoring only."""
    return {"final_report": report}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("monitoring", monitoring_agent)
    graph.add_node("decision", decision_agent_node)
    graph.add_node("root_cause", root_cause_agent)
    graph.add_node("knowledge", knowledge_agent)
    graph.add_node("historical", historical_events_agent)
    graph.add_node("recommendation", recommendation_agent)
    graph.add_node("report", report_agent)
    graph.add_node("finish", finish_agent)

    graph.set_entry_point("monitoring")
    graph.add_edge("monitoring", "decision")

    graph.add_conditional_edges(
        "decision", route_after_decision,
        {"root_cause": "root_cause", "finish": "finish"},
    )

    # Fan-out: both run in parallel off root_cause
    graph.add_edge("root_cause", "knowledge")
    graph.add_edge("root_cause", "historical")
    # Fan-in: recommendation waits for both branches to complete
    graph.add_edge("knowledge", "recommendation")
    graph.add_edge("historical", "recommendation")

    graph.add_edge("recommendation", "report")
    graph.add_edge("report", END)
    graph.add_edge("finish", END)

    return graph.compile()


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tower", default=None)
    parser.add_argument("--backend", default="lightgbm", choices=["lightgbm", "xgboost", "random_forest"])
    parser.add_argument("--provider", default="rule_based",
                         choices=["rule_based", "anthropic", "openai", "ollama"])
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="Probability threshold at --primary-horizon that triggers full analysis.")
    parser.add_argument("--primary-horizon", default="3h",
                         choices=[label for label, _ in PREDICTION_HORIZONS],
                         help="Which horizon's probability drives the threshold branch (3h was the "
                              "sweet-spot horizon in earlier evaluation, hence the default).")
    args = parser.parse_args()

    print(f"Loading {DATA_PATH} ...")
    df = pd.read_parquet(DATA_PATH)
    tower_id = args.tower or df["tower_id"].iloc[0]

    tdf = df[df["tower_id"] == tower_id].sort_values("timestamp")
    tdf = da.add_trend_features(tdf)
    degraded_rows = tdf[tdf["is_degraded"] == 1]
    row_series = degraded_rows.iloc[len(degraded_rows) // 2] if len(degraded_rows) else tdf.iloc[-1]
    row = row_series.to_dict()
    row["timestamp"] = str(row["timestamp"])

    initial_state: AgentState = {
        "tower_id": tower_id,
        "row": row,
        "evidence": {},
        "predictions": {},
        "primary_probability": 0.0,
        "primary_confidence": 0.0,
        "threshold": args.threshold,
        "root_cause_category": None,
        "root_cause_reasoning": None,
        "knowledge_context": None,
        "historical_context": None,
        "recommendation": None,
        "final_report": None,
        "provider_name": args.provider,
        "backend": args.backend,
        "primary_horizon": args.primary_horizon,
    }

    print(f"Running orchestration for tower {tower_id} "
          f"(backend={args.backend}, provider={args.provider}, threshold={args.threshold})...\n")
    app = build_graph()
    final_state = app.invoke(initial_state, config={"recursion_limit": 50})

    print("=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(final_state["final_report"])


if __name__ == "__main__":
    main()

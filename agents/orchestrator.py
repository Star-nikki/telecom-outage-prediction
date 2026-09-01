"""
LangGraph Orchestrator -- turns the four conceptual monitoring agents into
real, independently-callable steps, coordinated by an LLM supervisor that
decides whether it has enough evidence or needs to requery an agent for
more context before concluding.

WHERE TO SAVE THIS FILE:
    <your project folder>\\telecom_sim_v2\\agents\\orchestrator.py
    (alongside decision_agent.py and llm_providers.py -- it imports both)

ARCHITECTURE
------------
  coordinator --> tower_health --> coordinator --> traffic --> coordinator
              --> weather --> coordinator --> user_behavior --> coordinator
              --> decision --> supervisor --(conclude)--> report --> END
                                   |
                                   +--(requery_X)--> agent X --> supervisor
                                        (loops up to --max-requeries times)

The coordinator dispatches each of the 4 agents exactly once on the first
pass (tracked via a pending_agents queue in state). Once all 4 have reported
and the Decision Agent has run, the Supervisor takes over: it's given every
agent's evidence plus the multi-horizon predictions, and asks an LLM (or the
offline rule-based fallback) to decide whether the evidence is sufficient or
whether one agent should be asked again with a wider lookback window. This
is the genuinely "agentic" part -- the LLM chooses which agent to call and
whether another look is needed, not a fixed pipeline.

USAGE (run from the agents/ folder, after decision_agent.py --mode train
has already produced trained models in agents/models/):

    pip install langgraph --break-system-packages

    python orchestrator.py --tower T0000
    python orchestrator.py --tower T0000 --provider anthropic --backend lightgbm
    python orchestrator.py --tower T0000 --max-requeries 3
"""

import argparse
import os
import re
import sys
from typing import TypedDict, List, Optional

import joblib
import pandas as pd
from langgraph.graph import StateGraph, END

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import PREDICTION_HORIZONS  # noqa: E402
from llm_providers import get_provider  # noqa: E402
import decision_agent as da  # noqa: E402  -- reuses add_trend_features, predict_with_confidence, etc.

DATA_PATH = "../data/telecom_timeseries.parquet"
MODEL_DIR = "models"
AGENT_NAMES = ["tower_health", "traffic", "weather", "user_behavior"]


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    tower_id: str
    row: dict                      # the row being analyzed, as a plain dict
    evidence: dict                 # agent_name -> evidence string
    extended: dict                 # agent_name -> bool (has this agent been requeried?)
    pending_agents: List[str]      # queue for the initial dispatch pass
    predictions: dict              # horizon_label -> {probability, confidence}
    supervisor_notes: List[str]    # log of every supervisor decision + reason
    requery_count: int
    max_requeries: int
    next_action: Optional[str]     # set by supervisor: "requery_<agent>" or "conclude"
    provider_name: str
    final_report: Optional[str]


# ---------------------------------------------------------------------------
# The four monitoring agents -- each is an independent function, callable
# on its own and callable AGAIN with more context if the supervisor asks.
# ---------------------------------------------------------------------------
def _trend(row: dict, col: str, hours: int) -> Optional[float]:
    key = f"{col}_delta_{hours}h"
    return row.get(key)


def tower_health_agent(state: AgentState) -> AgentState:
    row = state["row"]
    extended = state["extended"].get("tower_health", False)

    battery, voltage, rsrp = row["battery_level_pct"], row["grid_voltage"], row["rsrp_dbm"]
    lines = [f"Battery {battery:.0f}%, grid voltage {voltage:.0f}V, RSRP {rsrp:.0f} dBm."]

    if battery < 50 or voltage < 210:
        lines.append("Power supply is degrading.")
    if extended:
        # requery = look further back for a fuller trend, not just the 6h window
        d1 = _trend(row, "battery_level_pct", 1)
        d6 = _trend(row, "battery_level_pct", 6)
        if d1 is not None and d6 is not None:
            lines.append(f"Extended look: battery changed {d1:+.1f} over 1h and {d6:+.1f} over 6h "
                         f"-- {'accelerating decline' if abs(d1) > abs(d6) / 6 * 1.5 else 'steady trend'}.")

    state["evidence"]["tower_health"] = " ".join(lines)
    return state


def traffic_agent(state: AgentState) -> AgentState:
    row = state["row"]
    extended = state["extended"].get("traffic", False)

    util, packet_loss, throughput, latency = (
        row["utilization_pct"], row["packet_loss_pct"], row["throughput_mbps"], row["latency_ms"]
    )
    lines = [f"Utilization {util:.0f}%, packet loss {packet_loss:.1f}%, "
             f"throughput {throughput:.0f} Mbps, latency {latency:.0f} ms."]

    if util > 80 or packet_loss > 5:
        lines.append("Network is congested.")
    if extended:
        d24 = _trend(row, "utilization_pct", 24)
        if d24 is not None:
            trend_desc = "climbing steadily" if d24 > 5 else ("declining" if d24 < -5 else "flat")
            lines.append(f"Extended look: utilization has been {trend_desc} over the last 24h ({d24:+.1f} pts) "
                         f"-- {'consistent with a slow capacity build-up' if d24 > 5 else 'no slow-creep pattern'}.")

    state["evidence"]["traffic"] = " ".join(lines)
    return state


def weather_agent(state: AgentState) -> AgentState:
    row = state["row"]
    extended = state["extended"].get("weather", False)

    storm, rainfall, wind = row["storm_flag"], row["rainfall_mm"], row["wind_kmh"]
    lines = [f"Storm flag={'ACTIVE' if storm else 'clear'}, rainfall {rainfall:.1f}mm, wind {wind:.0f}km/h."]

    if storm == 1 or rainfall > 5:
        lines.append("Storm probability high; weather-driven degradation plausible.")
    if extended:
        rsrp_trend = _trend(row, "rsrp_dbm", 6)
        if rsrp_trend is not None:
            lines.append(f"Extended look: signal strength has moved {rsrp_trend:+.1f} dBm over 6h -- "
                         f"{'consistent with storm-driven attenuation' if rsrp_trend < -3 and storm else 'not clearly storm-linked'}.")

    state["evidence"]["weather"] = " ".join(lines)
    return state


def user_behavior_agent(state: AgentState) -> AgentState:
    row = state["row"]
    load_mult = row["load_multiplier"]
    lines = [f"Load multiplier {load_mult:.2f}."]
    if load_mult > 1.5:
        lines.append("Unusual demand spike -- check for a named mobility event (festival/match/concert/emergency).")
    else:
        lines.append("Normal demand pattern, not a behavior-driven anomaly.")
    state["evidence"]["user_behavior"] = " ".join(lines)
    return state


AGENT_FUNCTIONS = {
    "tower_health": tower_health_agent,
    "traffic": traffic_agent,
    "weather": weather_agent,
    "user_behavior": user_behavior_agent,
}


# ---------------------------------------------------------------------------
# Coordinator: dispatches each agent once on the first pass
# ---------------------------------------------------------------------------
def coordinator(state: AgentState) -> AgentState:
    return state  # pure router; see route_from_coordinator below


def route_from_coordinator(state: AgentState) -> str:
    if state["pending_agents"]:
        return state["pending_agents"][0]
    return "decision"


def make_agent_wrapper(agent_name: str):
    """Wraps an agent function so it also pops itself off the pending queue
    after running (only relevant during the initial dispatch pass)."""
    def wrapped(state: AgentState) -> AgentState:
        state = AGENT_FUNCTIONS[agent_name](state)
        if state["pending_agents"] and state["pending_agents"][0] == agent_name:
            state["pending_agents"] = state["pending_agents"][1:]
        return state
    return wrapped


def route_after_agent(state: AgentState) -> str:
    """After an agent runs: if we're still in the initial dispatch pass
    (pending_agents non-empty or just emptied), go back to the coordinator
    to dispatch the next one. If this was a targeted requery (pending_agents
    already empty before this call), go straight back to the supervisor."""
    if state["pending_agents"]:
        return "coordinator"
    if state["next_action"] and state["next_action"].startswith("requery"):
        return "supervisor"
    return "coordinator"  # end of initial pass -- coordinator will route to decision


# ---------------------------------------------------------------------------
# Decision node -- loads your trained models, produces multi-horizon predictions
# ---------------------------------------------------------------------------
def decision_node(state: AgentState, backend: str) -> AgentState:
    row = pd.Series(state["row"])
    predictions = {}
    for label, _ in PREDICTION_HORIZONS:
        model_path = os.path.join(MODEL_DIR, f"decision_model_{backend}_{label}.joblib")
        if not os.path.exists(model_path):
            continue
        models = joblib.load(model_path)
        prob, conf = da.predict_with_confidence(models, row)
        predictions[label] = {"probability": prob, "confidence": conf}
    state["predictions"] = predictions
    return state


# ---------------------------------------------------------------------------
# Supervisor -- the genuinely agentic part: LLM decides requery vs conclude
# ---------------------------------------------------------------------------
def build_supervisor_prompt(state: AgentState) -> str:
    evidence_block = "\n".join(f"- {name}: {text}" for name, text in state["evidence"].items())
    pred_block = "\n".join(
        f"- {h}: {p['probability']*100:.0f}% probability, {p['confidence']*100:.0f}% confidence"
        for h, p in state["predictions"].items()
    )
    already_requeried = [a for a, done in state["extended"].items() if done]

    return f"""You are the supervisor of a telecom outage-prediction system, coordinating four
monitoring agents (tower_health, traffic, weather, user_behavior) and a Decision Agent.

Current evidence:
{evidence_block}

Decision Agent predictions by horizon:
{pred_block}

Agents already given an extended second look: {already_requeried or 'none'}
Requeries used so far: {state['requery_count']} / {state['max_requeries']}

Decide ONE of two things:
1. If the evidence is sufficient and consistent enough to explain the prediction, conclude.
2. If evidence from one agent seems incomplete or you'd benefit from a deeper trend look from
   ONE specific agent to resolve an ambiguity, request that agent again (it will return with a
   wider trend window).

Respond in EXACTLY this format, nothing else:
DECISION: <conclude|requery_tower_health|requery_traffic|requery_weather|requery_user_behavior>
REASON: <one sentence explaining why>"""


def parse_supervisor_response(text: str) -> tuple:
    decision_match = re.search(r"DECISION:\s*(\w+)", text)
    reason_match = re.search(r"REASON:\s*(.+)", text)
    decision = decision_match.group(1).strip() if decision_match else "conclude"
    reason = reason_match.group(1).strip() if reason_match else "(no reason parsed)"
    valid = {"conclude"} | {f"requery_{a}" for a in AGENT_NAMES}
    if decision not in valid:
        decision = "conclude"  # safe fallback if the LLM's output didn't match the expected format
    return decision, reason


def supervisor_node(state: AgentState) -> AgentState:
    if state["requery_count"] >= state["max_requeries"]:
        state["next_action"] = "conclude"
        state["supervisor_notes"].append(
            f"Max requeries ({state['max_requeries']}) reached -- concluding regardless."
        )
        return state

    provider = get_provider(state["provider_name"])
    prompt = build_supervisor_prompt(state)

    if state["provider_name"] == "rule_based":
        # Deterministic stand-in (documented as such): requery once if
        # weather looks active but tower_health looks normal (a genuinely
        # ambiguous case worth a second look), otherwise conclude.
        decision, reason = _rule_based_supervisor_decision(state)
    else:
        raw = provider.generate(prompt)
        decision, reason = parse_supervisor_response(raw)

    state["next_action"] = decision
    state["supervisor_notes"].append(f"[{decision}] {reason}")

    if decision.startswith("requery_"):
        agent_name = decision.replace("requery_", "")
        state["extended"][agent_name] = True
        state["pending_agents"] = [agent_name]  # dispatch just this one, targeted
        state["requery_count"] += 1

    return state


def _rule_based_supervisor_decision(state: AgentState) -> tuple:
    """Offline fallback: simple heuristic instead of an LLM call, so the
    graph still runs end-to-end with zero API keys configured."""
    ev = state["evidence"]
    weather_active = "high" in ev.get("weather", "").lower() or "ACTIVE" in ev.get("weather", "")
    tower_ok = "degrading" not in ev.get("tower_health", "").lower()
    if weather_active and tower_ok and not state["extended"].get("weather", False):
        return ("requery_weather",
                "Weather looks active but power supply looks normal -- worth a deeper trend look "
                "before attributing to weather.")
    return "conclude", "Evidence is internally consistent enough to conclude without another look."


# ---------------------------------------------------------------------------
# Report node -- the final structured incident report
# ---------------------------------------------------------------------------
def report_node(state: AgentState) -> AgentState:
    provider_name = state["provider_name"]
    row = state["row"]

    if provider_name == "rule_based":
        state["final_report"] = _rule_based_report(state)
    else:
        provider = get_provider(provider_name)
        evidence_block = "\n".join(f"- {name}: {text}" for name, text in state["evidence"].items())
        pred_block = "\n".join(
            f"- {h}: {p['probability']*100:.0f}% probability, {p['confidence']*100:.0f}% confidence"
            for h, p in state["predictions"].items()
        )
        supervisor_block = "\n".join(state["supervisor_notes"])

        prompt = f"""Generate a full incident report for tower {state['tower_id']} at {row.get('timestamp')}.

Evidence:
{evidence_block}

Predictions by horizon:
{pred_block}

Supervisor reasoning trail:
{supervisor_block}

True root cause (ground truth, for internal reference only -- do not just repeat it, reason to it):
{row.get('root_cause')}

Write the report with EXACTLY these sections, each with a heading:
Executive Summary
Root Cause
Timeline
Evidence
Confidence
Recommended Actions
Estimated Customer Impact
Suggested Maintenance Priority"""
        state["final_report"] = provider.generate(prompt)

    return state


def _rule_based_report(state: AgentState) -> str:
    """Offline fallback report -- fully templated, no LLM needed."""
    row = state["row"]
    ev = state["evidence"]
    preds = state["predictions"]
    best_horizon = max(preds, key=lambda h: preds[h]["probability"]) if preds else None
    top_pred = preds.get(best_horizon, {"probability": 0, "confidence": 0})

    connected_users = row.get("connected_users", 0)
    impact = "High" if connected_users > 500 else ("Medium" if connected_users > 150 else "Low")
    priority = "Urgent" if top_pred["probability"] > 0.7 else ("High" if top_pred["probability"] > 0.4 else "Routine")

    pred_lines = "\n".join(
        f"- {h}: {p['probability']*100:.0f}% probability, {p['confidence']*100:.0f}% confidence"
        for h, p in preds.items()
    )

    return f"""EXECUTIVE SUMMARY
Tower {state['tower_id']} shows a {top_pred['probability']*100:.0f}% probability of outage within
{best_horizon or 'the monitored horizons'}, with {top_pred['confidence']*100:.0f}% model confidence.

ROOT CAUSE
Based on combined evidence: {ev.get('tower_health', '')} {ev.get('weather', '')} {ev.get('traffic', '')}
(Rule-based summary -- for deeper reasoning, run with --provider anthropic/openai/ollama.)

TIMELINE
Snapshot analyzed at {row.get('timestamp')}. See failure_events.csv for exact ramp/outage windows for this tower.

EVIDENCE
- Tower Health: {ev.get('tower_health', 'n/a')}
- Traffic: {ev.get('traffic', 'n/a')}
- Weather: {ev.get('weather', 'n/a')}
- User Behavior: {ev.get('user_behavior', 'n/a')}

CONFIDENCE
{pred_lines}

RECOMMENDED ACTIONS
- Dispatch inspection if probability exceeds 70% at any horizon under 6h.
- Cross-check with mobility_events.csv in case a named event, not a fault, explains any traffic spike.

ESTIMATED CUSTOMER IMPACT
{impact} ({connected_users} connected users at time of snapshot).

SUGGESTED MAINTENANCE PRIORITY
{priority}

Supervisor reasoning trail:
{chr(10).join(state['supervisor_notes'])}"""


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_graph(backend: str):
    graph = StateGraph(AgentState)

    graph.add_node("coordinator", coordinator)
    for name in AGENT_NAMES:
        graph.add_node(name, make_agent_wrapper(name))
    graph.add_node("decision", lambda state: decision_node(state, backend))
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("coordinator")

    graph.add_conditional_edges(
        "coordinator", route_from_coordinator,
        {**{name: name for name in AGENT_NAMES}, "decision": "decision"},
    )
    for name in AGENT_NAMES:
        graph.add_conditional_edges(
            name, route_after_agent,
            {"coordinator": "coordinator", "supervisor": "supervisor"},
        )
    graph.add_edge("decision", "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["next_action"] if state["next_action"] != "conclude" else "conclude",
        {**{f"requery_{a}": a for a in AGENT_NAMES}, "conclude": "report"},
    )
    graph.add_edge("report", END)

    return graph.compile()


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tower", default=None)
    parser.add_argument("--backend", default="lightgbm", choices=["lightgbm", "xgboost", "random_forest"])
    parser.add_argument("--provider", default="rule_based",
                         choices=["rule_based", "anthropic", "openai", "ollama"])
    parser.add_argument("--max-requeries", type=int, default=2)
    args = parser.parse_args()

    print(f"Loading {DATA_PATH} ...")
    df = pd.read_parquet(DATA_PATH)
    tower_id = args.tower or df["tower_id"].iloc[0]

    tdf = df[df["tower_id"] == tower_id].sort_values("timestamp")
    tdf = da.add_trend_features(tdf)
    degraded_rows = tdf[tdf["is_degraded"] == 1]
    row_series = degraded_rows.iloc[len(degraded_rows) // 2] if len(degraded_rows) else tdf.iloc[-1]
    row = row_series.to_dict()
    row["timestamp"] = str(row["timestamp"])  # keep state JSON/print friendly

    initial_state: AgentState = {
        "tower_id": tower_id,
        "row": row,
        "evidence": {},
        "extended": {a: False for a in AGENT_NAMES},
        "pending_agents": list(AGENT_NAMES),
        "predictions": {},
        "supervisor_notes": [],
        "requery_count": 0,
        "max_requeries": args.max_requeries,
        "next_action": None,
        "provider_name": args.provider,
        "final_report": None,
    }

    print(f"Running orchestration for tower {tower_id} (backend={args.backend}, provider={args.provider})...\n")
    app = build_graph(args.backend)
    final_state = app.invoke(initial_state, config={"recursion_limit": 50})

    print("="*70)
    print("SUPERVISOR REASONING TRAIL")
    print("="*70)
    for note in final_state["supervisor_notes"]:
        print(f"  {note}")

    print("\n" + "="*70)
    print("FINAL REPORT")
    print("="*70)
    print(final_state["final_report"])


if __name__ == "__main__":
    main()

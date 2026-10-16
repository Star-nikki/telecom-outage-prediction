"""
Orchestrator v3 -- Agentic RAG architecture: Monitoring -> Decision ->
threshold branch -> Root Cause -> {Knowledge Agent, Historical Events
Agent} running in PARALLEL -> Recommendation (LLM) -> Report.

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
              +--------+--------+
              |                 |
              v                 v
      Knowledge Agent    Historical Events
      (hybrid RAG over    Agent (this tower's
      SOP corpus, via     own recorded incidents,
      advanced_rag.py)    via failure_events.csv)
              |                 |
              +--------+--------+
                       |
                       v
             Recommendation Agent (LLM)
                       |
                       v
                Report Agent --> END

Each node has a genuinely distinct responsibility and the graph only
proceeds past Decision if the threshold condition is met -- this isn't
"function A calls function B calls function C." The Root Cause Agent makes
a real categorization judgment. The Knowledge and Historical Events agents
are two INDEPENDENT retrieval agents over two DIFFERENT sources (general
SOP knowledge vs. this specific tower's own incident history) running as
true parallel LangGraph branches, not one calling the other -- this is what
makes the RAG layer "agentic" rather than a single retrieval call. The
Recommendation Agent is LLM-prompted with both retrieved contexts, not
hardcoded if/else logic.

WHERE TO SAVE THIS FILE:
    <your project folder>\\telecom_sim_v2\\agents\\orchestrator_v3.py
    (alongside decision_agent.py, llm_providers.py, advanced_rag.py --
    imports all three. knowledge_base.py is no longer imported by this file
    but can stay on disk for reference/comparison.)

USAGE (run from the agents/ folder, after decision_agent.py --mode train
has produced trained models in agents/models/):

    pip install langgraph scikit-learn rank_bm25 --break-system-packages

    python orchestrator_v3.py --tower T0000
    python orchestrator_v3.py --tower T0000 --provider anthropic
    python orchestrator_v3.py --tower T0000 --threshold 0.6 --primary-horizon 3h

CANONICAL VERSIONS IN USE (per project decision): orchestrator_v3.py (this
file, rule-based/deterministic engine) and orchestrator_llm.py (LLM-layered
version built on top of it). No cleanup/deprecation pass needed here.
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
import advanced_rag as arag  # noqa: E402  -- upgraded from knowledge_base.py's plain
                                            # TF-IDF retrieval to hybrid BM25+TF-IDF,
                                            # contextual queries, parent-child chunking,
                                            # and reranking. Interface-compatible: still
                                            # returns a list of {id, failure_type, text,
                                            # relevance_score} dicts, so nothing downstream
                                            # (recommendation_agent, report_agent) changes.
from meta_confidence_prediction import predict_with_meta_confidence  # noqa: E402
                                            # -- Priority 5: wraps predict_with_confidence_calibrated()
                                            # (still runs underneath) and ADDITIONALLY returns an
                                            # alert-only meta-confidence score for horizons where
                                            # meta_confidence_diagnostics.py found it genuinely beats
                                            # existing confidence on alerts specifically (currently 3h
                                            # only). Returns None for meta_confidence anywhere that
                                            # wasn't validated -- see that file's docstring.
                                            # -- Priority 4: opt-in probability calibration.
                                            # Reads models/calibration_recommendation.json
                                            # (produced by calibration_experiment.py) and
                                            # applies Platt/Isotonic ONLY where it was found
                                            # to genuinely help; falls back to raw, unchanged
                                            # behavior if that file doesn't exist yet.

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "telecom_timeseries.parquet"
EVENTS_PATH = PROJECT_ROOT / "data" / "failure_events.csv"
MODEL_DIR = PROJECT_ROOT / "agents" / "models"
FAILURE_CATEGORIES = ["hardware_power_failure", "congestion_collapse",
                       "weather_induced_degradation", "capacity_creep"]


def pick_demo_row(tdf: pd.DataFrame, tower_id: str, hours_before_outage: float = 3.0,
                   failure_type_filter: str = None):
    """
    Picks a snapshot GENUINELY BEFORE an outage, not the midpoint of the
    whole degraded span (which often lands inside or at the edge of the
    actual outage itself).

    IMPORTANT: different failure types have very different lead times by
    design (see config.py FAILURE_TYPES) -- congestion_collapse is only
    1-4h, hardware_power_failure is 3-6h, weather_induced_degradation is
    2-5h, but capacity_creep is 24-72h. Testing a short-lead-time failure
    (like congestion_collapse) at --hours-before 12 or 24 will correctly
    show ~0% every time, because there's genuinely no signal that far out
    for that failure type -- that's not a bug. Pass failure_type_filter to
    deliberately pick a longer-ramp event (e.g. 'capacity_creep') if you
    want to see gradual probability increase across a wider range of
    --hours-before values.
    """
    if os.path.exists(EVENTS_PATH):
        events = pd.read_csv(EVENTS_PATH, parse_dates=["ramp_start", "outage_start", "outage_end"])
        tower_events = events[events["tower_id"] == tower_id]
        if failure_type_filter:
            filtered = tower_events[tower_events["failure_type"] == failure_type_filter]
            if len(filtered) == 0:
                available = tower_events["failure_type"].unique().tolist()
                print(f"No {failure_type_filter} event found for {tower_id}. "
                      f"Available types for this tower: {available or 'none'}. Using first available event instead.")
            else:
                tower_events = filtered

        if len(tower_events):
            event = tower_events.iloc[0]
            lead_time_available = (event["outage_start"] - event["ramp_start"]).total_seconds() / 3600
            if hours_before_outage > lead_time_available:
                print(f"NOTE: this event's actual lead time was only {lead_time_available:.1f}h -- "
                      f"requesting {hours_before_outage}h before means you're asking for a snapshot "
                      f"BEFORE the ramp even starts, so ~0% probability here is correct, not a bug.")
            target_time = event["outage_start"] - pd.Timedelta(hours=hours_before_outage)
            tdf = tdf.copy()
            tdf["_time_diff"] = (tdf["timestamp"] - target_time).abs()
            row = tdf.loc[tdf["_time_diff"].idxmin()]
            print(f"Demo snapshot: {hours_before_outage}h before outage_start "
                  f"({event['outage_start']}), actual snapshot time: {row['timestamp']}, "
                  f"true failure type: {event['failure_type']} (this event's actual lead time: "
                  f"{lead_time_available:.1f}h)")
            return row

    print(f"No recorded failure event for {tower_id} -- falling back to midpoint-of-degraded selection "
          f"(results may look like a mid-outage snapshot rather than a genuine early warning).")
    degraded_rows = tdf[tdf["is_degraded"] == 1]
    return degraded_rows.iloc[len(degraded_rows) // 2] if len(degraded_rows) else tdf.iloc[-1]


def list_events_for_tower(tower_id: str):
    """Helper: prints every recorded failure event for a tower, with lead
    time, so you can pick a good candidate for a --hours-before demo."""
    if not os.path.exists(EVENTS_PATH):
        print(f"No events file found at {EVENTS_PATH}")
        return
    events = pd.read_csv(EVENTS_PATH, parse_dates=["ramp_start", "outage_start", "outage_end"])
    tower_events = events[events["tower_id"] == tower_id]
    if len(tower_events) == 0:
        print(f"No recorded events for {tower_id}")
        return
    print(f"Events for {tower_id}:")
    for _, e in tower_events.iterrows():
        lead_h = (e["outage_start"] - e["ramp_start"]).total_seconds() / 3600
        print(f"  {e['failure_type']:>28} | lead time {lead_h:5.1f}h | outage_start={e['outage_start']}")


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
    primary_calibration_method: Optional[str]
    primary_meta_confidence: Optional[float]
    threshold: float
    root_cause_category: Optional[str]
    root_cause_reasoning: Optional[str]
    knowledge_context: Optional[List[dict]]
    historical_context: Optional[dict]
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
    th_note = f"Battery {battery:.0f}%, grid voltage {voltage:.0f}V, RSRP {rsrp:.0f} dBm."
    if battery < 50 or voltage < 210:
        th_note += " Power supply degrading."
    if rsrp < -95:
        th_note += " Rapid signal attenuation."
    if "degrading" not in th_note and "attenuation" not in th_note:
        th_note += " Power and signal normal."
    evidence["tower_health"] = th_note

    util, packet_loss, throughput, latency = (
        row["utilization_pct"], row["packet_loss_pct"], row["throughput_mbps"], row["latency_ms"]
    )
    traffic_note = (
        f"Utilization {util:.0f}%, packet loss {packet_loss:.1f}%, throughput {throughput:.0f} Mbps, "
        f"latency {latency:.0f} ms."
    )
    # FIX #2: don't lump "high packet loss" and "high utilization" into one
    # generic "Congested" label -- they point to different root causes.
    # High utilization alone (or with packet loss) = real congestion.
    # High packet loss WITHOUT high utilization = something else is going
    # on (radio interference, hardware fault, weather) -- flag it as such
    # rather than mislabeling it congestion.
    if util > 80:
        traffic_note += " Network congestion (utilization-driven)."
    elif packet_loss > 15 and util <= 50:
        traffic_note += (" Elevated packet loss WITHOUT high utilization -- not simple congestion; "
                          "check radio quality / interference / hardware before assuming capacity issue.")
    elif packet_loss > 5:
        traffic_note += " Mild packet loss, utilization not yet congested."
    else:
        traffic_note += " Normal."
    evidence["traffic"] = traffic_note

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
def _enforce_monotonic_across_horizons(predictions: dict, horizon_order: list) -> dict:
    """
    outage_in_next_15m is a SUBSET of outage_in_next_3h, which is a subset
    of outage_in_next_24h -- if an outage happens within 15 minutes, it
    necessarily also happened within 3 hours. So P(15m) <= P(30m) <= ... <=
    P(24h) is a genuine logical constraint on the underlying events, not
    just a preference. Each horizon's model (and each horizon's Platt/
    Isotonic calibrator, from Priority 4) is fit completely independently,
    so nothing enforces that agreement between horizons -- this pass fixes
    it post-hoc via a simple running-maximum in horizon order (the standard
    pool-adjacent-violators-style fix for enforcing monotonicity on an
    already-computed sequence). Confidence is untouched -- this only
    touches probability, and only adjusts a horizon's value UP to match a
    smaller horizon it violated, never down.
    """
    running_max = 0.0
    for label in horizon_order:
        if label not in predictions:
            continue
        original = predictions[label]["probability"]
        adjusted = max(original, running_max)
        predictions[label]["probability"] = adjusted
        predictions[label]["monotonic_adjusted"] = (adjusted != original)
        if adjusted != original:
            predictions[label]["probability_before_monotonic_fix"] = original
        running_max = adjusted
    return predictions


def decision_agent_node(state: AgentState) -> dict:
    row = pd.Series(state["row"])
    predictions = {}
    horizon_order = [label for label, _ in PREDICTION_HORIZONS]
    for label, _ in PREDICTION_HORIZONS:
        model_path = os.path.join(MODEL_DIR, f"decision_model_{state['backend']}_{label}.joblib")
        if not os.path.exists(model_path):
            continue
        models = joblib.load(model_path)
        prob, conf, method, meta_conf = predict_with_meta_confidence(
            models, row, state["backend"], label, model_dir=MODEL_DIR)
        predictions[label] = {"probability": prob, "confidence": conf, "calibration_method": method,
                               "meta_confidence": meta_conf}

    predictions = _enforce_monotonic_across_horizons(predictions, horizon_order)

    primary = predictions.get(state["primary_horizon"],
                               {"probability": 0.0, "confidence": 0.0, "calibration_method": "raw",
                                "meta_confidence": None})
    return {
        "predictions": predictions,
        "primary_probability": primary["probability"],
        "primary_confidence": primary["confidence"],
        "primary_calibration_method": primary["calibration_method"],
        "primary_meta_confidence": primary.get("meta_confidence"),
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
        category, reasoning = _rule_based_root_cause(state["row"])
    else:
        provider = get_provider(provider_name)
        evidence_block = "\n".join(f"- {k}: {v}" for k, v in evidence.items())
        prompt = f"""Evidence from telecom monitoring agents for tower {state['tower_id']}:
{evidence_block}

Predicted outage probability at {state['primary_horizon']}: {state['primary_probability']*100:.1f}%

Which single category best explains this, and why? Choose exactly one:
hardware_power_failure, congestion_collapse, weather_induced_degradation, capacity_creep

Respond in EXACTLY this format:
CATEGORY: <one of the four categories above>
REASONING: <2-3 sentences explaining which signals support this and which don't>"""
        raw = provider.generate(prompt)
        category, reasoning = _parse_root_cause(raw)

    return {"root_cause_category": category, "root_cause_reasoning": reasoning}


def _rule_based_root_cause(row: dict) -> tuple:
    """
    Multi-signal scoring across all 4 categories.

    FIX: congestion_collapse no longer requires utilization > 60 to score AT
    ALL -- severe packet loss and latency are direct congestion evidence on
    their own. Previously, a case with packet_loss=34%, latency=172ms, but
    utilization=56% (just under the old 60% gate) scored ZERO for
    congestion_collapse and lost to capacity_creep purely on the 24h trend.

    FIX 2 (root cause of the T0002 misclassification -- superseded below):
    an earlier version of this function used the ratio of the 6h
    utilization trend to the 24h utilization trend as the congestion-vs-creep
    discriminator (concentrated-in-last-6h => acute congestion; spread evenly
    => genuine creep). That heuristic is wrong: capacity_creep in this
    simulator accelerates as it approaches saturation. Traced against
    T0002's actual pre-outage trajectory, packet_loss/latency climb smoothly
    and monotonically the entire 26h+ window, but utilization itself is
    noisy and dipped earlier before its final climb -- so its last-6h move
    (+35.4pts) was bigger than its full-24h move (+25.3pts). That made the
    6h/24h ratio (1.4) look like an "acute recent ramp," which is backwards:
    it's the tail end of a genuine multi-day creep, not a fresh spike.
    Utilization is too noisy, over too short a lookback, to reliably tell
    acute from gradual.

    FIX 3 (the actual generalizable discriminator): what distinguishes
    "just started" from "has been building for a while" is not how much
    utilization moved, but how long packet_loss/latency have already been
    elevated. add_trend_features() computes a 6h rolling MIN for every base
    column (not just deltas), so packet_loss_pct_min_6h and
    latency_ms_min_6h give the lowest reading in the trailing 6h window:
      - congestion_collapse (checked against T0000's and T0001's actual
        pre-outage trajectories): packet loss/latency sit near baseline for
        the entire lookback and then rocket up in the last few hours --
        min_6h stays near baseline right up to the snapshot, so
        min_6h / current is tiny (T0001: 0.003; T0000: 0.01).
      - capacity_creep (checked the same way against T0002): packet loss/
        latency have been smoothly climbing for 24h+ before the snapshot,
        so the current reading was already mostly reached 6h ago --
        min_6h / current is large (T0002: 0.85).
    This ratio is a stock rolling feature available for any tower/event, not
    anything specific to T0002, and it measures onset duration directly
    instead of inferring it indirectly from a noisy metric's short-window
    delta.

    Practically: severe packet_loss/latency are still congestion evidence
    (FIX 1 above is preserved), but when the onset-acuity check shows that
    severity has clearly been sustained across the whole 6h window rather
    than appearing suddenly, that score is damped for congestion_collapse
    and credited to capacity_creep instead -- late-stage capacity_creep IS
    supposed to show high packet loss and latency by the time it's close to
    outage, so "packet loss is high" was never evidence that should have
    ruled it out.
    """
    scores, signals_fired = _score_root_cause_signals(row)
    best_category = max(scores, key=lambda c: (scores[c], c == "capacity_creep"))

    if scores[best_category] == 0:
        return ("capacity_creep",
                "No combination of signals reached a confident threshold for any category; defaulting to the "
                "mildest explanation. This snapshot may be too early in the ramp for a confident diagnosis.")

    reasoning = (f"Category selected on combined-signal score {scores[best_category]:.1f}: "
                 + "; ".join(signals_fired[best_category]) + ".")

    runner_up = sorted(((c, s) for c, s in scores.items() if c != best_category), key=lambda x: -x[1])[0]
    if runner_up[1] > 0 and runner_up[1] >= scores[best_category] - 1:
        reasoning += (f" (Note: {runner_up[0]} also scored {runner_up[1]:.1f} -- evidence is somewhat "
                      f"ambiguous between the two.)")

    return best_category, reasoning


def _score_root_cause_signals(row: dict) -> tuple:
    """
    The actual multi-signal scorer, factored out of _rule_based_root_cause()
    so the diagnostic helper (diagnose_root_cause, below) can print every
    category's score and contributing signals without duplicating the
    scoring logic.
    """
    battery = row.get("battery_level_pct", 100)
    voltage = row.get("grid_voltage", 230)
    rsrp = row.get("rsrp_dbm", -80)
    util = row.get("utilization_pct", 0)
    packet_loss = row.get("packet_loss_pct", 0)
    latency = row.get("latency_ms", 20)
    storm = row.get("storm_flag", 0)
    rainfall = row.get("rainfall_mm", 0)
    util_trend_24h = row.get("utilization_pct_delta_24h") or 0
    packet_loss_min_6h = row.get("packet_loss_pct_min_6h")
    latency_min_6h = row.get("latency_ms_min_6h")

    scores = {c: 0.0 for c in FAILURE_CATEGORIES}
    signals_fired = {c: [] for c in FAILURE_CATEGORIES}

    # --- hardware_power_failure ---
    # FIX (Priority 3 testing exposed this): voltage<210 was previously
    # worth only +3, while congestion_collapse's severity+acute-onset score
    # could reach into the low teens from the SAME packet_loss/latency
    # spike -- a power fault degrades RF/backhaul too, so congestion always
    # out-scored hardware even when voltage clearly showed a power fault
    # (e.g. voltage=178V). Checked against a 687-snapshot sample spanning
    # every tower and all 4 categories: voltage<210 occurred in 176/176
    # hardware_power_failure snapshots and ZERO snapshots of any other
    # category (min voltage elsewhere was 223.76V) -- this is a clean,
    # 100%-specific signal, so it's scored accordingly, and it now also
    # gates the congestion damping below.
    hardware_strong = battery < 50 or voltage < 210
    if hardware_strong:
        scores["hardware_power_failure"] += 5
        signals_fired["hardware_power_failure"].append(
            f"battery={battery:.0f}%, voltage={voltage:.0f}V -- below the power-fault thresholds "
            f"(battery<50%, voltage<210V), consistent with direct power degradation")
    if rsrp < -95 and packet_loss > 10:
        scores["hardware_power_failure"] += 2
        signals_fired["hardware_power_failure"].append(
            f"RSRP={rsrp:.0f}dBm combined with packet_loss={packet_loss:.1f}% (signal/hardware combo)")

    # --- weather_induced_degradation ---
    # FIX (same class of bug as hardware, above): checked against the same
    # 687-snapshot sample, rsrp<-90 alone (independent of the storm flag)
    # occurred in 164/169 weather_induced_degradation snapshots and ZERO
    # snapshots of any other category -- also a clean, near-fully-specific
    # signal on its own, not just as a bonus nested under "storm active".
    # storm/rainfall remains a separate, weaker corroborating signal.
    weather_strong = rsrp < -90
    if weather_strong:
        scores["weather_induced_degradation"] += 5
        signals_fired["weather_induced_degradation"].append(
            f"RSRP={rsrp:.0f}dBm severely degraded (below -90dBm), consistent with storm-driven "
            f"RF interference rather than a network-side fault")
    if storm == 1 or rainfall > 5:
        scores["weather_induced_degradation"] += 1.5
        signals_fired["weather_induced_degradation"].append(f"storm/rainfall active (rainfall={rainfall:.1f}mm)")

    # --- onset-acuity check: has the elevated packet loss/latency been
    # sustained across the whole trailing 6h window, or did it just start?
    # Computed once and used below to route the congestion severity score
    # to whichever of congestion_collapse/capacity_creep it actually supports.
    ACUTE_RATIO = 0.15     # min_6h below 15% of current => fresh onset
    SUSTAINED_RATIO = 0.5  # min_6h above 50% of current => already elevated 6h ago
    acute_count = 0
    sustained_count = 0
    for label, current_val, min_6h, severity_gate in (
        ("packet_loss", packet_loss, packet_loss_min_6h, 15),
        ("latency", latency, latency_min_6h, 100),
    ):
        if min_6h is None or current_val is None or current_val <= severity_gate:
            continue
        ratio = min_6h / current_val if current_val else 1.0
        if ratio < ACUTE_RATIO:
            acute_count += 1
            if not (hardware_strong or weather_strong):
                signals_fired["congestion_collapse"].append(
                    f"{label} was near baseline 6h ago ({min_6h:.1f}) vs current {current_val:.1f} "
                    f"-- sudden onset within the last 6h, not a gradual buildup")
        elif ratio > SUSTAINED_RATIO:
            sustained_count += 1
            signals_fired["capacity_creep"].append(
                f"{label} was already {min_6h:.1f} six hours ago, close to the current {current_val:.1f} "
                f"-- this elevation has been sustained, not a fresh spike")

    # --- congestion_collapse: packet loss/latency/utilization severity.
    # Still counts on its own (FIX 1), but damped when the onset-acuity
    # check above shows the elevation has clearly been sustained rather
    # than sudden -- that pattern belongs to capacity_creep, even though the
    # raw numbers look just as severe. ALSO damped (new) when hardware_strong
    # or weather_strong fired: a sudden packet_loss/latency spike is just as
    # consistent with "the power just failed" or "a storm just hit" as with
    # genuine congestion, so it shouldn't be credited to congestion_collapse
    # when a clean, independent competing cause is already present -- and the
    # acute-onset bonus (which specifically rewards "just started") is
    # suppressed entirely in that case rather than merely damped, since a
    # sudden onset is exactly what a fresh hardware/weather event looks like.
    if hardware_strong or weather_strong:
        severity_damping = 0.2
        acute_count = 0
    else:
        severity_damping = max(0.0, 1.0 - 0.3 * sustained_count)
    congestion_severity = 0.0
    if packet_loss > 15:
        congestion_severity += 2
        signals_fired["congestion_collapse"].append(f"packet_loss={packet_loss:.1f}%")
    if packet_loss > 30:
        congestion_severity += 2
        signals_fired["congestion_collapse"].append("packet loss is severe")
    if latency > 100:
        congestion_severity += 2
        signals_fired["congestion_collapse"].append(f"latency={latency:.0f}ms")
    if latency > 150:
        congestion_severity += 1
        signals_fired["congestion_collapse"].append("latency is severely elevated")
    if util > 60:
        congestion_severity += 1
        signals_fired["congestion_collapse"].append(f"utilization={util:.0f}% also elevated")
    scores["congestion_collapse"] += congestion_severity * severity_damping
    scores["congestion_collapse"] += 3 * acute_count

    # --- capacity_creep: sustained growth evidence ---
    if util_trend_24h > 10:
        scores["capacity_creep"] += 1  # weak base credit alone
        signals_fired["capacity_creep"].append(f"24h utilization trend is +{util_trend_24h:.1f}pts")
    scores["capacity_creep"] += 3 * sustained_count
    if packet_loss <= 15 and latency <= 100:
        scores["capacity_creep"] += 1
        signals_fired["capacity_creep"].append("no elevated packet loss/latency alongside the trend yet")

    return scores, signals_fired


def diagnose_root_cause(row: dict, tower_id: str = None) -> None:
    """
    Diagnostic helper: prints every category's score and the signals that
    contributed to it, plus the final prediction. Doesn't change any
    behavior -- just exposes what _rule_based_root_cause() is doing
    internally, for debugging/validating new test cases.

    Usage:
        diagnose_root_cause(row, tower_id="T0002")
    """
    scores, signals_fired = _score_root_cause_signals(row)
    best_category = max(scores, key=lambda c: (scores[c], c == "capacity_creep"))

    if tower_id:
        print(tower_id)
    for category in FAILURE_CATEGORIES:
        print(f"{category}: {scores[category]:.1f}")
    print(f"PREDICTED: {best_category}")
    print()
    for category in FAILURE_CATEGORIES:
        if signals_fired[category]:
            print(f"  [{category}] " + "; ".join(signals_fired[category]))


def _parse_root_cause(text: str) -> tuple:
    cat_match = re.search(r"CATEGORY:\s*(\w+)", text)
    reason_match = re.search(r"REASONING:\s*(.+)", text, re.DOTALL)
    category = cat_match.group(1).strip() if cat_match else "capacity_creep"
    if category not in FAILURE_CATEGORIES:
        category = "capacity_creep"
    reasoning = reason_match.group(1).strip() if reason_match else "(no reasoning parsed)"
    return category, reasoning


# ---------------------------------------------------------------------------
# Knowledge / RAG Agent (parallel branch 1) -- upgraded to advanced_rag.py:
# hybrid BM25+TF-IDF retrieval, a contextual query built from live telemetry
# (not just the category name), parent-child chunking, and a reranking pass.
# ---------------------------------------------------------------------------
def knowledge_agent(state: AgentState) -> dict:
    query = arag.build_contextual_query(
        state["root_cause_category"], state["root_cause_reasoning"], row=state["row"],
    )
    results = arag.retrieve(query, top_k=2, failure_type=state["root_cause_category"])
    return {"knowledge_context": results}


# ---------------------------------------------------------------------------
# Historical Events Agent (parallel branch 2 -- this is what makes the RAG
# layer "agentic": two independent retrieval agents run off Root Cause in
# parallel, over two genuinely different sources, and both feed forward.
# Knowledge Agent retrieves generic SOP text; this agent retrieves THIS
# TOWER's own recorded incident history from failure_events.csv (real data,
# already produced by the simulator -- same file pick_demo_row() and
# list_events_for_tower() already read from) and summarizes whether the
# currently-predicted root cause category has precedent here, how often,
# and what the typical lead time / duration looked like historically.
# ---------------------------------------------------------------------------
_events_cache = None


def _load_events_df():
    global _events_cache
    if _events_cache is None:
        if os.path.exists(EVENTS_PATH):
            _events_cache = pd.read_csv(
                EVENTS_PATH, parse_dates=["ramp_start", "outage_start", "outage_end"])
        else:
            _events_cache = pd.DataFrame(
                columns=["tower_id", "failure_type", "ramp_start", "outage_start", "outage_end"])
    return _events_cache


def historical_events_agent(state: AgentState) -> dict:
    events = _load_events_df()
    tower_id = state["tower_id"]
    category = state["root_cause_category"]
    tower_events = events[events["tower_id"] == tower_id]

    if len(tower_events) == 0:
        return {"historical_context": {
            "tower_event_count": 0,
            "category_event_count": 0,
            "has_category_precedent": False,
            "summary": f"No recorded prior failure events for {tower_id} in failure_events.csv -- "
                       f"this would be the first recorded incident at this tower.",
        }}

    same_category = tower_events[tower_events["failure_type"] == category]
    lead_times = (tower_events["outage_start"] - tower_events["ramp_start"]).dt.total_seconds() / 3600
    durations = (tower_events["outage_end"] - tower_events["outage_start"]).dt.total_seconds() / 3600
    most_recent = tower_events.sort_values("outage_start").iloc[-1]

    if len(same_category):
        cat_lead_times = (same_category["outage_start"] - same_category["ramp_start"]).dt.total_seconds() / 3600
        precedent = (f"{len(same_category)} prior {category} event(s) recorded at {tower_id}, "
                     f"average lead time {cat_lead_times.mean():.1f}h before outage_start.")
    else:
        other_types = ", ".join(sorted(tower_events["failure_type"].unique()))
        precedent = (f"No prior {category} events recorded at {tower_id} specifically -- "
                     f"{len(tower_events)} prior event(s) of other type(s) exist ({other_types}).")

    summary = (
        f"{tower_id} has {len(tower_events)} recorded prior failure event(s) total. {precedent} "
        f"Most recent recorded event: {most_recent['failure_type']} "
        f"(outage_start={most_recent['outage_start']}). "
        f"Across all recorded events at this tower: average lead time {lead_times.mean():.1f}h, "
        f"average outage duration {durations.mean():.1f}h."
    )

    return {"historical_context": {
        "tower_event_count": int(len(tower_events)),
        "category_event_count": int(len(same_category)),
        "has_category_precedent": bool(len(same_category) > 0),
        "summary": summary,
    }}


def compute_priority(probability: float, connected_users: int) -> tuple:
    """
    FIX #3: priority now considers BOTH probability and customer impact,
    not probability alone -- a 90% probability outage on a tower with 20
    users is not the same operational priority as one with 2000 users.
    Returns (priority_label, impact_label).
    """
    if connected_users > 500:
        impact = "High"
    elif connected_users > 150:
        impact = "Medium"
    else:
        impact = "Low"

    if probability >= 0.7:
        prob_tier = "High"
    elif probability >= 0.4:
        prob_tier = "Medium"
    else:
        prob_tier = "Low"

    matrix = {
        ("High", "High"): "Critical",
        ("High", "Medium"): "High",
        ("High", "Low"): "High",
        ("Medium", "High"): "High",
        ("Medium", "Medium"): "Medium",
        ("Medium", "Low"): "Medium",
        ("Low", "High"): "Medium",
        ("Low", "Medium"): "Low",
        ("Low", "Low"): "Low",
    }
    priority = matrix[(prob_tier, impact)]
    return priority, impact


# ---------------------------------------------------------------------------
# Recommendation Agent -- LLM-prompted, not hardcoded if/else
# ---------------------------------------------------------------------------
def recommendation_agent(state: AgentState) -> dict:
    row = state["row"]
    knowledge_text = "\n".join(f"- [{d['id']}] {d['text']}" for d in (state["knowledge_context"] or []))
    historical_text = (state.get("historical_context") or {}).get(
        "summary", "No historical event data available.")

    context = f"""You are a telecom network operations engineer.

Context:
Battery: {row['battery_level_pct']:.0f}%
Voltage: {row['grid_voltage']:.0f}V
Packet loss: {row['packet_loss_pct']:.1f}%
Predicted outage probability ({state['primary_horizon']}): {state['primary_probability']*100:.1f}%
Model confidence: {state['primary_confidence']*100:.1f}%
Root cause: {state['root_cause_category']} -- {state['root_cause_reasoning']}

Relevant maintenance knowledge:
{knowledge_text}

Historical pattern at this tower:
{historical_text}

Generate specific, actionable recommendations for the operations team. Be concrete
about urgency and timing, and reference the knowledge context where relevant."""

    if state["provider_name"] == "rule_based":
        recommendation = _rule_based_recommendation(state)
    else:
        provider = get_provider(state["provider_name"])
        recommendation = provider.generate(context)

    return {"recommendation": recommendation}


def _rule_based_recommendation(state: AgentState) -> str:
    top_doc = state["knowledge_context"][0] if state["knowledge_context"] else None
    priority, impact = compute_priority(state["primary_probability"], state["row"].get("connected_users", 0))
    lines = [
        f"Priority: {priority} (probability tier + {impact} customer impact)",
        f"Root cause: {state['root_cause_category']} -- {state['root_cause_reasoning']}",
    ]
    if top_doc:
        lines.append(f"Per {top_doc['id']}: {top_doc['text']}")
    hist = state.get("historical_context") or {}
    if hist.get("summary"):
        lines.append(f"Historical pattern: {hist['summary']}")
    lines.append("(Rule-based summary -- for richer, context-synthesized recommendations, "
                  "run with --provider anthropic/openai/ollama.)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report Agent
# ---------------------------------------------------------------------------
def report_agent(state: AgentState) -> dict:
    row = state["row"]
    pred_lines = "\n".join(
        f"- {h}: {p['probability']*100:.1f}% probability, {p['confidence']*100:.1f}% confidence "
        f"[{p.get('calibration_method', 'raw')}]"
        + (f", meta-confidence {p['meta_confidence']*100:.1f}% (alert-specific)" if p.get("meta_confidence") is not None else "")
        + (f" (raised from {p['probability_before_monotonic_fix']*100:.1f}% to stay "
           f"consistent with a shorter horizon)" if p.get("monotonic_adjusted") else "")
        for h, p in state["predictions"].items()
    )
    knowledge_lines = "\n".join(f"- [{d['id']}] {d['text']}" for d in (state["knowledge_context"] or []))
    historical_text = (state.get("historical_context") or {}).get(
        "summary", "No historical event data available.")
    connected_users = row.get("connected_users", 0)
    priority, impact = compute_priority(state["primary_probability"], connected_users)
    calib_method = state.get("primary_calibration_method", "raw")
    calib_note = ("(Platt-calibrated)" if calib_method == "platt" else
                  "(Isotonic-calibrated)" if calib_method == "isotonic" else
                  "(uncalibrated -- run calibration_experiment.py to check if calibration helps here)")
    meta_conf = state.get("primary_meta_confidence")
    meta_conf_line = (f"\nAlert-specific meta-confidence: {meta_conf*100:.1f}% (validated on alerts at "
                       f"this horizon specifically -- see meta_confidence_diagnostics.py)."
                       if meta_conf is not None else "")

    report = f"""EXECUTIVE SUMMARY
Tower {state['tower_id']} shows a {state['primary_probability']*100:.1f}% probability of outage
within {state['primary_horizon']} {calib_note}, confidence {state['primary_confidence']*100:.1f}%.{meta_conf_line}
Root cause assessed as {state['root_cause_category']}.

ROOT CAUSE
{state['root_cause_reasoning']}

HISTORICAL PATTERN AT THIS TOWER
{historical_text}

TIMELINE
Snapshot analyzed at {row.get('timestamp')}. See failure_events.csv for exact historical windows.

EVIDENCE
{chr(10).join(f'- {k}: {v}' for k, v in state['evidence'].items())}

CONFIDENCE (all horizons) -- [calibration method applied to probability]
{pred_lines}

KNOWLEDGE BASE REFERENCES
{knowledge_lines or 'None retrieved.'}

RECOMMENDED ACTIONS
{state['recommendation']}

ESTIMATED CUSTOMER IMPACT
{impact} ({connected_users} connected users at time of snapshot).

SUGGESTED MAINTENANCE PRIORITY
{priority}"""

    return {"final_report": report}


def finish_agent(state: AgentState) -> dict:
    report = f"""EXECUTIVE SUMMARY
Tower {state['tower_id']} shows a {state['primary_probability']*100:.1f}% probability of outage
within {state['primary_horizon']} -- below the {state['threshold']*100:.1f}% action threshold.
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
    graph.add_node("historical_events", historical_events_agent)
    graph.add_node("recommendation", recommendation_agent)
    graph.add_node("report", report_agent)
    graph.add_node("finish", finish_agent)

    graph.set_entry_point("monitoring")
    graph.add_edge("monitoring", "decision")

    graph.add_conditional_edges(
        "decision", route_after_decision,
        {"root_cause": "root_cause", "finish": "finish"},
    )

    # Agentic RAG: Knowledge Agent (SOP corpus, advanced_rag.py) and
    # Historical Events Agent (this tower's own incident history,
    # failure_events.csv) both fan out from Root Cause and run in parallel,
    # then fan back in to Recommendation -- two independent retrieval
    # agents over two different sources, not one agent calling the other.
    graph.add_edge("root_cause", "knowledge")
    graph.add_edge("root_cause", "historical_events")
    graph.add_edge("knowledge", "recommendation")
    graph.add_edge("historical_events", "recommendation")
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
                         choices=["rule_based", "anthropic", "openai", "ollama", "phi4_local"])
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="Probability threshold at --primary-horizon that triggers full analysis.")
    parser.add_argument("--primary-horizon", default="3h",
                         choices=[label for label, _ in PREDICTION_HORIZONS],
                         help="Which horizon's probability drives the threshold branch (3h was the "
                              "sweet-spot horizon in earlier evaluation, hence the default).")
    parser.add_argument("--hours-before", type=float, default=3.0,
                         help="How many hours before the tower's actual recorded outage to snapshot "
                              "for this demo. Lower = closer to the failure (easier, higher confidence "
                              "expected). Higher = a genuine early-warning test (harder, more honest).")
    args = parser.parse_args()

    print(f"Loading {DATA_PATH} ...")
    df = pd.read_parquet(DATA_PATH)
    tower_id = args.tower or df["tower_id"].iloc[0]

    tdf = df[df["tower_id"] == tower_id].sort_values("timestamp")
    tdf = da.add_trend_features(tdf)
    row_series = pick_demo_row(tdf, tower_id, hours_before_outage=args.hours_before)
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

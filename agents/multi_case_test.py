"""
Priority 6: multi-case testing.

Runs the REAL orchestrator pipeline (orchestrator_v3.py or
orchestrator_llm.py -- your choice via --use-llm) end-to-end
(evidence -> root cause -> SOP -> recommendation -> probability) across:

  1. All 4 failure categories, using REAL historical events from
     failure_events.csv (not synthetic data) -- for each category, picks
     towers that actually had that failure type recorded, and checks
     whether the pipeline's predicted root cause matches the real
     historical label.
  2. A negative case -- a tower/snapshot chosen to be far from any
     recorded event window, checking the pipeline correctly stays quiet
     (no root cause triggered, no unnecessary diagnosis) rather than
     manufacturing a confident answer where there's nothing wrong.
  3. An ambiguous case -- a real snapshot with high utilization but normal
     packet_loss/latency (the conflicting-signal pattern named in the
     Priority 6 handoff notes). There's no "correct answer" key for this
     one by design -- ambiguity handling quality is a judgment call, not a
     pass/fail fact, so this case reports what happened for you to review
     rather than asserting it was handled "correctly."

Every genuine-failure and negative case also gets an automated consistency
check via eval_harness.py's RuleBasedJudge (the same free, keyword-based
checker that already knows the exact congestion/capacity-creep
contradiction pattern found and fixed earlier) -- this turns that bug fix
into a standing regression check, not just a one-off fix.

WHERE TO SAVE:
    Same folder as orchestrator_v3.py / orchestrator_llm.py / eval_harness.py
    (agents/).

RUN:
    python multi_case_test.py
    python multi_case_test.py --use-llm --provider ollama
    python multi_case_test.py --n-per-category 3 --threshold 0.5

WHAT COUNTS AS PASS/FAIL, STATED EXPLICITLY (nothing hidden in code):
    Genuine failure case:  PASS if predicted root cause category ==
                            recorded historical failure_type AND the
                            pipeline actually triggered (didn't route to
                            "finish" below threshold).
    Negative case:         PASS if the pipeline did NOT trigger a root
                            cause (stayed below threshold, or produced
                            no diagnosis) -- a false alarm is a FAIL.
    Ambiguous case:        No pass/fail. Reported for manual review.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decision_agent as da
import orchestrator_v3 as o3
from config import PREDICTION_HORIZONS
from eval_harness import RuleBasedJudge

# Reasonable default snapshot lead times per category -- these are
# ASSUMPTIONS about typical ramp windows, stated explicitly so they can be
# corrected. Override per-run with --hours-before if these don't match your
# simulator's actual ramp characteristics. capacity_creep is capped well
# under 24h deliberately -- that's the longest horizon the model actually
# predicts, so asking about anything further out is untestable by
# construction, not a pipeline gap.
DEFAULT_HOURS_BEFORE = {
    "congestion_collapse": 2.0,
    "hardware_power_failure": 4.0,
    "weather_induced_degradation": 3.0,
    "capacity_creep": 20.0,
}

EXPECTED_SOP_PREFIX = {
    "hardware_power_failure": "SOP-PWR",
    "congestion_collapse": "SOP-CONG",
    "weather_induced_degradation": "SOP-WTHR",
    "capacity_creep": "SOP-CAP",
}

# Ambiguous-case search thresholds -- high utilization, but NOT the
# degradation signals that would make it an unambiguous congestion case.
AMBIGUOUS_UTIL_MIN = 65.0
AMBIGUOUS_PACKET_LOSS_MAX = 3.0
AMBIGUOUS_LATENCY_MAX = 40.0

# How far (hours) a "normal" snapshot must be from any recorded event
# window at that tower, in either direction.
NORMAL_CASE_BUFFER_HOURS = 24.0


def _resolve_model_dir():
    candidates = [
        da.MODEL_DIR, "models", os.path.join("agents", "models"), os.path.join("..", "models"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return da.MODEL_DIR


def _pick_horizon_for(hours_before: float) -> str:
    """The smallest available prediction horizon that comfortably covers
    hours_before -- e.g. a snapshot taken 4h before outage needs at least
    the 6h horizon, not a hardcoded 3h that would ask 'will it happen in
    the next 3h?' about something that's actually 4h away. Falls back to
    the largest available horizon if hours_before exceeds all of them."""
    for label, hrs in PREDICTION_HORIZONS:
        if hrs >= hours_before:
            return label
    return PREDICTION_HORIZONS[-1][0]


def _get_actual_lead_time(events_df, tower_id, failure_type_filter):
    """Mirrors pick_demo_row()'s own event-selection logic exactly (same
    filtering, same iloc[0]) so this looks up lead time for the SAME event
    pick_demo_row will actually pick -- without duplicating or modifying
    pick_demo_row itself. Returns None if no matching event exists (caller
    then just proceeds and lets pick_demo_row's own fallback handle it)."""
    tower_events = events_df[events_df["tower_id"] == tower_id]
    if failure_type_filter:
        filtered = tower_events[tower_events["failure_type"] == failure_type_filter]
        if len(filtered):
            tower_events = filtered
    if len(tower_events) == 0:
        return None
    event = tower_events.iloc[0]
    return (event["outage_start"] - event["ramp_start"]).total_seconds() / 3600


def build_and_run(row_series, tower_id, backend, provider, threshold, use_llm, primary_horizon="3h"):
    row = row_series.to_dict()
    row["timestamp"] = str(row["timestamp"])
    initial_state = {
        "tower_id": tower_id, "row": row, "evidence": {}, "predictions": {},
        "primary_probability": 0.0, "primary_confidence": 0.0, "threshold": threshold,
        "root_cause_category": None, "root_cause_reasoning": None,
        "knowledge_context": None, "historical_context": None, "recommendation": None,
        "final_report": None, "provider_name": provider, "backend": backend,
        "primary_horizon": primary_horizon,
    }
    if use_llm:
        import orchestrator_llm as ollm
        app = ollm.build_graph()
    else:
        app = o3.build_graph()
    return app.invoke(initial_state, config={"recursion_limit": 50})


def run_failure_case(df, tower_id, expected_category, hours_before, backend, provider, threshold, use_llm, judge, events_df):
    tdf = df[df["tower_id"] == tower_id].sort_values("timestamp")
    tdf = da.add_trend_features(tdf)

    # Clamp hours_before to what this SPECIFIC event's actual lead time
    # supports -- requesting more than the event actually ramped for asks
    # for a snapshot before the ramp even started (genuinely no signal
    # available yet, confirmed by pick_demo_row's own built-in NOTE for
    # this exact situation). Also cap at the largest available prediction
    # horizon (24h) -- a snapshot 28h before outage is just as untestable
    # as one 36h before, since no horizon covers a 28h-away event either.
    # That's not a pipeline miss, so don't test it as one -- test the
    # largest lead time this event and this model's horizons can both
    # actually support.
    actual_lead_time = _get_actual_lead_time(events_df, tower_id, expected_category)
    max_horizon_hours = PREDICTION_HORIZONS[-1][1]
    effective_hours_before = min(hours_before, max_horizon_hours)
    clamp_note = ""
    if actual_lead_time is not None and effective_hours_before > actual_lead_time:
        effective_hours_before = max(actual_lead_time - 0.5, 0.25)
    if effective_hours_before != hours_before:
        clamp_note = (f"Requested hours_before={hours_before} was clamped to "
                       f"{effective_hours_before:.2f}h (event's actual lead time: "
                       f"{actual_lead_time if actual_lead_time else 'unknown'}, "
                       f"max available horizon: {max_horizon_hours}h) so the snapshot falls inside "
                       f"both the actual ramp window and what the model can be asked about.")

    primary_horizon = _pick_horizon_for(effective_hours_before)
    row_series = o3.pick_demo_row(tdf, tower_id, hours_before_outage=effective_hours_before,
                                   failure_type_filter=expected_category)
    if row_series is None:
        return {"case_type": "failure", "expected_category": expected_category, "tower_id": tower_id,
                "result": "SKIPPED", "notes": "pick_demo_row found no matching event."}

    final_state = build_and_run(row_series, tower_id, backend, provider, threshold, use_llm, primary_horizon)

    triggered = final_state.get("root_cause_category") is not None
    predicted = final_state.get("root_cause_category")
    correct = triggered and predicted == expected_category

    sop_ok = None
    if triggered and final_state.get("knowledge_context"):
        expected_prefix = EXPECTED_SOP_PREFIX.get(expected_category)
        sop_ids = [d["id"] for d in final_state["knowledge_context"]]
        sop_ok = any(sid.startswith(expected_prefix) for sid in sop_ids) if expected_prefix else None

    consistency_score, consistency_note = None, ""
    if triggered and final_state.get("final_report"):
        judgment = judge.judge(final_state["final_report"])
        consistency_score = judgment["consistency"]["score"]
        consistency_note = judgment["consistency"]["justification"]

    return {
        "case_type": "failure", "expected_category": expected_category, "tower_id": tower_id,
        "result": "PASS" if correct else ("FAIL - not triggered" if not triggered else "FAIL - wrong category"),
        "predicted_category": predicted,
        "probability": final_state.get("primary_probability"),
        "confidence": final_state.get("primary_confidence"),
        "primary_horizon_used": primary_horizon,
        "effective_hours_before": effective_hours_before,
        "sop_prefix_match": sop_ok,
        "consistency_score": consistency_score,
        "consistency_note": consistency_note,
        "notes": clamp_note,
    }


def find_normal_snapshot(df, events_df, tower_id, buffer_hours=NORMAL_CASE_BUFFER_HOURS):
    tdf = df[df["tower_id"] == tower_id].sort_values("timestamp")
    if len(tdf) == 0:
        return None
    tower_events = events_df[events_df["tower_id"] == tower_id]
    exclusions = []
    for _, ev in tower_events.iterrows():
        start = ev["ramp_start"] - pd.Timedelta(hours=buffer_hours)
        end = ev["outage_end"] + pd.Timedelta(hours=buffer_hours)
        exclusions.append((start, end))

    for _, row in tdf.iterrows():
        ts = row["timestamp"]
        if not any(start <= ts <= end for start, end in exclusions):
            return row
    return None


def run_normal_case(df, events_df, tower_id, backend, provider, threshold, use_llm, judge):
    tdf = df[df["tower_id"] == tower_id].sort_values("timestamp")
    tdf = da.add_trend_features(tdf)
    normal_row = find_normal_snapshot(tdf, events_df, tower_id)
    if normal_row is None:
        return {"case_type": "negative", "tower_id": tower_id, "result": "SKIPPED",
                "notes": f"Could not find a snapshot at {tower_id} outside all event windows "
                         f"(+/- {NORMAL_CASE_BUFFER_HOURS}h buffer)."}

    final_state = build_and_run(normal_row, tower_id, backend, provider, threshold, use_llm)
    triggered = final_state.get("root_cause_category") is not None

    consistency_score, consistency_note = None, ""
    if final_state.get("final_report"):
        judgment = judge.judge(final_state["final_report"])
        consistency_score = judgment["consistency"]["score"]
        consistency_note = judgment["consistency"]["justification"]

    return {
        "case_type": "negative", "tower_id": tower_id,
        "result": "FAIL - false alarm" if triggered else "PASS",
        "predicted_category": final_state.get("root_cause_category"),
        "probability": final_state.get("primary_probability"),
        "confidence": final_state.get("primary_confidence"),
        "consistency_score": consistency_score, "consistency_note": consistency_note,
        "notes": f"snapshot at {normal_row['timestamp']}",
    }


def find_ambiguous_snapshot(df):
    """Real snapshot with high utilization but normal packet_loss/latency
    -- the specific conflicting-signal pattern named in the handoff notes.
    Searches raw columns only (no trend features needed to locate a
    candidate), then hands back the (tower_id, timestamp) for the caller to
    build full features on just that tower."""
    candidates = df[
        (df["utilization_pct"] > AMBIGUOUS_UTIL_MIN) &
        (df["packet_loss_pct"] < AMBIGUOUS_PACKET_LOSS_MAX) &
        (df["latency_ms"] < AMBIGUOUS_LATENCY_MAX)
    ]
    if len(candidates) == 0:
        return None, None
    picked = candidates.iloc[len(candidates) // 2]  # a stable, reproducible pick
    return picked["tower_id"], picked["timestamp"]


def run_ambiguous_case(df, backend, provider, threshold, use_llm, judge):
    tower_id, ts = find_ambiguous_snapshot(df)
    if tower_id is None:
        return {"case_type": "ambiguous", "tower_id": None, "result": "SKIPPED",
                "notes": f"No snapshot found with utilization>{AMBIGUOUS_UTIL_MIN}%, "
                         f"packet_loss<{AMBIGUOUS_PACKET_LOSS_MAX}%, latency<{AMBIGUOUS_LATENCY_MAX}ms. "
                         f"Adjust thresholds if your dataset's ranges differ."}

    tdf = df[df["tower_id"] == tower_id].sort_values("timestamp")
    tdf = da.add_trend_features(tdf)
    row = tdf[tdf["timestamp"] == ts].iloc[0]

    final_state = build_and_run(row, tower_id, backend, provider, threshold, use_llm)
    triggered = final_state.get("root_cause_category") is not None

    consistency_score, consistency_note = None, ""
    if final_state.get("final_report"):
        judgment = judge.judge(final_state["final_report"])
        consistency_score = judgment["consistency"]["score"]
        consistency_note = judgment["consistency"]["justification"]

    return {
        "case_type": "ambiguous", "tower_id": tower_id,
        "result": "REPORTED - no pass/fail (see notes)",
        "predicted_category": final_state.get("root_cause_category"),
        "probability": final_state.get("primary_probability"),
        "confidence": final_state.get("primary_confidence"),
        "consistency_score": consistency_score, "consistency_note": consistency_note,
        "notes": (f"snapshot at {ts}: high utilization + normal packet_loss/latency. "
                  f"{'Triggered category ' + str(final_state.get('root_cause_category')) if triggered else 'Correctly did not force a diagnosis.'} "
                  f"Review manually -- ambiguity handling isn't a pass/fail fact."),
    }


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-category", type=int, default=2)
    parser.add_argument("--backend", default="lightgbm", choices=["lightgbm", "xgboost", "random_forest"])
    parser.add_argument("--provider", default="rule_based",
                         choices=["rule_based", "anthropic", "openai", "ollama", "phi4_local"])
    parser.add_argument("--use-llm", action="store_true",
                         help="Run orchestrator_llm.py instead of orchestrator_v3.py")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--hours-before", type=float, default=None,
                         help="Override DEFAULT_HOURS_BEFORE for ALL categories.")
    args = parser.parse_args()

    print(f"Loading {o3.DATA_PATH} ...")
    df = pd.read_parquet(o3.DATA_PATH)
    events_df = pd.read_csv(o3.EVENTS_PATH, parse_dates=["ramp_start", "outage_start", "outage_end"]) \
        if os.path.exists(o3.EVENTS_PATH) else pd.DataFrame(columns=["tower_id", "failure_type"])

    judge = RuleBasedJudge()
    results = []

    print(f"\n{'='*70}\n1. GENUINE FAILURE CASES (all {len(o3.FAILURE_CATEGORIES)} categories)\n{'='*70}")
    for category in o3.FAILURE_CATEGORIES:
        cat_events = events_df[events_df["failure_type"] == category]
        towers = list(dict.fromkeys(cat_events.sort_values("outage_start", ascending=False)["tower_id"]))
        towers = towers[:args.n_per_category]
        if not towers:
            print(f"  {category}: no recorded events found in failure_events.csv -- skipping.")
            continue
        hours_before = args.hours_before or DEFAULT_HOURS_BEFORE[category]
        for tower_id in towers:
            print(f"  Testing {category} @ {tower_id} (hours_before={hours_before})...", end=" ", flush=True)
            r = run_failure_case(df, tower_id, category, hours_before, args.backend, args.provider,
                                  args.threshold, args.use_llm, judge, events_df)
            results.append(r)
            print(r["result"])

    print(f"\n{'='*70}\n2. NEGATIVE CASE (should NOT trigger a diagnosis)\n{'='*70}")
    if len(events_df):
        # Pick a tower that has recorded events (so we know its normal
        # baseline is meaningfully different from its failure windows),
        # then find a snapshot far from all of them.
        candidate_tower = events_df["tower_id"].iloc[0]
        print(f"  Testing negative case @ {candidate_tower}...", end=" ", flush=True)
        r = run_normal_case(df, events_df, candidate_tower, args.backend, args.provider,
                             args.threshold, args.use_llm, judge)
        results.append(r)
        print(r["result"])
    else:
        print("  No events file available to pick a reference tower -- skipping.")

    print(f"\n{'='*70}\n3. AMBIGUOUS CASE (conflicting signals -- reported, not graded)\n{'='*70}")
    r = run_ambiguous_case(df, args.backend, args.provider, args.threshold, args.use_llm, judge)
    results.append(r)
    print(f"  {r['result']}: {r.get('notes', '')}")

    results_df = pd.DataFrame(results)
    results_df.to_csv("multi_case_test_results.csv", index=False)
    print(f"\n\n{'='*70}\nSUMMARY\n{'='*70}")
    print(results_df[["case_type", "tower_id", "result"]].to_string(index=False))
    print("\nSaved multi_case_test_results.csv (full detail, including consistency notes)")

    n_pass = (results_df["result"] == "PASS").sum()
    n_fail = results_df["result"].astype(str).str.startswith("FAIL").sum()
    n_skip = (results_df["result"] == "SKIPPED").sum()
    print(f"\nPASS: {n_pass}  FAIL: {n_fail}  SKIPPED: {n_skip}  "
          f"(ambiguous case is reported separately, not counted here)")


if __name__ == "__main__":
    run()

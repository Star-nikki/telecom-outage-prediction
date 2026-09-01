"""
Generates the training dataset for fine-tuning Phi-4 Mini as the LLM-based
Root Cause Agent. Runs entirely locally, CPU-only, no GPU needed.

CROSS-CATEGORY AWARE: every row is scored against ALL FOUR categories'
signal checklists, not just the true label's. Confidence is derived from
BOTH how strongly the true category's own signals fire AND how large the
margin is over the best-scoring competing category. If a competing
category is nearly as strong (e.g. severe packet loss AND a strong 24h
utilization trend both present), confidence is lowered and the reasoning
explicitly names the competing category and why it was still not chosen --
mirroring exactly what the rule-based orchestrator_v3 Root Cause Agent does
when evidence is genuinely ambiguous.

WHERE TO SAVE:
    telecom_sim_v2/finetune/generate_training_data.py

USAGE:
    cd finetune
    python generate_training_data.py
"""

import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agents"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from decision_agent import add_trend_features  # noqa: E402
from config import RNG  # noqa: E402

DATA_PATH = "../data/telecom_timeseries.parquet"
EVENTS_PATH = "../data/failure_events.csv"
OUTPUT_PATH = "training_data.jsonl"

LEAD_TIME_SAMPLES_HOURS = [0.5, 1, 2, 3, 6, 12, 24, 48]

SYSTEM_PROMPT = (
    "You are a telecom network root cause diagnosis assistant. Given telemetry "
    "evidence from monitoring agents, classify the most likely failure category "
    "and explain your reasoning using the specific signals present."
)

FAILURE_CATEGORIES = ["hardware_power_failure", "congestion_collapse",
                       "weather_induced_degradation", "capacity_creep"]

OPENERS = {
    "hardware_power_failure": [
        "Power supply readings point to hardware degradation rather than a traffic or weather cause.",
        "Battery and voltage trends indicate the backup power system is failing.",
        "The declining power supply signature is consistent with an imminent hardware/power failure.",
        "Electrical readings show a clear power-side fault developing.",
    ],
    "congestion_collapse": [
        "High utilization combined with packet loss indicates traffic saturation.",
        "The rising utilization trend alongside worsening latency suggests the site is approaching capacity exhaustion.",
        "Network performance is deteriorating while demand continues to rise, matching prior congestion collapse patterns.",
        "Acute traffic overload signatures are visible across multiple metrics simultaneously.",
    ],
    "capacity_creep": [
        "Utilization has been climbing steadily without a single acute trigger, consistent with organic demand growth.",
        "The sustained multi-day utilization increase points to capacity creep rather than a sudden event.",
        "Demand has been building gradually over an extended window, exceeding site capacity incrementally.",
        "This is a slow structural trend rather than an acute incident.",
    ],
}

# Weather openers are split into THREE strictly separate pools based on the
# row's ACTUAL storm state -- never mixed, never chosen independent of the
# data. This is what a single two-way (active/post-storm) split still
# missed: rows where storm_flag=0 AND signal isn't even degraded (RSRP not
# below -85) have NO real weather evidence at all, so they must never draw
# from a pool that implies storm/atmospheric causation.
ACTIVE_STORM_OPENERS = [
    "The timing of signal loss aligns with active storm conditions in this region.",
    "An active storm is directly correlated with the observed signal degradation.",
    "Current storm activity is the clear driver of this signal loss.",
]

POST_STORM_OPENERS = [
    "Although the storm has already passed, signal degradation persists, consistent with lingering weather-related impact.",
    "Signal quality remains degraded even after the storm cleared, suggesting lasting weather-related damage.",
    "The storm itself has ended, but its effects on signal strength continue to be visible.",
]

# Used ONLY when storm_flag=0 AND signal is not meaningfully degraded either
# -- i.e. genuinely weak/no direct evidence. Never mentions storms, weather,
# or atmosphere as an active/confirmed cause, since none of that is
# actually true in the data at this exact snapshot.
WEAK_SIGNAL_WEATHER_OPENERS = [
    "This event is attributed to weather-driven degradation, though current readings at this exact snapshot show only limited direct evidence -- likely an early or transitional stage before the impact becomes clearly visible.",
    "The assigned cause is weather-related, but the telemetry at this specific moment does not yet show strong corroborating signals.",
    "While the underlying event is weather-driven, this snapshot alone shows weak supporting evidence -- confidence should be treated accordingly.",
]

CATEGORY_LABELS = {
    "hardware_power_failure": "a hardware/power fault",
    "congestion_collapse": "acute traffic congestion",
    "weather_induced_degradation": "weather-driven degradation",
    "capacity_creep": "slow capacity creep",
}


# ---------------------------------------------------------------------------
# Weighted signal checklists, one per category. Each entry:
# (is_true: bool, description: str, weight: float)
# Weight reflects how strongly that single signal implies THIS category.
# ---------------------------------------------------------------------------
def _checks_hardware(row):
    battery, voltage = row["battery_level_pct"], row["grid_voltage"]
    battery_trend_1h = row.get("battery_level_pct_delta_1h")
    checks = [
        (battery < 70, f"battery has dropped to {battery:.0f}%", 3),
        (battery < 40, f"battery is critically low at {battery:.0f}%", 2),
        (voltage < 215, f"grid voltage has sagged to {voltage:.0f}V", 3),
        (voltage < 190, f"grid voltage is critically low at {voltage:.0f}V", 2),
    ]
    if battery_trend_1h is not None:
        checks.append((battery_trend_1h < -2, f"battery is declining at {battery_trend_1h:.1f} pts/hour", 2))
        checks.append((battery_trend_1h < -8, "battery is declining rapidly", 2))
    return checks


def _checks_congestion(row):
    util, packet_loss, latency = row["utilization_pct"], row["packet_loss_pct"], row["latency_ms"]
    util_trend_1h = row.get("utilization_pct_delta_1h") or 0
    return [
        (packet_loss > 10, f"packet loss has reached {packet_loss:.1f}%", 2),
        (packet_loss > 30, "packet loss is severe", 2),
        (latency > 100, f"latency has climbed to {latency:.0f}ms", 2),
        (latency > 200, "latency is severely elevated", 1),
        (util > 60, f"utilization is at {util:.0f}%", 2),
        (util > 85, "utilization is near saturation", 1),
        (util_trend_1h > 1, f"utilization rose {util_trend_1h:+.1f} points in the last hour", 1),
        (util_trend_1h > 10, "utilization spiked sharply within the last hour, not a slow build-up", 1),
    ]


def _checks_weather(row):
    rsrp, storm, rainfall, wind = row["rsrp_dbm"], row["storm_flag"], row["rainfall_mm"], row["wind_kmh"]
    return [
        (storm == 1, "an active storm is present at this exact moment", 3),
        (rainfall > 2, f"rainfall is measured at {rainfall:.1f}mm", 1),
        (rainfall > 10, "rainfall is heavy", 1),
        (rsrp < -85, f"signal strength has degraded to {rsrp:.0f} dBm", 2),
        (rsrp < -100, "signal strength is severely degraded", 2),
        (wind > 20, f"wind speed is elevated at {wind:.0f}km/h", 1),
        (wind > 40, "wind speed is severe", 1),
    ]


def _checks_capacity_creep(row):
    util = row["utilization_pct"]
    util_trend_24h = row.get("utilization_pct_delta_24h") or 0
    util_trend_1h = row.get("utilization_pct_delta_1h") or 0
    return [
        (util_trend_24h > 5, f"utilization has climbed {util_trend_24h:+.1f} points over the last 24 hours", 2),
        (util_trend_24h > 15, "this is a substantial sustained increase, not a brief spike", 2),
        (util_trend_24h > 30, "the sustained increase is extreme", 1),
        (util > 40, f"current utilization stands at {util:.0f}%", 1),
        (util_trend_1h <= 1, "the increase is gradual rather than sudden", 1),
    ]


CHECK_FUNCTIONS = {
    "hardware_power_failure": _checks_hardware,
    "congestion_collapse": _checks_congestion,
    "weather_induced_degradation": _checks_weather,
    "capacity_creep": _checks_capacity_creep,
}


def _score_category(category, row):
    """Returns (normalized_score_0_100, matched_descriptions)."""
    checks = CHECK_FUNCTIONS[category](row)
    max_possible = sum(weight for _, _, weight in checks) or 1
    matched = [(desc, weight) for is_true, desc, weight in checks if is_true]
    earned = sum(weight for _, weight in matched)
    score = (earned / max_possible) * 100
    return score, [desc for desc, _ in matched]


def build_evidence_text(row: pd.Series) -> str:
    lines = [
        f"Battery {row['battery_level_pct']:.0f}%, grid voltage {row['grid_voltage']:.0f}V, "
        f"RSRP {row['rsrp_dbm']:.0f} dBm.",
        f"Utilization {row['utilization_pct']:.0f}%, packet loss {row['packet_loss_pct']:.1f}%, "
        f"throughput {row['throughput_mbps']:.0f} Mbps, latency {row['latency_ms']:.0f} ms.",
        f"Storm={'ACTIVE' if row['storm_flag'] else 'clear'}, rainfall {row['rainfall_mm']:.1f}mm, "
        f"wind {row['wind_kmh']:.0f}km/h.",
        f"Load multiplier {row['load_multiplier']:.2f}.",
    ]
    d24 = row.get("utilization_pct_delta_24h")
    if pd.notna(d24):
        lines.append(f"24h utilization trend: {d24:+.1f} pts.")
    d1 = row.get("battery_level_pct_delta_1h")
    if pd.notna(d1) and abs(d1) > 0.5:
        lines.append(f"Battery changed {d1:+.1f} pts over the last hour.")
    return " ".join(lines)


def build_target_output(row: pd.Series, true_category: str, lead_time_h: float, actual_lead_time: float) -> str:
    all_scores = {cat: _score_category(cat, row) for cat in FAILURE_CATEGORIES}
    true_score, true_matched = all_scores[true_category]

    competitors = {cat: s for cat, s in all_scores.items() if cat != true_category}
    runner_up_cat = max(competitors, key=lambda c: competitors[c][0])
    runner_up_score, runner_up_matched = competitors[runner_up_cat]

    margin = true_score - runner_up_score
    # Confidence blends absolute evidence strength for the true category
    # with how clearly it beats the best competing explanation. A high
    # true_score with a small margin (a real competitor is nearly as
    # strong) still yields only moderate confidence.
    confidence_pct = int(np.clip(0.6 * true_score + 0.4 * margin, 10, 97))

    if true_category == "weather_induced_degradation":
        if row["storm_flag"] == 1:
            opener_pool = ACTIVE_STORM_OPENERS
        elif row["rsrp_dbm"] < -85:
            opener_pool = POST_STORM_OPENERS
        else:
            opener_pool = WEAK_SIGNAL_WEATHER_OPENERS
    else:
        opener_pool = OPENERS[true_category]
    opener = opener_pool[RNG.integers(0, len(opener_pool))]

    if true_matched:
        evidence_sentence = "Specifically, " + ", ".join(true_matched) + "."
    else:
        evidence_sentence = ("However, the supporting signals are still weak at this snapshot -- "
                              "this is an early-stage indicator rather than a clear-cut case.")

    ambiguity_note = ""
    if margin < 15 and runner_up_score > 25:
        competing_desc = ", ".join(runner_up_matched) if runner_up_matched else "some overlapping signals"
        ambiguity_note = (f" Note: {CATEGORY_LABELS[runner_up_cat]} is also plausible here ({competing_desc}), "
                           f"so this classification carries some uncertainty rather than being clear-cut.")

    reasoning = f"{opener} {evidence_sentence}{ambiguity_note}"

    return (f"CATEGORY: {true_category}\n"
            f"CONFIDENCE: {confidence_pct}%\n"
            f"REASONING: {reasoning} (snapshot taken approximately {lead_time_h:.1f}h before the actual outage.)")


def main():
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_parquet(DATA_PATH)
    events = pd.read_csv(EVENTS_PATH, parse_dates=["ramp_start", "outage_start", "outage_end"])
    print(f"{len(events)} events found across {events['tower_id'].nunique()} towers.")

    examples = []
    tower_trend_cache = {}

    for _, event in events.iterrows():
        tower_id, category = event["tower_id"], event["failure_type"]
        if category not in FAILURE_CATEGORIES:
            continue

        actual_lead_time = (event["outage_start"] - event["ramp_start"]).total_seconds() / 3600

        if tower_id not in tower_trend_cache:
            tdf = df[df["tower_id"] == tower_id].sort_values("timestamp")
            tower_trend_cache[tower_id] = add_trend_features(tdf)
        tdf = tower_trend_cache[tower_id]

        for lead_h in LEAD_TIME_SAMPLES_HOURS:
            if lead_h > actual_lead_time:
                continue
            target_time = event["outage_start"] - pd.Timedelta(hours=lead_h)
            tdf["_diff"] = (tdf["timestamp"] - target_time).abs()
            row = tdf.loc[tdf["_diff"].idxmin()]

            evidence_text = build_evidence_text(row)
            target_output = build_target_output(row, category, lead_h, actual_lead_time)

            examples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Evidence from telecom monitoring agents for tower {tower_id}:\n"
                                                  f"{evidence_text}\n\nWhich failure category best explains this, "
                                                  f"and why? Choose exactly one: hardware_power_failure, "
                                                  f"congestion_collapse, weather_induced_degradation, capacity_creep"},
                    {"role": "assistant", "content": target_output},
                ]
            })

    print(f"Generated {len(examples)} training examples.")
    categories_seen = [ex["messages"][2]["content"].split("\n")[0].replace("CATEGORY: ", "") for ex in examples]
    print("Category distribution:")
    print(pd.Series(categories_seen).value_counts())

    with open(OUTPUT_PATH, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nWrote {OUTPUT_PATH}.")


if __name__ == "__main__":
    main()

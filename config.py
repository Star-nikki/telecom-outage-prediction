"""
Central configuration for the synthetic 5G telecom network simulator.
Tune these to change scale (more towers, longer duration) or failure rates.
"""

import numpy as np

# ---------- Scale ----------
N_TOWERS = 60                 # number of towers in the fleet
N_REGIONS = 6                 # towers are grouped into regions (share weather)
START_DATE = "2024-01-01"     # simulation start (inclusive)
SIM_DAYS = 1096                # 2024 (366, leap year) + 2025 (365) + 2026 (365) = 1096
INTERVAL_MINUTES = 15         # sampling interval
RANDOM_SEED = 42

# ---------- Tower types ----------
TOWER_TYPES = {
    "macro":      {"capacity_mbps": 1000, "weight": 0.5, "battery_hours": 8},
    "micro":      {"capacity_mbps": 400,  "weight": 0.3, "battery_hours": 4},
    "small_cell": {"capacity_mbps": 150,  "weight": 0.2, "battery_hours": 2},
}

# ---------- Failure injection ----------
# Expected number of failure events per tower PER YEAR (Poisson rate).
# Scales naturally with SIM_DAYS instead of being a fixed count -- a 1-year
# sim and a 3-year sim will show proportionally the same failure density,
# which matters for realistic long-horizon evaluation (e.g. seasonal
# clustering of weather failures across years).
FAILURES_PER_TOWER_PER_YEAR = 8

FAILURE_TYPES = {
    "hardware_power_failure": {
        "lead_time_hours": (6, 12),    # degradation ramp starts this many hours before
        "outage_duration_hours": (1, 4),
        "weight": 0.30,
    },
    "congestion_collapse": {
        "lead_time_hours": (4, 8),
        "outage_duration_hours": (0.5, 2),
        "weight": 0.30,
    },
    "weather_induced_degradation": {
        "lead_time_hours": (6, 12),
        "outage_duration_hours": (1, 6),
        "weight": 0.20,
    },
    "capacity_creep": {
        "lead_time_hours": (24, 72),   # slow, multi-day build-up
        "outage_duration_hours": (2, 8),
        "weight": 0.20,
    },
}

# ---------- Mobility events (IMPROVEMENT #3) ----------
# Named, probabilistic demand events instead of anonymous random spikes.
# `rate_per_year` is a Poisson rate per region. `magnitude` is a multiplier
# range applied on top of the normal load curve. `duration_hours` is how
# long the event lasts. `regions_affected` controls whether an event hits
# one random region ("single") or several at once ("multi", e.g. a national
# holiday), which matters for realistic correlated-failure scenarios.
MOBILITY_EVENT_TYPES = {
    "festival": {
        "rate_per_year": 2.0,
        "magnitude": (1.8, 2.5),
        "duration_hours": (6, 18),
        "regions_affected": "multi",
    },
    "cricket_match": {
        "rate_per_year": 6.0,
        "magnitude": (2.0, 3.2),
        "duration_hours": (3, 5),
        "regions_affected": "single",
    },
    "concert": {
        "rate_per_year": 4.0,
        "magnitude": (1.6, 2.2),
        "duration_hours": (2, 4),
        "regions_affected": "single",
    },
    "emergency": {
        "rate_per_year": 0.5,
        "magnitude": (2.5, 4.0),
        "duration_hours": (1, 6),
        "regions_affected": "single",
    },
}

# ---------- Prediction target ----------
# IMPROVEMENT #6: multiple prediction horizons instead of one.
# Each tuple is (label_suffix, hours). Agents/models will be trained and
# evaluated separately per horizon, and you compare performance across them:
# short horizons (15 min) should be easy (near-detection), long horizons
# (24 hr) should be genuinely hard (true forecasting). That comparison curve
# is itself a real evaluation result worth reporting.
PREDICTION_HORIZONS = [
    ("15m", 0.25),
    ("30m", 0.5),
    ("1h", 1),
    ("3h", 3),
    ("6h", 6),
    ("12h", 12),
    ("24h", 24),
]

# ---------- Weather source ----------
# IMPROVEMENT #2: prefer real historical weather if present.
# Run scripts/fetch_historical_weather.py first to populate this file; if it
# doesn't exist, simulate.py silently falls back to the synthetic generator.
REAL_WEATHER_PATH = "data/real_weather.csv"

# ---------- Streaming ----------
# IMPROVEMENT #1: how fast the streaming simulator "plays back" ticks.
# 1.0 = one tick every real 15 minutes (realistic pacing).
# Set much higher (e.g. 3600) to fast-forward through history for demos.
STREAMING_SPEED_MULTIPLIER = 3600


RNG = np.random.default_rng(RANDOM_SEED)

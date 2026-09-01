"""
IMPROVEMENT #3: models expected traffic load using NAMED, probabilistic
mobility events (festival, cricket_match, concert, emergency) instead of
anonymous random spikes. Each event type has its own rate, magnitude, and
duration profile (see config.MOBILITY_EVENT_TYPES), and events are logged
to a ground-truth file (mobility_events.csv) -- useful later for explaining
*why* the Traffic Agent saw a spike, not just that it happened.
"""

import numpy as np
import pandas as pd
from config import N_REGIONS, MOBILITY_EVENT_TYPES, RNG


def generate_behavior(timestamps: pd.DatetimeIndex):
    """
    Returns (behavior_df, events_df):
      behavior_df: region, timestamp, load_multiplier
      events_df:   event_type, region, start, end, magnitude  (ground truth log)
    """
    n = len(timestamps)
    hours = timestamps.hour.values + timestamps.minute.values / 60
    is_weekend = (timestamps.dayofweek >= 5).astype(int)
    duration_years = (timestamps[-1] - timestamps[0]).total_seconds() / (365 * 24 * 3600)

    # ---- base diurnal/weekly pattern ----
    morning_peak = 0.5 * np.exp(-((hours - 9) ** 2) / (2 * 2.0 ** 2))
    evening_peak = 1.0 * np.exp(-((hours - 20) ** 2) / (2 * 2.5 ** 2))
    base_load = 0.4 + morning_peak + evening_peak
    base_load *= np.where(is_weekend, 0.85, 1.0)

    multiplier_by_region = {}
    for r in range(N_REGIONS):
        m = base_load + RNG.normal(0, 0.05, n)
        multiplier_by_region[r] = np.clip(m, 0.15, None)

    # ---- named probabilistic events ----
    event_log = []
    for event_type, spec in MOBILITY_EVENT_TYPES.items():
        expected_count = spec["rate_per_year"] * duration_years
        n_events = RNG.poisson(expected_count)

        for _ in range(n_events):
            duration_steps = int(RNG.uniform(*spec["duration_hours"]) * 4)  # 15-min steps
            if duration_steps >= n - 1:
                continue
            start_idx = RNG.integers(0, n - duration_steps - 1)
            magnitude = RNG.uniform(*spec["magnitude"])

            if spec["regions_affected"] == "multi":
                affected_regions = list(range(N_REGIONS))
            else:
                affected_regions = [int(RNG.integers(0, N_REGIONS))]

            for r in affected_regions:
                multiplier_by_region[r][start_idx:start_idx + duration_steps] *= magnitude

            event_log.append({
                "event_type": event_type,
                "region": "ALL" if spec["regions_affected"] == "multi" else f"R{affected_regions[0]}",
                "start": timestamps[start_idx],
                "end": timestamps[min(start_idx + duration_steps, n - 1)],
                "magnitude": round(magnitude, 2),
            })

    frames = []
    for r in range(N_REGIONS):
        frames.append(pd.DataFrame({
            "region": f"R{r}",
            "timestamp": timestamps,
            "load_multiplier": np.round(multiplier_by_region[r], 3),
        }))

    behavior_df = pd.concat(frames, ignore_index=True)
    events_df = pd.DataFrame(event_log).sort_values("start").reset_index(drop=True) if event_log else pd.DataFrame(
        columns=["event_type", "region", "start", "end", "magnitude"]
    )
    return behavior_df, events_df


if __name__ == "__main__":
    ts = pd.date_range("2024-01-01", periods=96 * 365, freq="15min")
    bdf, edf = generate_behavior(ts)
    print(bdf.head())
    print(f"\n{len(edf)} mobility events generated:")
    print(edf["event_type"].value_counts())
    print(f"\nMax multiplier seen: {bdf['load_multiplier'].max():.2f}")

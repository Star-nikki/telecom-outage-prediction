"""
Decides WHICH towers fail, WHEN, and WHAT TYPE -- producing an event schedule
that simulate.py uses to shape the actual metric curves. Keeping this
separate from metric generation means the ground-truth root cause is known
by construction, which is what lets you score prediction + RCA quality later.

FIX (found via a fine-tuning data review): weather_induced_degradation events
used to be scheduled at a completely random time, independent of whether a
storm was actually happening in that tower's region. That meant the label
"weather_induced_degradation" could be attached to a snapshot where
storm_flag=0 -- a genuine contradiction between the ground-truth label and
the underlying weather data, which would have taught a fine-tuned model to
hallucinate storms. Now, weather_induced_degradation events are anchored to
an ACTUAL storm window in that tower's region whenever one is available,
falling back to random placement only if no usable storm window exists
(tracked and reported so you know how often that happens).
"""

import pandas as pd
import numpy as np
from config import FAILURES_PER_TOWER_PER_YEAR, FAILURE_TYPES, RNG


def _find_storm_windows(storm_flags: np.ndarray, min_length: int):
    """Returns list of (start_idx, end_idx) inclusive index ranges where
    storm_flag==1 continuously for at least min_length steps."""
    windows = []
    in_storm = False
    start = None
    n = len(storm_flags)
    for i in range(n):
        flag = storm_flags[i]
        if flag == 1 and not in_storm:
            in_storm = True
            start = i
        elif flag == 0 and in_storm:
            in_storm = False
            if i - start >= min_length:
                windows.append((start, i - 1))
    if in_storm and n - start >= min_length:
        windows.append((start, n - 1))
    return windows


def _try_place_random(outage_lo, outage_hi, lead_steps, dur_steps, n, occupied, max_attempts):
    """Original random-placement logic, used for all non-weather failure
    types and as a fallback for weather events with no usable storm window."""
    for _ in range(max_attempts):
        outage_start_idx = RNG.integers(outage_lo, outage_hi)
        ramp_start_idx = outage_start_idx - lead_steps
        outage_end_idx = min(outage_start_idx + dur_steps, n - 1)

        overlap = any(
            ramp_start_idx <= occ_end and occ_start <= outage_end_idx
            for occ_start, occ_end in occupied
        )
        if not overlap:
            return ramp_start_idx, outage_start_idx, outage_end_idx
    return None


def _try_place_in_storm(storm_windows, lead_steps, dur_steps, n, occupied, max_attempts):
    """Anchors ramp_start to the START of a real storm window, so the
    degradation ramp begins genuinely inside a storm. The outage itself
    follows lead_steps later -- for short lead times this keeps the whole
    ramp within the storm; for longer lead times (e.g. 6-12h) the outage
    may land after the storm has passed, which is fine (lingering
    weather-related damage is realistic) as long as the RAMP START is
    truthfully inside real storm conditions."""
    candidates = list(storm_windows)
    RNG.shuffle(candidates)
    for window_start, window_end in candidates[:max_attempts]:
        ramp_start_idx = window_start
        outage_start_idx = ramp_start_idx + lead_steps
        outage_end_idx = min(outage_start_idx + dur_steps, n - 1)

        if outage_start_idx >= n - 1:
            continue  # would run off the end of the simulation window

        overlap = any(
            ramp_start_idx <= occ_end and occ_start <= outage_end_idx
            for occ_start, occ_end in occupied
        )
        if not overlap:
            return ramp_start_idx, outage_start_idx, outage_end_idx
    return None


def generate_failure_events(towers_df: pd.DataFrame, weather_df: pd.DataFrame,
                             timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(timestamps)
    interval_hours = (timestamps[1] - timestamps[0]).seconds / 3600
    steps_per_hour = round(1 / interval_hours)
    duration_years = (n * interval_hours) / (24 * 365)
    expected_events = FAILURES_PER_TOWER_PER_YEAR * duration_years

    types = list(FAILURE_TYPES.keys())
    weights = [FAILURE_TYPES[t]["weight"] for t in types]

    tower_region = dict(zip(towers_df["tower_id"], towers_df["region"]))

    # Precompute per-region storm windows once, reused for every tower in
    # that region (only actually used for weather_induced_degradation).
    storm_windows_by_region = {}
    min_storm_len = max(1, steps_per_hour)  # require >= 1h of continuous storm
    for region, group in weather_df.groupby("region"):
        storm_flags = group.sort_values("timestamp")["storm_flag"].values
        storm_windows_by_region[region] = _find_storm_windows(storm_flags, min_storm_len)

    events = []
    weather_anchored = 0
    weather_fallback = 0
    max_attempts = 20

    for tower_id in towers_df["tower_id"]:
        n_events = RNG.poisson(expected_events)
        if n_events == 0:
            continue

        occupied = []
        region = tower_region[tower_id]

        for _ in range(n_events):
            f_type = RNG.choice(types, p=weights)
            spec = FAILURE_TYPES[f_type]

            lead_h = RNG.uniform(*spec["lead_time_hours"])
            dur_h = RNG.uniform(*spec["outage_duration_hours"])
            lead_steps = int(lead_h * steps_per_hour)
            dur_steps = max(1, int(dur_h * steps_per_hour))

            margin = lead_steps + dur_steps + steps_per_hour
            if margin >= n - steps_per_hour:
                continue

            placement = None

            if f_type == "weather_induced_degradation" and storm_windows_by_region.get(region):
                placement = _try_place_in_storm(
                    storm_windows_by_region[region], lead_steps, dur_steps, n, occupied, max_attempts,
                )
                if placement is not None:
                    weather_anchored += 1
                else:
                    weather_fallback += 1

            if placement is None:
                placement = _try_place_random(
                    margin, n - steps_per_hour, lead_steps, dur_steps, n, occupied, max_attempts,
                )
                if placement is None:
                    continue  # couldn't find a free slot for this tower after several tries

            ramp_start_idx, outage_start_idx, outage_end_idx = placement
            occupied.append((ramp_start_idx, outage_end_idx))

            events.append({
                "tower_id": tower_id,
                "failure_type": f_type,
                "ramp_start": timestamps[ramp_start_idx],
                "outage_start": timestamps[outage_start_idx],
                "outage_end": timestamps[outage_end_idx],
                "ramp_start_idx": ramp_start_idx,
                "outage_start_idx": outage_start_idx,
                "outage_end_idx": outage_end_idx,
            })

    if weather_anchored or weather_fallback:
        print(f"Weather-induced events: {weather_anchored} anchored to real storm windows, "
              f"{weather_fallback} fell back to random placement (no usable storm window found in that region).")

    return pd.DataFrame(events)


if __name__ == "__main__":
    from towers import generate_towers
    from weather import load_weather

    ts = pd.date_range("2026-01-01", periods=96 * 60, freq="15min")
    towers_df = generate_towers()
    weather_df = load_weather(ts)
    ev = generate_failure_events(towers_df, weather_df, ts)
    print(ev[["tower_id", "failure_type", "ramp_start", "outage_start", "outage_end"]].head(10))
    print(f"\nTotal events: {len(ev)} across {ev['tower_id'].nunique()} towers")
    print(ev["failure_type"].value_counts())

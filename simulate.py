"""
Main simulation entry point. Produces a single long-format CSV where each
row is (tower_id, timestamp, metrics..., labels...).

Run:  python simulate.py

Output: data/telecom_timeseries.csv  (per-tower, per-timestamp metrics + labels)
        data/towers.csv              (static tower registry)
        data/failure_events.csv      (ground-truth event log, for RCA scoring)
"""

import os
import gzip
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import (
    N_TOWERS, SIM_DAYS, INTERVAL_MINUTES, PREDICTION_HORIZONS,
    START_DATE, FAILURE_TYPES, RNG,
)
from towers import generate_towers
from weather import load_weather
from behavior import generate_behavior
from failures import generate_failure_events

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")


def build_baseline(tower_row, timestamps, weather_row, behavior_row, n):
    """Baseline (no-failure) metric curves for a single tower."""
    load = behavior_row["load_multiplier"].values
    storm = weather_row["storm_flag"].values
    rain = weather_row["rainfall_mm"].values
    wind = weather_row["wind_kmh"].values

    capacity = tower_row["capacity_mbps"]

    utilization_pct = np.clip(45 * load + RNG.normal(0, 4, n), 3, 100)
    connected_users = np.clip(
        (capacity / 8) * load + RNG.normal(0, capacity / 100, n), 0, None
    ).round().astype(int)

    # Congestion penalty kicks in nonlinearly as utilization approaches 100
    congestion_penalty = np.clip((utilization_pct - 70) / 30, 0, 1) ** 2

    throughput_mbps = capacity * (utilization_pct / 100) * (1 - 0.5 * congestion_penalty)
    throughput_mbps = np.clip(throughput_mbps + RNG.normal(0, capacity * 0.02, n), 0, None)

    latency_ms = 15 + 40 * congestion_penalty + RNG.normal(0, 2, n)
    latency_ms = np.clip(latency_ms, 5, None)

    packet_loss_pct = (
        0.1 + 3 * congestion_penalty
        + 0.05 * rain + 0.02 * wind
        + RNG.normal(0, 0.05, n)
    )
    packet_loss_pct = np.clip(packet_loss_pct, 0, 100)

    rsrp_dbm = -75 - 0.4 * rain - 0.15 * wind + RNG.normal(0, 1.5, n)
    rsrp_dbm = np.clip(rsrp_dbm, -140, -50)

    grid_voltage = 230 + RNG.normal(0, 2, n)
    battery_level_pct = np.full(n, 100.0)

    return {
        "utilization_pct": utilization_pct,
        "connected_users": connected_users,
        "throughput_mbps": throughput_mbps,
        "latency_ms": latency_ms,
        "packet_loss_pct": packet_loss_pct,
        "rsrp_dbm": rsrp_dbm,
        "grid_voltage": grid_voltage,
        "battery_level_pct": battery_level_pct,
    }


def apply_failure(metrics, f_type, ramp_idx, outage_start_idx, outage_end_idx, n):
    """Mutates `metrics` in-place to reflect a failure ramp + outage window."""
    severity = np.zeros(n)
    ramp_len = max(1, outage_start_idx - ramp_idx)
    ramp_range = np.arange(ramp_idx, outage_start_idx)
    severity[ramp_range] = np.clip((ramp_range - ramp_idx) / ramp_len, 0, 1)
    severity[outage_start_idx:outage_end_idx + 1] = 1.0

    is_outage = np.zeros(n, dtype=int)
    is_outage[outage_start_idx:outage_end_idx + 1] = 1

    if f_type == "hardware_power_failure":
        voltage_drop = RNG.uniform(70, 100)
        battery_drain = RNG.uniform(80, 100)
        latency_spike = RNG.uniform(250, 350)

        metrics["grid_voltage"] -= severity * voltage_drop         # 230V -> ~130-160V
        metrics["battery_level_pct"] -= severity * battery_drain    # drains to ~0-20
        metrics["throughput_mbps"] *= (1 - severity)
        metrics["connected_users"] = (metrics["connected_users"] * (1 - severity)).astype(int)
        metrics["packet_loss_pct"] = metrics["packet_loss_pct"] + severity * (100 - metrics["packet_loss_pct"])
        metrics["latency_ms"] += severity * latency_spike

    elif f_type == "congestion_collapse":
        util_spike = RNG.uniform(30, 50)
        throughput_drop_frac = RNG.uniform(0.7, 0.95)
        latency_spike = RNG.uniform(200, 300)
        loss_spike = RNG.uniform(45, 70)

        metrics["utilization_pct"] = np.clip(metrics["utilization_pct"] + severity * util_spike, 0, 100)
        metrics["throughput_mbps"] *= (1 - throughput_drop_frac * severity)
        metrics["latency_ms"] += severity * latency_spike
        metrics["packet_loss_pct"] = np.clip(metrics["packet_loss_pct"] + severity * loss_spike, 0, 100)

    elif f_type == "weather_induced_degradation":
        rsrp_drop = RNG.uniform(20, 40)
        loss_spike = RNG.uniform(35, 65)
        latency_spike = RNG.uniform(100, 200)
        throughput_drop_frac = RNG.uniform(0.45, 0.75)

        metrics["rsrp_dbm"] -= severity * rsrp_drop
        metrics["packet_loss_pct"] = np.clip(metrics["packet_loss_pct"] + severity * loss_spike, 0, 100)
        metrics["latency_ms"] += severity * latency_spike
        metrics["throughput_mbps"] *= (1 - throughput_drop_frac * severity)

    elif f_type == "capacity_creep":
        util_spike = RNG.uniform(35, 55)
        throughput_drop_frac = RNG.uniform(0.55, 0.85)
        latency_spike = RNG.uniform(150, 250)
        loss_spike = RNG.uniform(40, 70)

        metrics["utilization_pct"] = np.clip(metrics["utilization_pct"] + severity * util_spike, 0, 100)
        metrics["throughput_mbps"] *= (1 - throughput_drop_frac * severity)
        metrics["latency_ms"] += severity * latency_spike
        metrics["packet_loss_pct"] = np.clip(metrics["packet_loss_pct"] + severity * loss_spike, 0, 100)

    return severity, is_outage


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    timestamps = pd.date_range(START_DATE, periods=int(SIM_DAYS * 24 * 60 / INTERVAL_MINUTES),
                                freq=f"{INTERVAL_MINUTES}min")
    n = len(timestamps)
    steps_per_hour = round(60 / INTERVAL_MINUTES)

    print(f"Simulating {N_TOWERS} towers x {n} timestamps ({SIM_DAYS} days)...")

    towers_df = generate_towers()
    weather_df = load_weather(timestamps)
    behavior_df, mobility_events_df = generate_behavior(timestamps)
    events_df = generate_failure_events(towers_df, weather_df, timestamps)

    weather_by_region = {r: g.reset_index(drop=True) for r, g in weather_df.groupby("region")}
    behavior_by_region = {r: g.reset_index(drop=True) for r, g in behavior_df.groupby("region")}
    events_by_tower = {t: g for t, g in events_df.groupby("tower_id")} if len(events_df) else {}

    # IMPROVEMENT #6: precompute step-counts for every prediction horizon
    horizon_steps_map = {label: max(1, round(hours * steps_per_hour)) for label, hours in PREDICTION_HORIZONS}

    parquet_writer = None
    parquet_path = os.path.join(OUT_DIR, "telecom_timeseries.parquet")
    year_files = {}  # year -> open gzip file handle, header written once each
    total_rows = 0
    outage_rows = 0
    degraded_root_causes = {}
    horizon_positive_counts = {label: 0 for label, _ in PREDICTION_HORIZONS}

    for i, (_, trow) in enumerate(towers_df.iterrows()):
        if i % 10 == 0:
            print(f"  ...tower {i+1}/{N_TOWERS}")
        tower_id = trow["tower_id"]
        region = trow["region"]
        w = weather_by_region[region]
        b = behavior_by_region[region]

        metrics = build_baseline(trow, timestamps, w, b, n)

        is_degraded = np.zeros(n, dtype=int)
        is_outage = np.zeros(n, dtype=int)
        root_cause = np.array(["none"] * n, dtype=object)

        if tower_id in events_by_tower:
            for _, ev in events_by_tower[tower_id].iterrows():
                sev, outage_flag = apply_failure(
                    metrics, ev["failure_type"],
                    ev["ramp_start_idx"], ev["outage_start_idx"], ev["outage_end_idx"], n,
                )
                touched = sev > 0
                is_degraded[touched] = 1
                is_outage = np.maximum(is_outage, outage_flag)
                root_cause[touched] = ev["failure_type"]

        # IMPROVEMENT #6: forward-looking prediction label for EVERY horizon.
        # outage_in_next_15m, _30m, _1h, _3h, _6h, _12h, _24h -- each answers
        # "will an outage START within this many hours from this timestamp?"
        outage_start_indices = np.where(np.diff(np.concatenate([[0], is_outage])) == 1)[0]
        horizon_labels = {}
        for label, hsteps in horizon_steps_map.items():
            col = np.zeros(n, dtype=int)
            for idx in outage_start_indices:
                lo = max(0, idx - hsteps)
                col[lo:idx] = 1
            horizon_labels[f"outage_in_next_{label}"] = col

        for col_name, col in horizon_labels.items():
            label = col_name.removeprefix("outage_in_next_")
            horizon_positive_counts[label] += int(col.sum())

        frame = pd.DataFrame({
            "tower_id": tower_id,
            "region": region,
            "timestamp": timestamps,
            **metrics,
            "storm_flag": w["storm_flag"].values,
            "rainfall_mm": w["rainfall_mm"].values,
            "wind_kmh": w["wind_kmh"].values,
            "load_multiplier": b["load_multiplier"].values,
            "is_degraded": is_degraded,
            "is_outage": is_outage,
            "root_cause": root_cause,
            **horizon_labels,
        })
        frame["connected_users"] = frame["connected_users"].clip(lower=0)
        frame = frame.round({
            "utilization_pct": 2, "throughput_mbps": 2, "latency_ms": 2,
            "packet_loss_pct": 3, "rsrp_dbm": 2, "grid_voltage": 2, "battery_level_pct": 2,
        })

        # --- stream this tower's slice straight to disk, then drop it ---
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(parquet_path, table.schema, compression="snappy")
        parquet_writer.write_table(table)

        for year, ydf in frame.groupby(frame["timestamp"].dt.year):
            path = os.path.join(OUT_DIR, f"telecom_timeseries_{year}.csv.gz")
            write_header = year not in year_files
            year_files[year] = True
            mode = "wt" if write_header else "at"
            with gzip.open(path, mode) as fh:
                ydf.to_csv(fh, index=False, header=write_header)

        total_rows += len(frame)
        outage_rows += int(is_outage.sum())
        for rc, cnt in zip(*np.unique(root_cause[is_degraded == 1], return_counts=True)):
            degraded_root_causes[rc] = degraded_root_causes.get(rc, 0) + int(cnt)

        del frame, table, metrics  # free memory before next tower

    if parquet_writer is not None:
        parquet_writer.close()

    towers_df.to_csv(os.path.join(OUT_DIR, "towers.csv"), index=False)
    events_df.to_csv(os.path.join(OUT_DIR, "failure_events.csv"), index=False)
    mobility_events_df.to_csv(os.path.join(OUT_DIR, "mobility_events.csv"), index=False)

    print(f"\nDone. {total_rows:,} rows written to {parquet_path}")
    print(f"Per-year gzipped CSVs also written to {OUT_DIR}/telecom_timeseries_<year>.csv.gz")
    print(f"Label columns written for horizons: {[label for label, _ in PREDICTION_HORIZONS]}")
    print(f"Mobility events (festival/cricket_match/concert/emergency) logged to {OUT_DIR}/mobility_events.csv")
    print(f"Towers with >=1 failure event: {events_df['tower_id'].nunique() if len(events_df) else 0} / {N_TOWERS}")
    print(f"Total failure events: {len(events_df)}")
    print(f"\nRoot cause distribution (rows where is_degraded=1): {degraded_root_causes}")
    print(f"Outage rate: {outage_rows/total_rows*100:.4f}% of all rows")

    print("\nPositive rate by horizon (target: 15m 0.3-1%, 30m 0.5-1.5%, "
          "1h 1-2%, 3h 3-6%, 6h 5-10%):")
    for label, _ in PREDICTION_HORIZONS:
        pos = horizon_positive_counts[label]
        print(f"  {label:>4}: {pos:>10,} positives  ({pos/total_rows*100:.4f}%)")


if __name__ == "__main__":
    main()

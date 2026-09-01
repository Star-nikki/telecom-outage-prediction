"""
IMPROVEMENT #1: Streaming Simulator.

Turns the static historical dataset into a live "feed" -- every 15 simulated
minutes, a new tick of data arrives for all towers at once, exactly like a
real monitoring system would receive it. Agents can subscribe to this feed
and react as data arrives, instead of analyzing a finished CSV.

This does NOT regenerate the data differently -- it reads the same Parquet
file simulate.py already produced, and replays it in time order as a
generator. That's the honest way to do this: the underlying model is the
same, only the *delivery mechanism* changes.

Usage (from the scripts/ folder, after simulate.py has been run once):

    python streaming_simulator.py

By default it fast-forwards through history (see config.STREAMING_SPEED_MULTIPLIER)
so you don't wait 3 years for a demo -- set it to 1.0 for true real-time
pacing (one tick every real 15 minutes).
"""

import time
import pandas as pd

from config import STREAMING_SPEED_MULTIPLIER

DATA_PATH = "data/telecom_timeseries.parquet"


def stream_ticks(data_path=DATA_PATH, speed_multiplier=STREAMING_SPEED_MULTIPLIER, start_at=None):
    """
    Generator that yields one tick (a DataFrame of all towers) at a time,
    in timestamp order, optionally sleeping between ticks to simulate
    real-world pacing.

    speed_multiplier: how much faster than real-time to play back.
        1.0    = real pacing (sleeps ~15 real minutes between ticks -- slow!)
        3600   = default; ~0.25 real seconds between ticks
        None/0 = no sleeping at all, replay as fast as possible
    """
    print(f"Loading dataset from {data_path} ...")
    df = pd.read_parquet(data_path)
    df = df.sort_values("timestamp")

    timestamps = df["timestamp"].unique()
    if start_at is not None:
        timestamps = timestamps[timestamps >= pd.Timestamp(start_at)]

    print(f"Streaming {len(timestamps):,} ticks across {df['tower_id'].nunique()} towers...")
    real_seconds_per_tick = 15 * 60  # each tick represents 15 simulated minutes

    grouped = {ts: g for ts, g in df.groupby("timestamp")}

    for ts in timestamps:
        tick_df = grouped[ts]
        yield ts, tick_df

        if speed_multiplier:
            time.sleep(real_seconds_per_tick / speed_multiplier)


def default_alert_handler(timestamp, tick_df):
    """
    A minimal example of what an 'agent' reacting to live data looks like.
    Replace this with actual calls into your Tower Health / Traffic /
    Decision Agents once they exist -- this just demonstrates the pattern:
    flag anything degraded or predicted to fail soon, right as it arrives.
    """
    alerts = tick_df[(tick_df["is_degraded"] == 1) | (tick_df["outage_in_next_1h"] == 1)]
    if len(alerts):
        for _, row in alerts.iterrows():
            status = "OUTAGE NOW" if row["is_outage"] == 1 else "AT RISK"
            print(f"[{timestamp}] ALERT ({status}) tower={row['tower_id']} "
                  f"cause={row['root_cause']} util={row['utilization_pct']:.1f}% "
                  f"battery={row['battery_level_pct']:.1f}% packet_loss={row['packet_loss_pct']:.2f}%")


if __name__ == "__main__":
    print("Starting streaming simulator (Ctrl+C to stop)...")
    print(f"Speed multiplier: {STREAMING_SPEED_MULTIPLIER}x "
          f"({'true real-time' if STREAMING_SPEED_MULTIPLIER == 1 else 'fast-forwarded'})\n")

    tick_count = 0
    try:
        for timestamp, tick_df in stream_ticks():
            tick_count += 1
            if tick_count % 500 == 0:
                print(f"... {tick_count:,} ticks processed, currently at {timestamp}")
            default_alert_handler(timestamp, tick_df)
    except KeyboardInterrupt:
        print(f"\nStopped after {tick_count:,} ticks.")

"""
IMPROVEMENT #2: pull REAL historical weather instead of fully synthetic
weather, using the Open-Meteo Archive API (free, no API key required).

Run this BEFORE simulate.py. It writes data/real_weather.csv, which
simulate.py automatically detects and prefers over synthetic weather.

IMPORTANT LIMITATION (read this): Open-Meteo's archive only has data for
days that have actually happened. If your simulation window includes dates
in the future relative to when you run this script (e.g. late 2026 if
you're running this mid-2026), those dates obviously don't have real
weather yet. For any date beyond "today", this script automatically falls
back to the same synthetic generator used elsewhere in the project (see
weather.py), and prints a clear warning telling you exactly which date
range was backfilled synthetically. This is an honest limitation to state
in your report, not something to hide.

Usage:
    pip install requests pandas numpy --break-system-packages
    python fetch_historical_weather.py
"""

import os
import time
import datetime
import numpy as np
import pandas as pd
import requests

from config import N_REGIONS, START_DATE, SIM_DAYS, RNG
from weather import generate_weather  # synthetic fallback

OUT_PATH = "data/real_weather.csv"

# One real-world city per region -- pick actual coordinates relevant to your
# network's footprint. These are placeholders (major Indian cities); change
# them to match where your towers actually are.
REGION_COORDINATES = {
    0: {"name": "Mumbai",    "lat": 19.0760, "lon": 72.8777},
    1: {"name": "Delhi",     "lat": 28.7041, "lon": 77.1025},
    2: {"name": "Bengaluru", "lat": 12.9716, "lon": 77.5946},
    3: {"name": "Chennai",   "lat": 13.0827, "lon": 80.2707},
    4: {"name": "Kolkata",   "lat": 22.5726, "lon": 88.3639},
    5: {"name": "Hyderabad", "lat": 17.3850, "lon": 78.4867},
}

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_region_weather(lat, lon, start_date, end_date):
    """One Open-Meteo call per region for the whole available date range."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,precipitation,windspeed_10m,weathercode",
        "timezone": "auto",
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def to_15min_dataframe(json_data, region_label):
    """Open-Meteo gives hourly data; upsample to 15-min to match the rest
    of the simulation, holding each hour's value across its four 15-min
    sub-intervals (no interpolation -- simplest honest approach)."""
    hourly = json_data["hourly"]
    hourly_ts = pd.to_datetime(hourly["time"])

    df = pd.DataFrame({
        "timestamp": hourly_ts,
        "temp_c": hourly["temperature_2m"],
        "rainfall_mm": hourly["precipitation"],
        "wind_kmh": np.array(hourly["windspeed_10m"]) * 1.0,  # already km/h from Open-Meteo
        "weathercode": hourly["weathercode"],
    })
    # WMO weather codes 95-99 = thunderstorm; treat as storm_flag
    df["storm_flag"] = df["weathercode"].isin([95, 96, 99]).astype(int)
    df = df.drop(columns=["weathercode"])

    df = df.set_index("timestamp").resample("15min").ffill().reset_index()
    df["region"] = region_label
    return df[["region", "timestamp", "temp_c", "rainfall_mm", "wind_kmh", "storm_flag"]]


def main():
    os.makedirs("data", exist_ok=True)

    sim_start = pd.Timestamp(START_DATE)
    sim_end = sim_start + pd.Timedelta(days=SIM_DAYS - 1)
    today = pd.Timestamp(datetime.date.today())

    real_end = min(sim_end, today - pd.Timedelta(days=2))  # archive lags ~2 days behind real-time
    print(f"Simulation window: {sim_start.date()} to {sim_end.date()}")

    all_frames = []

    if real_end >= sim_start:
        print(f"Fetching REAL historical weather: {sim_start.date()} to {real_end.date()}")
        for region_id in range(N_REGIONS):
            coord = REGION_COORDINATES.get(region_id)
            if coord is None:
                print(f"  No coordinates configured for region {region_id}, skipping (will backfill synthetic)")
                continue
            print(f"  Region R{region_id} ({coord['name']})...")
            try:
                data = fetch_region_weather(
                    coord["lat"], coord["lon"],
                    sim_start.strftime("%Y-%m-%d"), real_end.strftime("%Y-%m-%d"),
                )
                rdf = to_15min_dataframe(data, f"R{region_id}")
                all_frames.append(rdf)
                time.sleep(1)  # be polite to the free API
            except Exception as e:
                print(f"    FAILED for region {region_id}: {e}. Will backfill synthetic for this region.")
    else:
        print("Entire simulation window is in the future -- no real data available at all.")

    real_df = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame(
        columns=["region", "timestamp", "temp_c", "rainfall_mm", "wind_kmh", "storm_flag"]
    )

    # ---- backfill any gap (future dates, missing regions) with synthetic ----
    full_timestamps = pd.date_range(sim_start, periods=SIM_DAYS * 96, freq="15min")
    synthetic_df = generate_weather(full_timestamps)

    covered = set(zip(real_df["region"], real_df["timestamp"])) if len(real_df) else set()
    missing_mask = ~synthetic_df.apply(lambda r: (r["region"], r["timestamp"]) in covered, axis=1)
    backfill_df = synthetic_df[missing_mask]

    if len(backfill_df):
        print(f"\nBackfilling {len(backfill_df):,} rows with synthetic weather "
              f"(future dates and/or regions with no real data).")
        if len(backfill_df):
            print(f"Backfilled date range: {backfill_df['timestamp'].min()} to {backfill_df['timestamp'].max()}")

    final_df = pd.concat([real_df, backfill_df], ignore_index=True).sort_values(["region", "timestamp"])
    final_df.to_csv(OUT_PATH, index=False)
    print(f"\nDone. Wrote {len(final_df):,} rows to {OUT_PATH}")
    print(f"Real rows: {len(real_df):,} | Synthetic backfill rows: {len(backfill_df):,}")


if __name__ == "__main__":
    main()

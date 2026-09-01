"""
Generates per-region weather time series: rainfall, temperature, wind, storms.
Weather is shared across all towers in the same region (they're geographically
co-located), which is what lets the Weather Agent's signal correlate with
multiple towers' degradation at once -- a realistic pattern (a storm doesn't
hit one tower, it hits a region).
"""

import os
import numpy as np
import pandas as pd
from config import N_REGIONS, RNG, REAL_WEATHER_PATH


def load_weather(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """
    IMPROVEMENT #2: prefer real historical weather (data/real_weather.csv,
    produced by fetch_historical_weather.py) if it exists; otherwise fall
    back to the synthetic generator below. This is what simulate.py calls.
    """
    if os.path.exists(REAL_WEATHER_PATH):
        print(f"Using REAL weather data from {REAL_WEATHER_PATH}")
        df = pd.read_csv(REAL_WEATHER_PATH, parse_dates=["timestamp"])
        return df
    print("No real weather file found -- using fully synthetic weather "
          "(run fetch_historical_weather.py first if you want real data).")
    return generate_weather(timestamps)


def generate_weather(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(timestamps)
    hours = timestamps.hour.values
    day_of_year = timestamps.dayofyear.values

    frames = []
    for r in range(N_REGIONS):
        # Base seasonal temperature curve + daily oscillation + noise
        base_temp = 27 + 5 * np.sin(2 * np.pi * day_of_year / 365) 
        daily_cycle = 4 * np.sin(2 * np.pi * (hours - 6) / 24)
        temp = base_temp + daily_cycle + RNG.normal(0, 1.0, n)

        # Storms: rare, region-specific, last a few hours, bring rain + wind
        storm_flag = np.zeros(n, dtype=int)
        rainfall = np.abs(RNG.normal(0.2, 0.3, n))  # light background drizzle noise
        wind = np.abs(RNG.normal(8, 3, n))

        # Inject ~1 storm every ~10 days per region, lasting 2-8 hours
        n_storms = max(1, int(n / (10 * 24 * 4)))  # 4 samples/hour at 15-min interval
        storm_starts = RNG.choice(np.arange(n - 40), size=n_storms, replace=False)
        for s in storm_starts:
            duration = RNG.integers(8, 32)  # in 15-min steps -> 2 to 8 hours
            end = min(s + duration, n)
            storm_flag[s:end] = 1
            rainfall[s:end] += RNG.uniform(5, 30, end - s)
            wind[s:end] += RNG.uniform(15, 45, end - s)

        frames.append(pd.DataFrame({
            "region": f"R{r}",
            "timestamp": timestamps,
            "temp_c": np.round(temp, 1),
            "rainfall_mm": np.round(rainfall, 2),
            "wind_kmh": np.round(wind, 1),
            "storm_flag": storm_flag,
        }))

    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    ts = pd.date_range("2026-01-01", periods=96 * 30, freq="15min")
    df = generate_weather(ts)
    print(df.head())
    print(f"\nStorm hours by region:\n{df.groupby('region')['storm_flag'].sum() / 4}")

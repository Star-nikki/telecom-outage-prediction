"""
Builds the static tower registry: identity, location, type, capacity.
"""

import pandas as pd
import numpy as np
from config import N_TOWERS, N_REGIONS, TOWER_TYPES, RNG

# Rough India-wide bounding box just to get plausible lat/lon spread.
LAT_RANGE = (8.0, 28.0)
LON_RANGE = (72.0, 88.0)


def generate_towers() -> pd.DataFrame:
    types = list(TOWER_TYPES.keys())
    weights = [TOWER_TYPES[t]["weight"] for t in types]

    rows = []
    for i in range(N_TOWERS):
        tower_type = RNG.choice(types, p=weights)
        spec = TOWER_TYPES[tower_type]
        region_id = i % N_REGIONS  # evenly distribute towers across regions

        rows.append({
            "tower_id": f"T{i:04d}",
            "region": f"R{region_id}",
            "tower_type": tower_type,
            "capacity_mbps": spec["capacity_mbps"],
            "backup_battery_hours": spec["battery_hours"],
            "lat": round(RNG.uniform(*LAT_RANGE), 4),
            "lon": round(RNG.uniform(*LON_RANGE), 4),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_towers()
    print(df.head(10))
    print(f"\n{len(df)} towers across {df['region'].nunique()} regions")
    print(df["tower_type"].value_counts())

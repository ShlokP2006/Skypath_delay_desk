"""Generate SYNTHETIC train/test files with the same columns as the real dataset.

Purpose: let you run the whole pipeline (training script and Streamlit app)
in a minute, before you point it at the 1 GB train.csv. The numbers are made up,
so the charts and predictions from demo data mean nothing. The app shows a
banner whenever it is running on demo artifacts.

    python make_demo_data.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

BLOCKS = ["0001-0559"] + [f"{h:02d}00-{h:02d}59" for h in range(6, 24)]

# name: (lat, lon, flights/month, passengers/month, delay bias, cold climate)
AIRPORTS = {
    "Hartsfield-Jackson Atlanta International": (33.64, -84.43, 33000, 3_400_000, 0.05, 0),
    "Chicago O'Hare International": (41.98, -87.90, 30000, 3_000_000, 0.35, 1),
    "Dallas/Fort Worth International": (32.90, -97.04, 27000, 2_800_000, 0.10, 0),
    "Denver International": (39.86, -104.67, 24000, 2_500_000, 0.15, 1),
    "Los Angeles International": (33.94, -118.41, 22000, 3_100_000, 0.05, 0),
    "John F. Kennedy International": (40.64, -73.78, 15000, 2_200_000, 0.40, 1),
    "San Francisco International": (37.62, -122.38, 17000, 2_100_000, 0.35, 0),
    "Seattle/Tacoma International": (47.45, -122.31, 15000, 1_900_000, -0.10, 1),
    "Miami International": (25.79, -80.29, 14000, 1_700_000, 0.05, 0),
    "Logan International": (42.36, -71.01, 13000, 1_500_000, 0.30, 1),
    "Phoenix Sky Harbor International": (33.43, -112.01, 15000, 1_800_000, -0.20, 0),
    "McCarran International": (36.08, -115.15, 14000, 1_700_000, -0.10, 0),
}

# name: (flights/month, passengers/month, cabin crew per pax, ground staff per pax, seats, bias)
CARRIERS = {
    "Southwest Airlines Co.": (110_000, 12_000_000, 0.00012, 0.00030, 143, 0.10),
    "Delta Air Lines Inc.": (85_000, 11_000_000, 0.00018, 0.00040, 160, -0.20),
    "American Airlines Inc.": (90_000, 10_500_000, 0.00016, 0.00035, 150, 0.10),
    "United Air Lines Inc.": (70_000, 8_500_000, 0.00017, 0.00036, 155, 0.15),
    "JetBlue Airways": (30_000, 3_500_000, 0.00019, 0.00028, 150, 0.30),
    "Alaska Airlines Inc.": (25_000, 2_600_000, 0.00021, 0.00033, 165, -0.25),
    "SkyWest Airlines Inc.": (60_000, 2_800_000, 0.00030, 0.00025, 70, 0.15),
}


def generate(n: int = 60_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ap_names = list(AIRPORTS)
    cr_names = list(CARRIERS)

    airport = rng.choice(ap_names, n)
    carrier = rng.choice(cr_names, n)
    month = rng.integers(1, 13, n)
    dow = rng.integers(1, 8, n)

    block_w = np.array([0.03, 0.05] + [0.06] * 12 + [0.05, 0.04, 0.03, 0.02, 0.01])
    assert len(block_w) == len(BLOCKS)
    block_w = block_w / block_w.sum()
    block_idx = rng.choice(len(BLOCKS), n, p=block_w)
    block = np.array(BLOCKS)[block_idx]

    seg = np.minimum(rng.geometric(0.45, n), 8)
    prev = np.where(seg == 1, "NONE", rng.choice(ap_names, n))
    distance = rng.integers(1, 12, n)

    ap_info = np.array([AIRPORTS[a] for a in airport], dtype=float)
    cr_info = np.array([CARRIERS[c] for c in carrier], dtype=float)

    lat, lon = ap_info[:, 0], ap_info[:, 1]
    ap_flights, ap_pass, ap_bias, cold = ap_info[:, 2], ap_info[:, 3], ap_info[:, 4], ap_info[:, 5]
    cr_flights, cr_pass, fa, gs, seats_typ, cr_bias = cr_info.T

    seats = np.clip(seats_typ + rng.choice([-20, 0, 0, 0, 20, 30], n), 30, 300)
    age = np.clip(rng.gamma(4.0, 3.0, n), 0, 35)
    pair_flights = ap_flights * cr_flights / sum(v[0] for v in CARRIERS.values())

    lam = 3 + (ap_flights / 4000) * (0.5 + block_idx.clip(0, 18) / 9)
    concurrent = rng.poisson(lam) + 1

    seasonal = 60 + 25 * np.sin((month - 4) / 12 * 2 * np.pi)
    tmax = seasonal - 14 * cold + rng.normal(0, 9, n)
    prcp = np.where(rng.random(n) < 0.25, rng.exponential(0.3, n), 0.0)
    snow = np.where((tmax < 36) & (rng.random(n) < 0.30), rng.exponential(1.5, n), 0.0)
    snwd = np.where(tmax < 36, snow * 1.5 + rng.exponential(0.8, n) * (rng.random(n) < 0.3), 0.0)
    awnd = np.clip(rng.normal(8.5, 3.2, n), 1, 30)

    logit = (
        -2.6
        + 0.05 * concurrent
        + 1.1 * (block_idx / 18) ** 1.5
        + 0.30 * np.isin(month, [6, 7, 12])
        + 0.28 * np.minimum(snow, 6)
        + 0.9 * np.minimum(prcp, 2)
        + 0.04 * np.maximum(awnd - 10, 0)
        + ap_bias
        + cr_bias
        + 0.12 * (seg - 1)
        - 0.15 * (dow == 6)
        + 0.004 * age
        + rng.normal(0, 0.35, n)
    )
    delayed = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    return pd.DataFrame(
        {
            "MONTH": month,
            "DAY_OF_WEEK": dow,
            "DEP_DEL15": delayed,
            "DISTANCE_GROUP": distance,
            "DEP_TIME_BLK": block,  # raw-file name; the loader maps it to DEP_BLOCK
            "SEGMENT_NUMBER": seg,
            "CONCURRENT_FLIGHTS": concurrent,
            "NUMBER_OF_SEATS": seats.astype(int),
            "CARRIER_NAME": carrier,
            "AIRPORT_FLIGHTS_MONTH": ap_flights.astype(int),
            "AIRLINE_FLIGHTS_MONTH": cr_flights.astype(int),
            "AIRLINE_AIRPORT_FLIGHTS_MONTH": pair_flights.round().astype(int),
            "AVG_MONTHLY_PASS_AIRPORT": ap_pass.astype(int),
            "AVG_MONTHLY_PASS_AIRLINE": cr_pass.astype(int),
            "FLT_ATTENDANTS_PER_PASS": fa,
            "GROUND_SERV_PER_PASS": gs,
            "PLANE_AGE": age.round().astype(int),
            "DEPARTING_AIRPORT": airport,
            "LATITUDE": lat,
            "LONGITUDE": lon,
            "PREVIOUS_AIRPORT": prev,
            "PRCP": prcp.round(2),
            "SNOW": snow.round(1),
            "SNWD": snwd.round(1),
            "TMAX": tmax.round(0),
            "AWND": awnd.round(1),
        }
    )


def main() -> None:
    config.DATA_DIR.mkdir(exist_ok=True)
    train = generate(60_000, seed=1)
    test = generate(20_000, seed=2)
    train.to_csv(config.DATA_DIR / "demo_train.csv", index=False)
    test.to_csv(config.DATA_DIR / "demo_test.csv", index=False)
    print(f"Wrote demo_train.csv ({len(train):,} rows) and demo_test.csv ({len(test):,} rows) to {config.DATA_DIR}")
    print(f"Synthetic delay rate: {train['DEP_DEL15'].mean():.1%}")


if __name__ == "__main__":
    main()

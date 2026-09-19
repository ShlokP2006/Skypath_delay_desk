"""Small summary tables that the app reads instead of the 1 GB CSV.

Two things are built here:

* lookups - typical values (airport size, airline staffing, weather by month...)
  used to pre-fill the flight plan form, because a passenger does not know
  "average airline flights per month".
* eda - delay rates sliced by time, place, fleet and weather for the charts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

T = config.TARGET


def jsonable(x):
    """Recursively make numpy / NaN values safe for json.dump."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 4)
    if isinstance(x, np.integer):
        return int(x)
    return x


# --------------------------------------------------------------------------
# Lookups for the form
# --------------------------------------------------------------------------
def build_lookups(df: pd.DataFrame) -> dict:
    airports = df.groupby("DEPARTING_AIRPORT", observed=True).agg(
        latitude=("LATITUDE", "mean"),
        longitude=("LONGITUDE", "mean"),
        airport_flights_month=("AIRPORT_FLIGHTS_MONTH", "mean"),
        avg_monthly_pass=("AVG_MONTHLY_PASS_AIRPORT", "mean"),
        flights=(T, "size"),
        delay_rate=(T, "mean"),
    )
    carriers = df.groupby("CARRIER_NAME", observed=True).agg(
        airline_flights_month=("AIRLINE_FLIGHTS_MONTH", "mean"),
        avg_monthly_pass=("AVG_MONTHLY_PASS_AIRLINE", "mean"),
        flt_attendants_per_pass=("FLT_ATTENDANTS_PER_PASS", "mean"),
        ground_serv_per_pass=("GROUND_SERV_PER_PASS", "mean"),
        seats=("NUMBER_OF_SEATS", "median"),
        plane_age=("PLANE_AGE", "median"),
        flights=(T, "size"),
        delay_rate=(T, "mean"),
    )

    pair_means = df.groupby(["CARRIER_NAME", "DEPARTING_AIRPORT"], observed=True)[
        "AIRLINE_AIRPORT_FLIGHTS_MONTH"
    ].mean()
    pairs = {f"{c}||{a}": v for (c, a), v in pair_means.items()}

    conc_means = df.groupby(["DEPARTING_AIRPORT", "DEP_BLOCK"], observed=True)[
        "CONCURRENT_FLIGHTS"
    ].mean()
    concurrent: dict = {}
    for (airport, block), value in conc_means.items():
        concurrent.setdefault(airport, {})[block] = value

    wx_means = df.groupby(["DEPARTING_AIRPORT", "MONTH"], observed=True)[config.WEATHER].mean()
    weather: dict = {}
    for (airport, month), row in wx_means.iterrows():
        weather.setdefault(airport, {})[str(int(month))] = row.to_dict()

    ranges = {
        w: [float(df[w].quantile(0.001)), float(df[w].quantile(0.999))] for w in config.WEATHER
    }

    previous = sorted(set(df["PREVIOUS_AIRPORT"].astype(str).unique()) - {config.FIRST_LEG})

    lookups = {
        "airports": airports.to_dict("index"),
        "carriers": carriers.to_dict("index"),
        "pairs": pairs,
        "concurrent": concurrent,
        "weather": weather,
        "weather_range": ranges,
        "previous_airports": [config.FIRST_LEG] + previous,
        "dep_blocks": sorted(df["DEP_BLOCK"].astype(str).unique()),
        "max_concurrent": float(df["CONCURRENT_FLIGHTS"].quantile(0.999)),
        "max_plane_age": float(df["PLANE_AGE"].quantile(0.999)),
        "seat_range": [
            float(df["NUMBER_OF_SEATS"].quantile(0.001)),
            float(df["NUMBER_OF_SEATS"].quantile(0.999)),
        ],
    }
    return jsonable(lookups)


# --------------------------------------------------------------------------
# EDA summaries for the charts
# --------------------------------------------------------------------------
def _rate(series_key, df: pd.DataFrame) -> list[dict]:
    table = df.groupby(series_key, observed=True)[T].agg(["mean", "size"]).reset_index()
    table.columns = ["label", "rate", "n"]
    return table.to_dict("records")


def _binned(df: pd.DataFrame, col: str, edges: list, labels: list) -> list[dict]:
    bins = pd.cut(df[col], bins=edges, labels=labels, include_lowest=True)
    table = df.groupby(bins, observed=True)[T].agg(["mean", "size"]).reset_index()
    table.columns = ["label", "rate", "n"]
    table["label"] = table["label"].astype(str)
    return table.to_dict("records")


def build_eda(df: pd.DataFrame) -> dict:
    inf = 1e9
    heat = df.groupby(["DAY_OF_WEEK", "DEP_BLOCK"], observed=True)[T].mean().unstack()

    eda = {
        "overall": {"rows": int(len(df)), "delay_rate": float(df[T].mean())},
        "month": _rate(df["MONTH"].astype(int), df),
        "day_of_week": _rate(df["DAY_OF_WEEK"].astype(int), df),
        "dep_block": _rate(df["DEP_BLOCK"], df),
        "distance_group": _rate(df["DISTANCE_GROUP"].astype(int), df),
        "segment": _rate(df["SEGMENT_NUMBER"].clip(upper=8).astype(int), df),
        "heat": {
            "days": [int(d) for d in heat.index],
            "blocks": [str(b) for b in heat.columns],
            "z": heat.values.tolist(),
        },
        "weather": {
            "Rain": _binned(
                df, "PRCP", [-0.001, 0.0, 0.1, 0.25, 0.5, 1.0, inf],
                ["Dry", "Under 0.1 in", "0.1 to 0.25", "0.25 to 0.5", "0.5 to 1", "Over 1 in"],
            ),
            "Snow": _binned(
                df, "SNOW", [-0.001, 0.0, 0.5, 2.0, 5.0, inf],
                ["None", "Under 0.5 in", "0.5 to 2", "2 to 5", "Over 5 in"],
            ),
            "Snow on the ground": _binned(
                df, "SNWD", [-0.001, 0.0, 1.0, 3.0, 6.0, inf],
                ["None", "Up to 1 in", "1 to 3", "3 to 6", "Over 6 in"],
            ),
            "Wind": _binned(
                df, "AWND", [-0.001, 5, 8, 11, 14, 18, inf],
                ["Under 5", "5 to 8", "8 to 11", "11 to 14", "14 to 18", "Over 18"],
            ),
            "Temperature": _binned(
                df, "TMAX", [-inf, 20, 32, 50, 70, 85, 95, inf],
                ["Under 20", "20 to 32", "32 to 50", "50 to 70", "70 to 85", "85 to 95", "Over 95"],
            ),
        },
        "concurrent": _binned(
            df, "CONCURRENT_FLIGHTS", [-0.001, 5, 10, 15, 20, 30, 45, inf],
            ["1 to 5", "6 to 10", "11 to 15", "16 to 20", "21 to 30", "31 to 45", "Over 45"],
        ),
        "plane_age": _binned(
            df, "PLANE_AGE", [-0.001, 2, 5, 10, 15, 20, 25, inf],
            ["0 to 2", "3 to 5", "6 to 10", "11 to 15", "16 to 20", "21 to 25", "Over 25"],
        ),
        "seats": _binned(
            df, "NUMBER_OF_SEATS", [-0.001, 50, 100, 150, 200, 300, inf],
            ["Up to 50", "51 to 100", "101 to 150", "151 to 200", "201 to 300", "Over 300"],
        ),
    }
    return jsonable(eda)

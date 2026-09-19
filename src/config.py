"""Central configuration for the delay desk.

Everything that describes the dataset (column names, feature groups, labels)
lives here so the training script and the app can never drift apart.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
DOCS_DIR = ROOT / "docs"

MODEL_PATH = ARTIFACT_DIR / "model.joblib"
LOOKUPS_PATH = ARTIFACT_DIR / "lookups.json"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
EDA_PATH = ARTIFACT_DIR / "eda.json"

# --------------------------------------------------------------------------
# Branding
# --------------------------------------------------------------------------
APP_NAME = "SkyPath Delay Desk"
APP_TAGLINE = "Check the odds of an on-time pushback before you get to the gate."

# --------------------------------------------------------------------------
# Dataset columns (see docs/train_sets_documentation.txt)
# --------------------------------------------------------------------------
TARGET = "DEP_DEL15"  # 1 = departure delayed by more than 15 minutes

# The documentation calls this column DEP_BLOCK; the raw BTS file calls it
# DEP_TIME_BLK. Accept either and normalise to DEP_BLOCK.
COLUMN_ALIASES = {"DEP_TIME_BLK": "DEP_BLOCK"}

# Value of PREVIOUS_AIRPORT for an aircraft's first flight of the day (missing values are mapped to it too).
FIRST_LEG = "NONE"

CATEGORICAL = ["DEP_BLOCK", "CARRIER_NAME", "DEPARTING_AIRPORT", "PREVIOUS_AIRPORT"]

NUMERIC = [
    "MONTH",
    "DAY_OF_WEEK",
    "DISTANCE_GROUP",
    "SEGMENT_NUMBER",
    "CONCURRENT_FLIGHTS",
    "NUMBER_OF_SEATS",
    "AIRPORT_FLIGHTS_MONTH",
    "AIRLINE_FLIGHTS_MONTH",
    "AIRLINE_AIRPORT_FLIGHTS_MONTH",
    "AVG_MONTHLY_PASS_AIRPORT",
    "AVG_MONTHLY_PASS_AIRLINE",
    "FLT_ATTENDANTS_PER_PASS",
    "GROUND_SERV_PER_PASS",
    "PLANE_AGE",
    "LATITUDE",
    "LONGITUDE",
    "PRCP",
    "SNOW",
    "SNWD",
    "TMAX",
    "AWND",
]

FEATURES = NUMERIC + CATEGORICAL
WEATHER = ["PRCP", "SNOW", "SNWD", "TMAX", "AWND"]

# Friendly names used in the "what is moving the odds" chart.
FEATURE_LABELS = {
    "MONTH": "Month",
    "DAY_OF_WEEK": "Day of week",
    "DISTANCE_GROUP": "Trip length",
    "SEGMENT_NUMBER": "Leg of the day",
    "CONCURRENT_FLIGHTS": "Traffic in the time block",
    "NUMBER_OF_SEATS": "Aircraft size",
    "AIRPORT_FLIGHTS_MONTH": "Airport size (flights)",
    "AIRLINE_FLIGHTS_MONTH": "Airline size (flights)",
    "AIRLINE_AIRPORT_FLIGHTS_MONTH": "Airline presence at airport",
    "AVG_MONTHLY_PASS_AIRPORT": "Airport size (passengers)",
    "AVG_MONTHLY_PASS_AIRLINE": "Airline size (passengers)",
    "FLT_ATTENDANTS_PER_PASS": "Cabin crew per passenger",
    "GROUND_SERV_PER_PASS": "Ground staff per passenger",
    "PLANE_AGE": "Aircraft age",
    "LATITUDE": "Airport latitude",
    "LONGITUDE": "Airport longitude",
    "PRCP": "Rain",
    "SNOW": "Snowfall",
    "SNWD": "Snow on the ground",
    "TMAX": "Temperature",
    "AWND": "Wind",
    "DEP_BLOCK": "Departure time block",
    "CARRIER_NAME": "Airline",
    "DEPARTING_AIRPORT": "Departure airport",
    "PREVIOUS_AIRPORT": "Where the aircraft came from",
}

# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------
MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# BTS DISTANCE_GROUP: 250-mile bands.
DISTANCE_GROUPS = {
    1: "Under 250 mi",
    2: "250 to 499 mi",
    3: "500 to 749 mi",
    4: "750 to 999 mi",
    5: "1,000 to 1,249 mi",
    6: "1,250 to 1,499 mi",
    7: "1,500 to 1,749 mi",
    8: "1,750 to 1,999 mi",
    9: "2,000 to 2,249 mi",
    10: "2,250 to 2,499 mi",
    11: "2,500 mi and up",
}

# --------------------------------------------------------------------------
# Delay "flight categories". Pilots already read weather as VFR / MVFR / IFR /
# LIFR, so delay risk borrows the same four colours. Thresholds are multiples
# of the dataset's average delay rate (see modeling.category_bounds).
# --------------------------------------------------------------------------
INK = "#12233F"
PAPER = "#E8EEF2"
AMBER = "#C9861A"

CATEGORIES = [
    {
        "code": "VFR",
        "label": "Low risk",
        "color": "#2E9B4E",
        "message": "Green across the board. Flights in a setup like this usually push back on time.",
    },
    {
        "code": "MVFR",
        "label": "Slightly elevated",
        "color": "#2B6CB0",
        "message": "A little worse than the average departure. Leave some slack if you have a connection.",
    },
    {
        "code": "IFR",
        "label": "High risk",
        "color": "#D1352B",
        "message": "Expect a hold. Delays are well above average for a setup like this.",
    },
    {
        "code": "LIFR",
        "label": "Very high risk",
        "color": "#7A1F8F",
        "message": "Low and slow. Conditions like these are among the worst for on-time departures.",
    },
]

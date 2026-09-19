"""Loading and cleaning the train / test CSVs.

The real files are large (about 1.3 GB uncompressed), so they are read in
chunks, only the needed columns are kept, and an optional random fraction of
each chunk is sampled. That keeps memory use modest without biasing the sample
towards any part of the file.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config

_WANTED = set(config.FEATURES) | {config.TARGET} | set(config.COLUMN_ALIASES)


def _wanted(column: str) -> bool:
    return column.strip() in _WANTED


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=lambda c: c.strip())
    return df.rename(columns=config.COLUMN_ALIASES)


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col in config.CATEGORICAL:
            continue
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype("float32")
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    """Type the columns the way the model and lookups expect them."""
    missing = [c for c in config.FEATURES + [config.TARGET] if c not in df.columns]
    if missing:
        raise ValueError(
            "These columns from docs/train_sets_documentation.txt are missing "
            f"from the CSV: {missing}"
        )

    for col in config.NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    for col in config.CATEGORICAL:
        fill = config.FIRST_LEG if col == "PREVIOUS_AIRPORT" else "UNKNOWN"
        df[col] = df[col].astype(object).where(df[col].notna(), fill).astype(str).astype("category")

    df[config.TARGET] = pd.to_numeric(df[config.TARGET], errors="coerce")
    df = df.dropna(subset=[config.TARGET]).copy()
    df[config.TARGET] = df[config.TARGET].astype("int8")
    return df[config.FEATURES + [config.TARGET]].reset_index(drop=True)


def read_flights(
    path: str | Path,
    sample_frac: float = 1.0,
    seed: int = 42,
    chunksize: int = 500_000,
    verbose: bool = True,
) -> pd.DataFrame:
    """Read a train/test CSV, keep only the modelling columns, optionally sample."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Unzip the dataset and put train.csv / test.csv in the data/ folder."
        )

    parts, seen, kept = [], 0, 0
    reader = pd.read_csv(path, usecols=_wanted, chunksize=chunksize, low_memory=False)
    for i, chunk in enumerate(reader):
        chunk = _normalise_columns(chunk)
        seen += len(chunk)
        if sample_frac < 1.0:
            chunk = chunk.sample(frac=sample_frac, random_state=seed + i)
        kept += len(chunk)
        parts.append(_downcast(chunk))
        if verbose:
            print(f"  {path.name}: read {seen:,} rows, kept {kept:,}", end="\r")
    if verbose:
        print()

    df = pd.concat(parts, ignore_index=True)
    return _finish(df)

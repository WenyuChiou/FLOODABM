"""Shared flood-prone classification for manuscript outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FLOOD_PRONE_MIN_YEARS = 7


def flood_prone_flags(
    flood_years_file: Path,
    tracts: Iterable[str],
    *,
    min_flood_years: int = FLOOD_PRONE_MIN_YEARS,
    expected_years: Iterable[int] = range(2011, 2024),
) -> pd.Series:
    """Classify tracts flooded in at least ``min_flood_years`` simulated years."""
    path = Path(flood_years_file)
    if not path.is_file():
        raise FileNotFoundError(f"Flood-year classification input not found: {path}")
    frame = pd.read_csv(path, dtype={"tract_geoid": "string"})
    required = {"year", "tract_geoid", "depth_m"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Flood-year input missing columns: {missing}")
    if frame.duplicated(["year", "tract_geoid"]).any():
        raise ValueError("Flood-year input contains duplicate tract-year rows")
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    frame["depth_m"] = pd.to_numeric(frame["depth_m"], errors="raise")
    if not np.isfinite(frame["depth_m"]).all() or (frame["depth_m"] < 0).any():
        raise ValueError("depth_m must contain finite nonnegative values")

    requested = pd.Index([str(tract) for tract in tracts], name="tract_geoid")
    available = set(frame["tract_geoid"].astype(str))
    missing_tracts = sorted(set(requested) - available)
    if missing_tracts:
        raise ValueError(f"Flood-year input is missing requested tracts: {missing_tracts}")
    expected_year_set = {int(year) for year in expected_years}
    requested_frame = frame[frame["tract_geoid"].isin(requested)]
    year_sets = requested_frame.groupby("tract_geoid", observed=True)["year"].agg(set)
    incomplete = sorted(
        tract for tract in requested if year_sets.get(tract, set()) != expected_year_set
    )
    if incomplete:
        raise ValueError(
            "Flood-year input must contain the complete expected year set for "
            f"every requested tract; incomplete tracts: {incomplete}"
        )

    flooded_years = (
        frame.loc[frame["depth_m"] > 0]
        .groupby("tract_geoid", observed=True)["year"]
        .nunique()
    )
    flags = requested.to_series().map(flooded_years).fillna(0).ge(min_flood_years)
    flags.index = requested
    return flags.astype("int8")

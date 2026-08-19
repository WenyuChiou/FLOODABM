"""Build tract-year PRB flood-depth distributions used by the ABM.

Raster cells are assigned to TIGER/Line census tracts by cell-center location.
Depths remain in meters and are stored as JSON arrays for deterministic,
household-level sampling during each simulated flood year.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


YEARS = tuple(range(2011, 2024))
ROOT = Path(__file__).resolve().parents[2]


def read_ascii(path: Path) -> tuple[dict[str, float], np.ndarray]:
    """Read one ESRI ASCII grid without changing its meter-valued depths."""
    with path.open("r", encoding="utf-8") as handle:
        header: dict[str, float] = {}
        for _ in range(6):
            key, value = handle.readline().split()
            header[key.lower()] = float(value)
    values = np.loadtxt(path, skiprows=6)
    return header, values


def tract_ids(path: Path) -> set[str]:
    """Read the model's 27 census-tract GEOIDs from its depth configuration."""
    config = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(config, list):
        ids = {str(row.get("CensusTract")) for row in config if row.get("CensusTract")}
    elif isinstance(config, dict) and "tract_geoid" in config:
        ids = {str(value) for value in config["tract_geoid"]}
    else:
        ids = {str(value) for value in config if str(value).isdigit()}
    if len(ids) != 27:
        raise ValueError(f"Expected 27 model tracts, found {len(ids)}")
    return ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raster-root", type=Path, required=True)
    parser.add_argument("--tract-boundary", type=Path, required=True)
    parser.add_argument(
        "--tract-config",
        type=Path,
        default=ROOT / "config" / "overall_md_mean_by_tract_2011_2023.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "config" / "depth_distribution_by_tract_year.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tracts = tract_ids(args.tract_config)
    boundary = gpd.read_file(args.tract_boundary).to_crs("EPSG:4326")
    geoid_col = next(
        (column for column in ("GEOID", "GEOID20", "GEOID10") if column in boundary.columns),
        None,
    )
    if geoid_col is None:
        raise ValueError("Tract boundary has no GEOID column")
    boundary["tract_geoid"] = boundary[geoid_col].astype(str).str.zfill(11)
    boundary = boundary[boundary["tract_geoid"].isin(tracts)].copy()
    if set(boundary["tract_geoid"]) != tracts:
        raise ValueError("Tract boundary does not contain all 27 model tracts")

    rows: list[dict[str, object]] = []
    for year in YEARS:
        filename = "maxDepth2011_newXsecDS.asc" if year == 2011 else f"maxDepth{year}.asc"
        raster = args.raster_root / filename
        header, values = read_ascii(raster)
        nodata = header["nodata_value"]
        rr, cc = np.where(np.isfinite(values) & (values != nodata))
        x = header["xllcorner"] + (cc + 0.5) * header["cellsize"]
        y = header["yllcorner"] + (values.shape[0] - rr - 0.5) * header["cellsize"]
        points = gpd.GeoDataFrame(
            {"depth_m": values[rr, cc]},
            geometry=gpd.points_from_xy(x, y),
            crs="EPSG:4326",
        )
        joined = gpd.sjoin(
            points,
            boundary[["tract_geoid", "geometry"]],
            predicate="within",
            how="inner",
        )
        for tract, frame in joined.groupby("tract_geoid", sort=True):
            depths = pd.to_numeric(frame["depth_m"], errors="raise").to_numpy(dtype=float)
            depths = depths[np.isfinite(depths) & (depths >= 0)]
            if len(depths) == 0:
                raise ValueError(f"No valid PRB cells for {tract}-{year}")
            rows.append(
                {
                    "year": year,
                    "tract_geoid": str(tract),
                    "n_cells": len(depths),
                    "mean_depth_m": float(depths.mean()),
                    "depth_values_m": json.dumps(depths.tolist(), separators=(",", ":")),
                }
            )

    result = pd.DataFrame(rows).sort_values(["year", "tract_geoid"])
    expected = len(tracts) * len(YEARS)
    if len(result) != expected or result[["year", "tract_geoid"]].duplicated().any():
        raise ValueError(f"Expected {expected} unique tract-years, found {len(result)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} tract-years to {args.output}")


if __name__ == "__main__":
    main()

"""Rebuild the map tables for revised WRR Figure 10 and Supporting Figure S7."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .verify_wrr_finance import N_RUNS, ROOT, YEARS
else:
    from verify_wrr_finance import N_RUNS, ROOT, YEARS


BASELINE = 2020
MAP_YEARS = tuple(range(BASELINE, 2024))
POST_YEARS = MAP_YEARS[1:]
GROUPS = {"owner": "homeowner", "renter": "renter"}
KEYS = ["draw", "group", "tract_geoid"]
F10 = "Figure10_spatial_fi.csv"
S7 = "FigureS7_population_change.csv"
REFERENCE = {
    # Final Figure 10 source table in the corrected WRR bundle (BOM and CRLF).
    F10: "a5fd14743512deaa4fea828413376007a2c8948d8ca75ebbc1201b6949471be3",
    # Only the six final S7 map fields, sorted by GEOID as an LF CSV.
    S7: "1f172276a493d186764c5acc3e390b249d3262f834014b7f14c96e0ec337ba89",
}


def flood_counts(path: Path) -> pd.Series:
    depths = pd.read_csv(path, usecols=["tract_geoid", "year", "mean_depth_m"],
                         dtype={"tract_geoid": str})
    annual = depths.groupby("tract_geoid")["year"].agg(list)
    if (len(annual) != 27 or depths["tract_geoid"].isna().any()
            or not all(sorted(years) == list(YEARS) for years in annual)
            or not np.isfinite(depths["mean_depth_m"]).all()
            or depths["mean_depth_m"].lt(0).any()):
        raise ValueError("Expected the full, nonnegative 27-tract flood-depth grid")
    return depths["mean_depth_m"].gt(0).groupby(depths["tract_geoid"]).sum()


def read_action_counts(path: Path, year: int, tracts: pd.Index) -> pd.DataFrame:
    counters = [f"{group}_{field}" for group in GROUPS for field in ("n_FI", "N_total")]
    frame = pd.read_csv(path, usecols=["year", "tract_geoid", *counters],
                        dtype={"tract_geoid": str, **{col: "int64" for col in counters}})
    observed = pd.Index(frame["tract_geoid"])
    if (observed.hasnans or not observed.is_unique or set(observed) != set(tracts)
            or not frame["year"].eq(year).all() or (frame[counters] < 0).any().any()
            or any(frame[f"{g}_n_FI"].gt(frame[f"{g}_N_total"]).any() for g in GROUPS)):
        raise ValueError(f"Invalid tract membership, year, or household counts in {path}")
    return frame.set_index("tract_geoid").sort_index()


def load_ensemble(mc_root: Path, tracts: pd.Index) -> pd.DataFrame:
    frames = []
    for draw in range(1, N_RUNS + 1):
        decisions = mc_root / f"run_{draw:02d}" / "baseline" / "baseline" / "decisions"
        for year in MAP_YEARS:
            path = decisions / f"action_share_owner_renter_tract_{year}.csv"
            counts = read_action_counts(path, year, tracts)
            for prefix, group in GROUPS.items():
                frame = counts[[f"{prefix}_n_FI", f"{prefix}_N_total"]].copy()
                frame.columns = ["insured", "population"]
                frames.append(frame.assign(draw=draw, year=year, group=group).reset_index())
    return pd.concat(frames, ignore_index=True)


def bivariate_code(baseline: pd.Series, change: pd.Series) -> np.ndarray:
    level = np.select([baseline.lt(.30), baseline.lt(.70)], [1, 2], default=3)
    direction = np.select([change.lt(-.03), change.le(.03)], [1, 2], default=3)
    return (level + 3 * (direction - 1)).astype(float)


def summarize(counts: pd.DataFrame, depth: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    population = counts.pivot(index=KEYS, columns="year", values="population")
    insured = counts.pivot(index=KEYS, columns="year", values="insured")
    if population.isna().any().any() or insured.isna().any().any():
        raise ValueError("Incomplete map data: every draw must have all four years")
    base_population = population[BASELINE]
    if base_population.le(0).any():
        raise ValueError("The revised benchmark requires positive 2020 tract/tenure populations")
    rates = insured.div(base_population, axis="index").rename(columns=lambda y: f"fi_{y}")
    for year in POST_YEARS:
        rates[f"delta_{year}"] = rates[f"fi_{year}"] - rates[f"fi_{BASELINE}"]
    spatial_fi = rates.groupby(level=["group", "tract_geoid"]).median()
    for year in POST_YEARS:
        spatial_fi[f"bivariate_{year}"] = bivariate_code(
            spatial_fi[f"fi_{BASELINE}"], spatial_fi[f"delta_{year}"],
        )
    spatial_fi["flood_year_count"] = spatial_fi.index.get_level_values("tract_geoid").map(depth)
    spatial_fi["flood_prone"] = spatial_fi["flood_year_count"].ge(7)
    spatial_fi.columns.name = None

    median_count = population.groupby(level=["group", "tract_geoid"]).median()
    pop_changes = {}
    for prefix, group in GROUPS.items():
        cohort = median_count.loc[group]
        for year in POST_YEARS:
            pop_changes[f"{prefix}_dpct_{year}"] = (
                (cohort[year] - cohort[BASELINE]) / cohort[BASELINE] * 100.0
            )
    return spatial_fi.reset_index(), pd.DataFrame(pop_changes).reset_index()


def render_csvs(fi: pd.DataFrame, population: pd.DataFrame) -> dict[str, bytes]:
    return {
        F10: fi.to_csv(index=False, lineterminator="\r\n").encode("utf-8-sig"),
        S7: population.to_csv(index=False, lineterminator="\n", float_format="%.9f").encode("ascii"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mc-root", type=Path, required=True, help="Root of the 50 adaptation draws")
    parser.add_argument("--out-dir", type=Path, help="New directory for the verified CSVs")
    args = parser.parse_args()
    if args.out_dir is not None and (args.out_dir.exists() or args.out_dir.is_symlink()):
        parser.error("--out-dir must not exist")
    try:
        depth = flood_counts(ROOT / "config" / "depth_distribution_by_tract_year.csv")
        fi, pop = summarize(load_ensemble(args.mc_root, depth.index), depth)
        for name, payload in (tables := render_csvs(fi, pop)).items():
            actual = hashlib.sha256(payload).hexdigest()
            if actual != REFERENCE[name]:
                raise ValueError(f"{name}: differs from the pinned revised reference ({actual})")
            print(f"PASS {name}: SHA-256 {actual}")
        if args.out_dir is not None:
            args.out_dir.mkdir(parents=True)
            for name, payload in tables.items():
                with (args.out_dir / name).open("xb") as target:
                    target.write(payload)
    except (OSError, ValueError, UnicodeError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()

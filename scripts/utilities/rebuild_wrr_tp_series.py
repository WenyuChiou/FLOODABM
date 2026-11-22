"""Rebuild the revised Figure 9 series from the 50 adaptation runs."""

import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

if __package__:
    from .verify_wrr_finance import N_RUNS, ROOT, YEARS
else:
    from verify_wrr_finance import N_RUNS, ROOT, YEARS


FILENAME = "Figure9_tp_by_prone.csv"
STRATA_SOURCE = ROOT / "config" / "depth_distribution_by_tract_year.csv"
REFERENCE = ROOT / "data" / "figures" / FILENAME
KEYS = ["year", "group", "exposure"]
STATS = ["median", "q25", "q75"]
TENURE = {"owner": "homeowner", "renter": "renter"}
EXPOSURE = {True: "flood-prone", False: "non-prone"}
TP_COLUMNS = ["TP_owner", "TP_renter"]
PUBLISHED_MISSING = (2023, "renter", "flood-prone", 37)


def read_strata(path: Path) -> pd.Series:
    depth = pd.read_csv(
        path, usecols=["tract_geoid", "year", "mean_depth_m"], dtype={"tract_geoid": str}
    )
    years = depth.groupby("tract_geoid")["year"].agg(list)
    depths = depth["mean_depth_m"]
    if (
        len(years) != 27
        or depth["tract_geoid"].isna().any()
        or not all(sorted(values) == list(YEARS) for values in years)
        or not np.isfinite(depths).all()
        or depths.lt(0).any()
    ):
        raise ValueError(f"Expected a complete mean-depth grid for 27 tracts: {path}")
    prone = depths.gt(0).groupby(depth["tract_geoid"]).sum().ge(7)
    if prone.sum() != 13:
        raise ValueError("Figure 9 uses 13 flood-prone and 14 non-prone tracts")
    return prone


def read_post_action_tp(path: Path, tracts: pd.Index) -> pd.DataFrame:
    tp = pd.read_csv(
        path, usecols=["year", "tract_geoid", "phase", *TP_COLUMNS],
        dtype={"tract_geoid": str, "phase": str, **dict.fromkeys(TP_COLUMNS, float)},
    )
    expected = {(year, tract) for year in YEARS for tract in tracts}
    actual = set(tp[["year", "tract_geoid"]].itertuples(index=False, name=None))
    if (
        actual != expected or len(tp) != len(expected)
        or not tp["phase"].eq("after").all()
        or np.isinf(tp[TP_COLUMNS].to_numpy()).any()
    ):
        raise ValueError(f"Expected one post-action TP row per tract and year: {path}")
    return tp.set_index(["year", "tract_geoid"])[TP_COLUMNS]


def read_counts(path: Path, year: int, tracts: pd.Index) -> pd.Series:
    decisions = pd.read_csv(
        path, usecols=["i", "year", "group", "tract_geoid"],
        dtype={"i": str, "group": str, "tract_geoid": str},
    )
    if (
        set(decisions["group"]) != set(TENURE)
        or not decisions["tract_geoid"].isin(tracts).all()
        or not decisions["year"].eq(year).all()
        or decisions["i"].isna().any()
        or not decisions["i"].is_unique
    ):
        raise ValueError(f"Invalid tenure, tract, year or unique household ID: {path}")
    return decisions.groupby(["group", "tract_geoid"]).size()


def weighted_tp(
    counts: pd.Series, after_tp: pd.DataFrame, strata: pd.Series
) -> list[tuple[str, str, float]]:
    rows = []
    for raw_group, group in TENURE.items():
        weights = counts.loc[raw_group]
        prone = strata.loc[weights.index]
        for flooded, exposure in EXPOSURE.items():
            selected = weights.loc[prone.eq(flooded)]
            if selected.empty:
                raise ValueError(f"No decision population for {group}/{exposure}")
            values = after_tp.loc[selected.index, f"TP_{raw_group}"]
            # Missing TP in a tract without decision households contributes no weight.
            # Missing TP in a tract with positive weight invalidates this draw-stratum.
            rows.append((group, exposure, float(np.average(values, weights=selected))))
    return rows


def summarize_draws(rows: list[tuple]) -> pd.DataFrame:
    data = pd.DataFrame(rows, columns=[*KEYS, "draw", "value"])
    expected = pd.MultiIndex.from_product(
        [YEARS, TENURE.values(), EXPOSURE.values(), range(1, N_RUNS + 1)],
        names=[*KEYS, "draw"],
    )
    actual = pd.MultiIndex.from_frame(data[[*KEYS, "draw"]])
    if not actual.is_unique or not expected.sort_values().equals(actual.sort_values()):
        raise ValueError("Incomplete or repeated Figure 9 draw/year/group/exposure cells")
    missing = data.loc[data["value"].isna(), [*KEYS, "draw"]]
    if list(missing.itertuples(index=False, name=None)) != [PUBLISHED_MISSING]:
        raise ValueError("Missing TP draws differ from the revised Figure 9 benchmark")
    return data.groupby(KEYS)["value"].agg(
        median="median", q25=lambda values: values.quantile(.25),
        q75=lambda values: values.quantile(.75),
    ).reset_index()


def rebuild_series(mc_root: Path, strata: pd.Series) -> pd.DataFrame:
    rows = []
    for draw in range(1, N_RUNS + 1):
        base = mc_root / f"run_{draw:02d}" / "baseline" / "baseline"
        tp = read_post_action_tp(base / "tp_traj.csv", strata.index)
        for year in YEARS:
            path = base / "decisions" / f"decisions_mgmix_{year}.csv"
            weights = read_counts(path, year, strata.index)
            rows.extend(
                (year, group, exposure, draw, value)
                for group, exposure, value in weighted_tp(weights, tp.loc[year], strata)
            )
        if draw % 10 == 0:
            print(f"Read {draw}/{N_RUNS} adaptation draws", flush=True)
    return summarize_draws(rows)


def verify_series(actual: pd.DataFrame, expected: pd.DataFrame) -> float:
    a = actual.sort_values(KEYS).reset_index(drop=True)
    b = expected.sort_values(KEYS).reset_index(drop=True)
    if a.duplicated(KEYS).any() or not a[KEYS].equals(b[KEYS]):
        raise ValueError("Figure 9 keys differ from the revised series")
    delta = np.abs(a[STATS].to_numpy(float) - b[STATS].to_numpy(float))
    if not delta.size or not np.isfinite(delta).all() or delta.max() > 1e-12:
        raise ValueError("A rebuilt Figure 9 statistic differs from the revised series")
    return float(delta.max())


def publish(out_dir: Path, payload: bytes, *, filename: str = FILENAME) -> Path:
    if Path(filename).name != filename or not filename.endswith(".csv"):
        raise ValueError("Expected a single CSV filename")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    output = out_dir / filename
    with TemporaryDirectory(prefix=".wrr-series-", dir=out_dir.parent) as folder:
        staged = Path(folder) / filename
        if staged.write_bytes(payload) != len(payload):
            raise OSError(f"Incomplete write of the verified series: {filename}")
        out_dir.mkdir()
        try:
            # Publish the complete file on the same filesystem, without replacing a target.
            os.link(staged, output)
        except OSError:
            try:
                out_dir.rmdir()
            except OSError:
                pass  # Never remove files a different process created.
            raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mc-root", type=Path, required=True, help="Root of the 50 paired runs")
    parser.add_argument("--out-dir", type=Path, required=True, help="New directory for the verified CSV")
    args = parser.parse_args()
    if args.out_dir.exists() or args.out_dir.is_symlink():
        parser.error("--out-dir must not exist")
    try:
        actual = rebuild_series(args.mc_root, read_strata(STRATA_SOURCE))
        expected = pd.read_csv(REFERENCE, float_precision="round_trip")
        max_delta = verify_series(actual, expected)
        payload = actual.to_csv(index=False, lineterminator="\n").encode("ascii")
        output = publish(args.out_dir, payload)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"PASS {output}: {len(actual)} rows, maximum difference {max_delta:.3g}")


if __name__ == "__main__":
    main()

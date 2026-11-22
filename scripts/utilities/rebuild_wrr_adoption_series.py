"""Rebuild Figure 8a–c and retain the revised Figure 8d series."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .rebuild_wrr_tp_series import EXPOSURE, STRATA_SOURCE, TENURE, publish, read_strata
    from .verify_wrr_finance import N_RUNS, ROOT, YEARS
else:
    from rebuild_wrr_tp_series import EXPOSURE, STRATA_SOURCE, TENURE, publish, read_strata
    from verify_wrr_finance import N_RUNS, ROOT, YEARS


FILENAME = "Figure8_adaptation_trajectories.csv"
REFERENCE = ROOT / "data" / "figures" / FILENAME
KEYS = ["year", "group", "action", "exposure"]
STATS = ["median", "q25", "q75"]
COLUMNS = ["year", "exposure", *STATS, "group", "action", "denominator"]
DECISION_COLUMNS = ["i", "year", "group", "tract_geoid", "action", "ELEV_FT", "POLICY_NAME"]
REBUILT = (("homeowner", "FI"), ("renter", "FI"), ("homeowner", "EH"))
RETAINED = ("renter", "RL")


def read_decisions(path: Path, year: int, tracts: pd.Index) -> pd.DataFrame:
    decisions = pd.read_csv(
        path, usecols=DECISION_COLUMNS,
        dtype={"i": str, "group": str, "tract_geoid": str, "POLICY_NAME": "string"},
    )
    if (
        set(decisions["group"]) != set(TENURE)
        or not decisions["tract_geoid"].isin(tracts).all()
        or not decisions["year"].eq(year).all()
        or not decisions["action"].isin({"DN", "FI", "EH", "BP", "RL"}).all()
        or decisions["i"].isna().any()
        or not decisions["i"].is_unique
    ):
        raise ValueError(f"Invalid year, decision, or unique household ID: {path}")
    return decisions


def adoption_rates(decisions: pd.DataFrame, strata: pd.Series) -> list[tuple]:
    rows = []
    for raw_group, group in TENURE.items():
        current = decisions.loc[decisions["group"].eq(raw_group)]
        prone = current["tract_geoid"].map(strata)
        insured = current["action"].eq("FI") | current["POLICY_NAME"].fillna("").str.strip().ne("")
        if raw_group == "owner":
            height = pd.to_numeric(current["ELEV_FT"], errors="raise").fillna(0)
            if not np.isfinite(height).all() or height.lt(0).any():
                raise ValueError("Invalid home elevation in a decision file")
            elevated = current["action"].eq("EH") | height.gt(0)
        for flooded, exposure in EXPOSURE.items():
            here = prone.eq(flooded)
            if not here.any():
                raise ValueError(f"No decision population for {group}/{exposure}")
            rows.append((group, "FI", exposure, float(insured.loc[here].mean())))
            if raw_group == "owner":
                rows.append((group, "EH", exposure, float(elevated.loc[here].mean())))
    return rows


def expected_keys(panels: tuple[tuple[str, str], ...]) -> set[tuple]:
    return {(year, group, action, exposure) for year in YEARS
            for group, action in panels for exposure in EXPOSURE.values()}


def summarize_draws(rows: list[tuple]) -> pd.DataFrame:
    data = pd.DataFrame(rows, columns=[*KEYS, "draw", "value"])
    cells = [*KEYS, "draw"]
    expected = {(*key, draw) for key in expected_keys(REBUILT) for draw in range(1, N_RUNS + 1)}
    if len(data) != len(expected) or set(data[cells].itertuples(index=False, name=None)) != expected:
        raise ValueError("Incomplete or repeated Figure 8a–c draw/year/group/action/exposure cells")
    if not data["value"].between(0, 1).all():
        raise ValueError("An adoption rate is missing or outside [0, 1]")
    return data.groupby(KEYS)["value"].agg(
        median="median", q25=lambda values: values.quantile(.25),
        q75=lambda values: values.quantile(.75),
    ).reset_index().assign(denominator="households")


def rebuild_series(mc_root: Path, strata: pd.Series) -> pd.DataFrame:
    rows = []
    for draw in range(1, N_RUNS + 1):
        decisions_dir = mc_root / f"run_{draw:02d}" / "baseline" / "baseline" / "decisions"
        for year in YEARS:
            file = decisions_dir / f"decisions_mgmix_{year}.csv"
            rows.extend((year, group, action, exposure, draw, rate)
                        for group, action, exposure, rate in
                        adoption_rates(read_decisions(file, year, strata.index), strata))
        if draw % 10 == 0:
            print(f"Read {draw}/{N_RUNS} adaptation draws", flush=True)
    return summarize_draws(rows)


def assemble_verified(
    rebuilt: pd.DataFrame, reference: pd.DataFrame
) -> tuple[pd.DataFrame, float]:
    expected = expected_keys((*REBUILT, RETAINED))
    if len(reference) != len(expected) or set(reference[KEYS].itertuples(index=False, name=None)) != expected:
        raise ValueError("Expected all 104 published Figure 8 rows exactly once")
    limits = reference[STATS]
    if not (
        limits.ge(0).all().all() and limits.le(1).all().all()
        and limits["q25"].le(limits["median"]).all() and limits["median"].le(limits["q75"]).all()
    ):
        raise ValueError("Invalid published Figure 8 statistics")
    keep = reference["group"].eq(RETAINED[0]) & reference["action"].eq(RETAINED[1])
    carried = reference.loc[keep, COLUMNS]
    baseline = reference.loc[~keep].sort_values(KEYS).reset_index(drop=True)
    actual = rebuilt.sort_values(KEYS).reset_index(drop=True)
    keys = [*KEYS, "denominator"]
    if not actual[keys].equals(baseline[keys]):
        raise ValueError("Figure 8a–c keys differ from the revised series")
    delta = np.abs(actual[STATS].to_numpy(float) - baseline[STATS].to_numpy(float))
    if not np.isfinite(delta).all() or delta.max() > 1e-12:
        raise ValueError("A rebuilt Figure 8a–c statistic differs from the revised series")
    assembled = pd.concat([actual[COLUMNS], carried], ignore_index=True).sort_values(KEYS)
    return assembled, float(delta.max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mc-root", type=Path, required=True, help="Root of the 50 paired runs")
    parser.add_argument("--out-dir", type=Path, required=True, help="New directory for the verified CSV")
    args = parser.parse_args()
    if args.out_dir.exists() or args.out_dir.is_symlink():
        parser.error("--out-dir must not exist")
    try:
        reference = pd.read_csv(REFERENCE, float_precision="round_trip")
        rebuilt = rebuild_series(args.mc_root, read_strata(STRATA_SOURCE))
        result, difference = assemble_verified(rebuilt, reference)
        payload = result.to_csv(index=False, lineterminator="\n").encode("ascii")
        output = publish(args.out_dir, payload, filename=FILENAME)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"PASS {output}: {len(rebuilt)} rows rebuilt (Figure 8a–c), maximum difference {difference:.3g}")
    print(f"Retained {len(result) - len(rebuilt)} revised Figure 8d rows from the bundled series")


if __name__ == "__main__":
    main()

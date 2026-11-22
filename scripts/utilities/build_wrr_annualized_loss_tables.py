"""Build the revised aggregate and annualized loss tables (S7 and response R3)."""

import argparse
import csv
import hashlib
from io import BytesIO, StringIO
from pathlib import Path

import pandas as pd

if __package__:
    from .rebuild_wrr_finance_series import (
        CUMULATIVE_NAME, GROUPS, TABLE_SPECS, compare_series, read_initial_counts,
    )
    from .verify_wrr_finance import ROOT, YEARS
else:
    from rebuild_wrr_finance_series import (
        CUMULATIVE_NAME, GROUPS, TABLE_SPECS, compare_series, read_initial_counts,
    )
    from verify_wrr_finance import ROOT, YEARS


REFERENCE = ROOT / "data" / "figures" / CUMULATIVE_NAME
REFERENCE_SHA256 = "da7ac3cd7b2ce9cac6747f7a48fb017e47e2335a9cb1bcf10bb54ec3320e4d04"
CASES = {"adaptation": "Adaptation", "no_adaptation": "No adaptation"}
TENURES = {"owner": "Homeowner", "renter": "Renter"}
METRICS = ("gul", "actual")
S7_HEADER = (
    "scenario", "household_group", "aggregate_gul_billion_usd", "aggregate_actual_loss_billion_usd",
    "annual_gul_kusd_per_household", "annual_actual_loss_kusd_per_household",
)
R3_HEADER = (
    "scenario", "group", "aggregate_gul", "aggregate_actual_loss",
    "annualized_gul_per_household_usd", "annualized_actual_loss_per_household_usd",
)


def read_series(rebuilt_dir: Path | None) -> pd.DataFrame:
    raw = REFERENCE.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(raw).hexdigest() != REFERENCE_SHA256:
        raise ValueError("The bundled cumulative series differs from the revised Figure 6 reference")
    reference = pd.read_csv(BytesIO(raw), encoding="utf-8-sig", float_precision="round_trip")
    if rebuilt_dir is None:
        return reference
    rebuilt = pd.read_csv(rebuilt_dir / CUMULATIVE_NAME, encoding="utf-8-sig", float_precision="round_trip")
    keys, row_count = TABLE_SPECS[CUMULATIVE_NAME]
    if not set(reference.columns).issubset(rebuilt.columns):
        raise ValueError("The rebuilt cumulative series lacks required Figure 6 columns")
    compare_series(rebuilt, reference, keys, row_count)
    return rebuilt


def render_tables(series: pd.DataFrame, initial: dict[str, int]) -> dict[str, str]:
    final = series.loc[series["year"].eq(YEARS.stop - 1)]
    medians = final.set_index(["scenario", "group", "metric"], verify_integrity=True)["median"]
    expected = {(scenario, tenure, metric)
                for scenario in CASES for tenure in GROUPS for metric in METRICS}
    if set(medians.index) != expected:
        raise ValueError("Expected both final cumulative metrics for each scenario and tenure")
    s7, r3 = StringIO(newline=""), StringIO(newline="")
    sm_writer, response_writer = (csv.writer(stream, lineterminator="\n") for stream in (s7, r3))
    sm_writer.writerow(S7_HEADER)
    response_writer.writerow(R3_HEADER)
    for scenario, label in CASES.items():
        for group in GROUPS:
            household_gul, household_actual = (medians[scenario, group, metric] for metric in METRICS)
            if not 0 <= household_actual <= household_gul < float("inf") or initial[group] <= 0:
                raise ValueError(f"Invalid cumulative loss or initial count for {scenario}, {group}")
            aggregate = [loss * initial[group] for loss in (household_gul, household_actual)]
            annual = [loss / len(YEARS) for loss in (household_gul, household_actual)]
            sm_writer.writerow((label, TENURES[group], *(f"{v / 1e9:.3f}" for v in aggregate),
                                *(f"{v / 1000:.2f}" for v in annual)))
            units, multiplier, places = ("billion", 1e9, 2) if group == "owner" else ("million", 1e6, 1)
            response_scenario = "Adaptation scenario" if scenario == "adaptation" else "No-adaptation scenario"
            response_writer.writerow((response_scenario, TENURES[group] + "s",
                                      *(f"${v / multiplier:.{places}f} {units}" for v in aggregate),
                                      *(f"{v:.0f}" for v in annual)))
    return {"Table_S7.csv": s7.getvalue(), "Table_R3.csv": r3.getvalue()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finance-series-dir", type=Path, help="Use the verified/rebuilt Figure 6 cumulative series")
    parser.add_argument("--out-dir", type=Path, required=True, help="New directory for the two tables")
    args = parser.parse_args()
    if args.out_dir.exists() or args.out_dir.is_symlink():
        parser.error("--out-dir must not exist")
    try:
        tables = render_tables(read_series(args.finance_series_dir), read_initial_counts())
        args.out_dir.mkdir(parents=True)
        for name, table in tables.items():
            with (args.out_dir / name).open("x", encoding="utf-8", newline="") as stream:
                stream.write(table)
        print("Built Tables S7 and R3 from the verified Figure 6 cumulative series")
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()

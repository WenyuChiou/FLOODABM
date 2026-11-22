"""Assemble the annual ensemble quantiles promised in WRR Supporting Information Text S7."""

import argparse
from collections import defaultdict
import csv
from decimal import Decimal
from io import StringIO
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "data" / "figures"
PUBLISHED = ROOT / "data" / "supplementary" / "Table_S_MC_variance.csv"
FIELDS = (
    "figure", "metric", "unit", "year", "median", "q25", "q75",
    "panels", "group", "scenario", "exposure", "county", "basis", "source",
)
YEARS = set(range(2011, 2024))
FINANCE_RATES = {"OOP rate", "Payout rate"}
FINANCE_DOLLARS = {
    "Premium per household", "OOP cost per household", "Premium plus OOP per household",
    "Payout per household",
}
ACTION_PANELS = {("homeowner", "FI"): "a", ("renter", "FI"): "b",
                 ("homeowner", "EH"): "c", ("renter", "RL"): "d"}


def build(sources: Path = SOURCES) -> str:
    table = []
    coverage = defaultdict(set)

    def add(source, record, figure, metric, unit, **metadata):
        year = int(record["year"])
        quantile_text = [record[key] for key in ("median", "q25", "q75")]
        median, q25, q75 = map(Decimal, quantile_text)
        if year not in YEARS or not all(v.is_finite() for v in (median, q25, q75)) or not q25 <= median <= q75:
            raise ValueError(f"Invalid ensemble quantiles: {source}, {record!r}")
        if unit == "fraction" and not 0 <= q25 <= q75 <= 1:
            raise ValueError(f"Expected a fraction in [0, 1]: {source}, {record!r}")
        identity = (source, metric, *(metadata.get(key, "") for key in FIELDS[7:-1]))
        if year in coverage[identity]:
            raise ValueError(f"Repeated year {year} for {identity}")
        coverage[identity].add(year)
        table.append(dict(zip(FIELDS, (figure, metric, unit, year, *quantile_text)),
                          source=source, **metadata))

    def read(source, expected_rows):
        with (sources / source).open(newline="", encoding="utf-8-sig") as file:
            rows = list(csv.DictReader(file))
        if len(rows) != expected_rows:
            raise ValueError(f"{source} has {len(rows)} rows; expected {expected_rows}")
        return rows

    source = "Figure5_nfip_validation_corrected_county_labels.csv"
    for row in read(source, 39):
        add(source, row, "Figure 5", "Simulated average payout per claim", "z-score",
            scenario="adaptation", county=row["county"],
            basis="Simulated standardized county payout per claim; excludes observations")

    source = "Figure6_cumulative_loss.csv"
    for row in read(source, 104):
        group, panel = {"owner": ("homeowner", "a"), "renter": ("renter", "b")}[row["group"]]
        metric = {"gul": "Cumulative ground-up loss", "actual": "Cumulative actual loss"}[row["metric"]]
        add(source, row, "Figure 6", metric, "USD/household", panels=panel,
            group=group, scenario=row["scenario"], basis="Cumulative totals divided by initial tenure count")

    source = "Figure6_annual_finance.csv"
    for row in read(source, 156):
        metric, group = row["metric"], row["group"]
        unit = {name: "USD/household" for name in FINANCE_DOLLARS} | {name: "fraction" for name in FINANCE_RATES}
        panel = {"homeowner": "e", "renter": "f"}[group] if metric.startswith("Payout") else (
            {"homeowner": "c", "renter": "d"}[group]
        )
        if metric == "Premium plus OOP per household":
            basis = ("Premium and OOP paired within each run; current-year tenure count. "
                     "Figure stacks separate medians but uses this combined IQR.")
        elif metric in FINANCE_RATES:
            basis = "Includes all households of that tenure; numerator divided by annual tenure GUL"
        else:
            basis = "Annual tenure totals divided by current-year tenure count"
        add(source, row, "Figure 6", metric, unit[metric], panels=panel,
            group=group, scenario="adaptation", basis=basis)

    source = "Figure8_adaptation_trajectories.csv"
    for row in read(source, 104):
        group, action = row["group"], row["action"]
        add(source, row, "Figure 8", f"{action} rate", "fraction", panels=ACTION_PANELS[(group, action)],
            group=group, scenario="adaptation", exposure=row["exposure"], basis=row["denominator"])

    source = "Figure9_tp_by_prone.csv"
    for row in read(source, 52):
        group = row["group"]
        add(source, row, "Figure 9", "Mean threat perception", "score",
            panels={"homeowner": "a", "renter": "b"}[group],
            group=group, scenario="adaptation", exposure=row["exposure"],
            basis="Annual mean among households in the indicated tenure and exposure group")

    if any(years != YEARS for years in coverage.values()) or len(table) != 455:
        raise ValueError("Incomplete revised annual-quantile table")
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(table)
    return output.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Compare with the bundled publication table")
    mode.add_argument("--output", type=Path, help="Write a new CSV (never overwrites)")
    args = parser.parse_args()
    data = build()
    if args.check:
        if PUBLISHED.read_text(encoding="utf-8") != data:
            parser.exit(1, "The published table does not match the revised figure aggregates\n")
        print("PASS: all 455 published annual-quantile rows match the revised aggregates")
    else:
        with args.output.open("x", encoding="utf-8", newline="") as csv_file:
            csv_file.write(data)


if __name__ == "__main__":
    main()

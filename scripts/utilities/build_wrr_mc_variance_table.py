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
QUANTILES = ("median", "q25", "q75")
BEHAVIOR_TABLES = {"Figure8_adaptation_trajectories.csv", "Figure9_tp_by_prone.csv"}
REBUILD_TOLERANCE = Decimal("1e-9")


def build(sources: Path = SOURCES, *, behavior_sources: Path | None = None) -> str:
    table = []
    coverage = defaultdict(set)

    def add(source, record, figure, metric, unit, **metadata):
        year = int(record["year"])
        quantile_text = [record[key] for key in QUANTILES]
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
        directory = behavior_sources if behavior_sources is not None and source in BEHAVIOR_TABLES else sources
        with (directory / source).open(newline="", encoding="utf-8-sig") as file:
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


def verify_rebuilt(data: str, expected: str) -> Decimal:
    keys = [field for field in FIELDS if field not in QUANTILES]

    def parse(text):
        rows = csv.DictReader(StringIO(text))
        if tuple(rows.fieldnames or ()) != FIELDS:
            raise ValueError("Unexpected annual-quantile columns")
        result = {}
        for row in rows:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Annual-quantile row has extra or missing cells")
            key = tuple(row[name] for name in keys)
            quantiles = tuple(Decimal(row[name]) for name in QUANTILES)
            if key in result or not all(value.is_finite() for value in quantiles):
                raise ValueError(f"Duplicate key or invalid quantile: {key}")
            result[key] = quantiles
        if len(result) != 455:
            raise ValueError("Expected 455 annual-quantile rows")
        return result

    actual = parse(data)
    reference = parse(expected)
    if actual.keys() != reference.keys():
        raise ValueError("A rebuilt annual-quantile row differs in year, metric, or metadata")
    delta = max(abs(a - b) for key in actual
                for a, b in zip(actual[key], reference[key], strict=True))
    if delta > REBUILD_TOLERANCE:
        raise ValueError(f"A rebuilt annual quantile differs by {delta} (limit {REBUILD_TOLERANCE})")
    return delta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Compare with the bundled publication table")
    mode.add_argument("--output", type=Path, help="Write a new CSV (never overwrites)")
    parser.add_argument("--finance-series-dir", type=Path, help="Read the Figure 5 and 6 rebuilt series")
    parser.add_argument("--behavior-series-dir", type=Path, help="Read the Figure 8 and 9 rebuilt series")
    args = parser.parse_args()
    supplied = args.finance_series_dir is not None
    if supplied != (args.behavior_series_dir is not None):
        parser.error("Supply both rebuilt series directories or neither")
    try:
        data = build(args.finance_series_dir or SOURCES, behavior_sources=args.behavior_series_dir)
        if args.check:
            if supplied:
                delta = verify_rebuilt(data, PUBLISHED.read_text(encoding="utf-8"))
                print(f"PASS: 455 rebuilt annual-quantile rows match; largest absolute difference {delta:.3g}")
            else:
                if PUBLISHED.read_bytes() != data.encode("utf-8"):
                    raise ValueError("The published table does not match the revised figure aggregates")
                print("PASS: all 455 published annual-quantile rows match the revised aggregates")
        else:
            with args.output.open("x", encoding="utf-8", newline="") as csv_file:
                csv_file.write(data)
    except (OSError, ValueError, ArithmeticError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()

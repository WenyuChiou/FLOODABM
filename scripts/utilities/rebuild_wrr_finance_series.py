"""Recompute the simulated finance series in revised WRR Figures 5–7 and 11."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .verify_wrr_finance import N_RUNS, ROOT, SCENARIOS, YEARS, verify
else:
    from verify_wrr_finance import N_RUNS, ROOT, SCENARIOS, YEARS, verify


GROUPS = ("owner", "renter")
INITIAL_COUNTS = {"owner": 41517, "renter": 10624}
N_TRACTS = 27
RAW_METRICS = {
    "gul": "gross_total_usd",
    "payout": "payout_total_usd",
    "premium": "premium_total_usd",
    "oop": "oop_total_usd",
    "households": "households",
}
CUMULATIVE_NAME = "Figure6_cumulative_loss.csv"
ANNUAL_NAME = "Figure6_annual_finance.csv"
NFIP_NAME = "Figure5_nfip_validation_corrected_county_labels.csv"
NFIP_METRICS_NAME = "Figure5_metrics_corrected_county_labels.csv"
COUNTIES_BY_FIPS = {"34013": "Essex", "34027": "Morris", "34031": "Passaic"}
AEP_NAME = "Figure7_aep.csv"
INCOME_AEP_NAME = "Figure11_income_aep.csv"
MEDIAN_INCOME_USD = {"homeowner": 148097, "renter": 63699}
AEP_KEYS = ["group", "metric", "aep"]
AEP_METRICS = ("GUL", "Actual loss")
AEP_PERIOD = range(2012, 2024)
TABLE_SPECS = {
    CUMULATIVE_NAME: (["scenario", "group", "metric", "year"], 104),
    ANNUAL_NAME: (["group", "year", "metric"], 156),
}
STATS = ["median", "q25", "q75"]
FINANCE_REFERENCE = ROOT / "data" / "supplementary" / "wrr_mc50_finance_reference.csv"


def read_initial_counts() -> dict[str, int]:
    path = ROOT / "config" / "households_for_abm_sfha.csv"
    counts = pd.read_csv(path, usecols=["group"])["group"].value_counts().to_dict()
    if counts != INITIAL_COUNTS:
        raise ValueError(f"Benchmark initial household counts changed: {counts}")
    return counts


def read_tract_finance(path: Path) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    """Summarize tenure-level losses and county-level payouts/claim counts."""
    columns = [f"{group}_{name}" for group in GROUPS for name in RAW_METRICS.values()]
    county_columns = ["payout_total_usd", "claims"]
    columns += county_columns
    data = pd.read_csv(path, usecols=["tract_geoid", *columns], dtype={"tract_geoid": str})
    if len(data) != N_TRACTS or data["tract_geoid"].isna().any() or data["tract_geoid"].duplicated().any():
        raise ValueError(f"Expected {N_TRACTS} distinct tracts: {path}")
    values = data[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"Missing, nonfinite or negative finance values: {path}")
    totals = {column: float(data[column].sum()) for column in columns}
    tenure = {
        group: {metric: totals[f"{group}_{column}"] for metric, column in RAW_METRICS.items()}
        for group in GROUPS
    }
    data["county"] = data["tract_geoid"].str[:5].map(COUNTIES_BY_FIPS)
    if data["county"].isna().any() or set(data["county"]) != set(COUNTIES_BY_FIPS.values()):
        raise ValueError(f"Unexpected counties in the three-county benchmark: {path}")
    county = data.groupby("county", as_index=False).agg(payout=("payout_total_usd", "sum"),
                                                         claims=("claims", "sum"))
    return tenure, county


def read_run_years(mc_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read each of the 1,300 raw CSVs once for all finance-derived figure metrics."""
    values, counties = [], []
    for draw in range(1, N_RUNS + 1):
        for scenario in SCENARIOS:
            finance = mc_root / f"run_{draw:02d}" / "baseline" / scenario / "finance"
            for year in YEARS:
                totals, county = read_tract_finance(finance / f"finance_tract_{year}.csv")
                for group, amounts in totals.items():
                    values.append({"run_id": draw, "scenario": scenario, "year": year,
                                   "group": group, **amounts})
                if scenario == "baseline":
                    counties.append(county.assign(run_id=draw, year=year))
    return pd.DataFrame(values), pd.concat(counties, ignore_index=True)


def validate_panel(data: pd.DataFrame) -> None:
    key = ["run_id", "scenario", "year", "group"]
    expected = {(n, scenario, year, group)
                for n in range(1, N_RUNS + 1) for scenario in SCENARIOS
                for year in YEARS for group in GROUPS}
    actual = set(data[key].itertuples(index=False, name=None))
    if actual != expected or len(data) != len(expected):
        raise ValueError("The finance ledger must contain each draw/scenario/year/tenure exactly once")
    if not np.isfinite(data[list(RAW_METRICS)].to_numpy(dtype=float)).all():
        raise ValueError("The finance ledger contains nonfinite values")
    if (data["households"] <= 0).any():
        raise ValueError("A run-year contains no households of the selected tenure")


def summarize(long: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    values = long.groupby(keys)["value"]
    if not (values.count() == N_RUNS).all():
        raise ValueError("An ensemble quantile is missing one or more draws")
    return values.agg(
        median="median", q25=lambda v: v.quantile(.25), q75=lambda v: v.quantile(.75),
    ).reset_index()


def figure6_series(ledger: pd.DataFrame, initial_counts: dict[str, int]) -> dict[str, pd.DataFrame]:
    """Apply the final distinct denominator for each panel before summarizing across draws."""
    validate_panel(ledger)
    if set(initial_counts) != set(GROUPS) or any(n <= 0 for n in initial_counts.values()):
        raise ValueError("Initial homeowner and renter counts must both be positive")
    base = ledger.loc[ledger["scenario"] == "baseline"].copy()
    denominator = base["households"]
    annual_columns = {
        "Premium per household": base["premium"] / denominator,
        "OOP cost per household": base["oop"] / denominator,
        "Payout per household": base["payout"] / denominator,
        "OOP rate": base["oop"].div(base["gul"].where(base["gul"] > 0)).fillna(0),
        "Payout rate": base["payout"].div(base["gul"].where(base["gul"] > 0)).fillna(0),
        "Premium plus OOP per household": (base["premium"] + base["oop"]) / denominator,
    }
    annual = base[["run_id", "group", "year"]].assign(**annual_columns)
    annual["group"] = annual["group"].replace({"owner": "homeowner"})
    annual_long = annual.melt(id_vars=["run_id", "group", "year"], var_name="metric")

    cum = ledger.sort_values(["run_id", "scenario", "group", "year"]).copy()
    cum["actual"] = cum["gul"] - cum["payout"]
    cum[["gul", "actual"]] = cum.groupby(["run_id", "scenario", "group"])[["gul", "actual"]].cumsum().div(
        cum["group"].map(initial_counts), axis=0,
    )
    cum["scenario"] = cum["scenario"].map({"baseline": "adaptation", "worst": "no_adaptation"})
    cum_long = cum.melt(id_vars=["scenario", "group", "year", "run_id"],
                        value_vars=["gul", "actual"], var_name="metric")

    return {
        CUMULATIVE_NAME: summarize(cum_long, TABLE_SPECS[CUMULATIVE_NAME][0]),
        ANNUAL_NAME: summarize(annual_long, TABLE_SPECS[ANNUAL_NAME][0]),
    }


def compare_series(actual: pd.DataFrame, expected: pd.DataFrame, keys: list[str], rows: int) -> float:
    """Check all coordinates and values; allow only floating-point summation differences."""
    current = actual.sort_values(keys).reset_index(drop=True)
    accepted = expected.sort_values(keys).reset_index(drop=True)
    if len(current) != rows or len(accepted) != rows or current[keys].duplicated().any():
        raise ValueError("Unexpected figure table row count or duplicate coordinate")
    if not current[keys].equals(accepted[keys]):
        raise ValueError("Rebuilt figure coordinates differ from the accepted table")
    a, b = current[STATS].to_numpy(), accepted[STATS].to_numpy()
    if not np.allclose(a, b, rtol=1e-12, atol=1e-10):
        failing = ~np.isclose(a, b, rtol=1e-12, atol=1e-10)
        index, stat = np.argwhere(failing)[0]
        raise ValueError(f"Rebuilt value differs: {current.loc[index, keys].to_dict()}, "
                         f"{STATS[stat]}={a[index, stat]} (accepted {b[index, stat]})")
    return float(np.max(np.abs(a - b)))


def aep_series(ledger: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Pool adaptation-era run-years, keeping the GUL and actual-loss rankings independent."""
    validate_panel(ledger)
    base = ledger.loc[ledger["scenario"].eq("baseline") & ledger["year"].isin(AEP_PERIOD)].copy()
    base["GUL"] = base["gul"] / base["households"]
    base["Actual loss"] = (base["gul"] - base["payout"]) / base["households"]
    base["group"] = base["group"].replace({"owner": "homeowner"})
    n = N_RUNS * len(AEP_PERIOD)
    probabilities = 1 - np.arange(1, n + 1) / n
    dollars, income = [], []
    for group, denominator in MEDIAN_INCOME_USD.items():
        for metric in AEP_METRICS:
            values = np.sort(base.loc[base["group"].eq(group), metric].to_numpy())
            if len(values) != n:
                raise ValueError(f"Expected {n} adaptation run-years for {group}")
            coordinates = {"group": group, "metric": metric, "aep": probabilities}
            dollars.append(pd.DataFrame(coordinates | {"value": values}))
            income.append(pd.DataFrame(coordinates | {"value": values / denominator * 100.0}))
    return {
        AEP_NAME: pd.concat(dollars, ignore_index=True),
        INCOME_AEP_NAME: pd.concat(income, ignore_index=True),
    }


def compare_aep(actual: pd.DataFrame, expected: pd.DataFrame, *, tolerance: float) -> float:
    """Check all 2,400 plotting positions and their numeric values."""
    n = len(MEDIAN_INCOME_USD) * len(AEP_METRICS) * N_RUNS * len(AEP_PERIOD)
    current = actual.sort_values(AEP_KEYS).reset_index(drop=True)
    accepted = expected.sort_values(AEP_KEYS).reset_index(drop=True)
    if len(current) != n or len(accepted) != n or current[AEP_KEYS].duplicated().any():
        raise ValueError(f"Expected {n} unique AEP coordinates")
    if not current[AEP_KEYS].equals(accepted[AEP_KEYS]):
        raise ValueError("Rebuilt exceedance coordinates differ from the accepted table")
    delta = (current["value"] - accepted["value"]).abs().to_numpy()
    if not np.isfinite(delta).all() or np.max(delta) > tolerance:
        raise ValueError("A rebuilt AEP value differs from the accepted table")
    return float(np.max(delta))


def nfip_series(counties: pd.DataFrame, observed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute simulated county payout bands and the selected-year correlations."""
    keys = ["county", "year"]
    names = set(COUNTIES_BY_FIPS.values())
    expected = {(c, y, n) for c in names for y in YEARS for n in range(1, N_RUNS + 1)}
    actual = set(counties[[*keys, "run_id"]].itertuples(index=False, name=None))
    if actual != expected or len(counties) != len(expected) or (counties["claims"] <= 0).any():
        raise ValueError("Expected one positive-claim record for each county, year and adaptation draw")
    observed = observed[[*keys, "observed_z", "correlation_year"]].copy()
    if set(observed[keys].itertuples(index=False, name=None)) != {(c, y) for c in names for y in YEARS}:
        raise ValueError("Observed payout series does not have the expected county/year coordinates")
    series = counties.assign(sim_avg=counties["payout"] / counties["claims"])

    def standardize(values):
        sd = values.std(ddof=1)
        if sd <= 0:
            raise ValueError("The archived simulated county payout series has no temporal variance")
        return (values - values.mean()) / sd

    series["value"] = series.groupby(["run_id", "county"])["sim_avg"].transform(standardize)
    bands = summarize(series, keys).merge(observed, on=keys, validate="one_to_one")
    metrics = []
    for county, obs in observed[observed["correlation_year"]].groupby("county", sort=False):
        obs = obs.sort_values("year")
        subset = series[series["county"].eq(county) & series["year"].isin(obs["year"])]
        values = subset.sort_values(["run_id", "year"])["sim_avg"].to_numpy().reshape(N_RUNS, len(obs))
        sd = values.std(axis=1, ddof=0, keepdims=True)
        z = np.divide(values - values.mean(axis=1, keepdims=True), sd,
                      out=np.zeros_like(values), where=sd > 0)
        medians = np.median(z, axis=0)
        targets = obs["observed_z"].to_numpy()
        valid = np.isfinite(medians) & np.isfinite(targets)
        if valid.sum() < 2:
            raise ValueError(f"Not enough selected correlation years for {county}")
        pearson_r = float(np.corrcoef(medians[valid], targets[valid])[0, 1])
        if not np.isfinite(pearson_r):
            raise ValueError(f"No finite correlation for {county}")
        metrics.append({"county": county, "n_years": int(valid.sum()), "pearson_r": pearson_r})
    if {x["county"] for x in metrics} != names:
        raise ValueError("Missing selected correlation years for a benchmark county")
    return bands, pd.DataFrame(metrics)


def compare_county_metrics(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    keys = ["county", "n_years"]
    a = actual.sort_values(keys).reset_index(drop=True)
    b = expected.sort_values(keys).reset_index(drop=True)
    if not a[keys].equals(b[keys]) or not np.allclose(a["pearson_r"], b["pearson_r"], rtol=0, atol=1e-12):
        raise ValueError("Rebuilt county correlation statistics differ from the accepted values")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mc-root", type=Path, required=True, help="Outputs of all 50 paired draws")
    parser.add_argument("--out-dir", type=Path, help="Optional new directory for the verified series")
    args = parser.parse_args()
    if args.out_dir is not None and (args.out_dir.exists() or args.out_dir.is_symlink()):
        parser.error("--out-dir must not exist")
    try:
        verified = verify(args.mc_root, FINANCE_REFERENCE, 1, N_RUNS)
        print(f"Verified {verified} raw finance scenario/draw bundles against the revised archive.")
        ledger, counties = read_run_years(args.mc_root)
        accepted_nfip = pd.read_csv(ROOT / "data" / "figures" / NFIP_NAME, float_precision="round_trip")
        bands, metrics = nfip_series(counties, accepted_nfip)
        delta = compare_series(bands, accepted_nfip, ["county", "year"], 39)
        compare_county_metrics(metrics, pd.read_csv(ROOT / "data" / "figures" / NFIP_METRICS_NAME))
        print(f"PASS {NFIP_NAME}: 39 rows, maximum absolute difference {delta:.3g}; three correlations match")
        finance = figure6_series(ledger, read_initial_counts())
        aep = aep_series(ledger)
        for name, table in finance.items():
            keys, count = TABLE_SPECS[name]
            expected = pd.read_csv(ROOT / "data" / "figures" / name)
            delta = compare_series(table, expected, keys, count)
            print(f"PASS {name}: {count} rows, maximum absolute difference {delta:.3g}")
        for name, table in aep.items():
            expected = pd.read_csv(ROOT / "data" / "figures" / name, float_precision="round_trip")
            delta = compare_aep(table, expected, tolerance=1e-8 if name == AEP_NAME else 1e-11)
            print(f"PASS {name}: {len(table)} rows, maximum absolute difference {delta:.3g}")
        if args.out_dir is not None:
            args.out_dir.mkdir(parents=True)
            for name, table in (finance | {NFIP_NAME: bands, NFIP_METRICS_NAME: metrics}).items():
                table.to_csv(args.out_dir / name, index=False, mode="x", encoding="utf-8", lineterminator="\n")
            for name, table in aep.items():
                table.to_csv(args.out_dir / name, index=False, mode="x", encoding="utf-8-sig", lineterminator="\r\n")
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()

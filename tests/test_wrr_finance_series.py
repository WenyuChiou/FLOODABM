import pandas as pd
import pytest

from scripts.utilities import rebuild_wrr_finance_series as series


@pytest.fixture
def ledger():
    rows = []
    for run in range(1, series.N_RUNS + 1):
        for scenario in series.SCENARIOS:
            for year in series.YEARS:
                for group in series.GROUPS:
                    n = 4 if year == 2011 else 2
                    gul = 10_000 if year == 2011 else 100
                    payout = 10
                    if scenario == "worst":
                        gul, payout = 20_000, 0
                    rows.append({"run_id": run, "scenario": scenario, "year": year, "group": group,
                                 "gul": gul, "payout": payout, "premium": n * run,
                                 "oop": n * (51 - run), "households": n})
    return pd.DataFrame(rows)


def test_figure6_uses_fixed_initial_counts_for_cumulative_and_pairs_annual_components(ledger):
    tables = series.figure6_series(ledger, {"owner": 4, "renter": 8})
    cumulative = tables[series.CUMULATIVE_NAME]
    expected = cumulative.query("scenario == 'adaptation' and group == 'owner' and year == 2012 and metric == 'gul'")
    assert expected["median"].iloc[0] == (10_000 + 100) / 4
    annual = tables[series.ANNUAL_NAME].query("group == 'homeowner' and year == 2012").set_index("metric")
    assert annual.loc["Payout per household", "median"] == 5.0
    assert annual.loc["Payout rate", "median"] == 0.1
    assert annual.loc["Premium plus OOP per household", series.STATS].tolist() == [51, 51, 51]
    assert annual.loc["Premium per household", "q25"] + annual.loc["OOP cost per household", "q25"] < 51


def test_aep_uses_only_adaptation_2012_to_2023_and_independent_ranks(ledger):
    select = ledger["scenario"].eq("baseline") & ledger["year"].eq(2021)
    ledger.loc[select & ledger["run_id"].eq(50), ["gul", "payout"]] = [900, 800]
    ledger.loc[select & ledger["run_id"].eq(49), ["gul", "payout"]] = [600, 0]
    tables = series.aep_series(ledger)
    aep = tables[series.AEP_NAME]
    assert len(aep) == 2400
    curve = aep.query("group == 'homeowner' and metric == 'GUL'")
    assert len(curve) == 600 and curve["aep"].iloc[0] == 1 - 1 / 600
    tail = aep.query("group == 'homeowner' and aep == 0").set_index("metric")
    assert tail.loc["GUL", "value"] == 450 and tail.loc["Actual loss", "value"] == 300
    income = tables[series.INCOME_AEP_NAME]
    percent = income.query("group == 'homeowner' and metric == 'GUL' and aep == 0")["value"].iloc[0]
    assert percent == 450 / 148097 * 100
    assert aep[series.AEP_KEYS].equals(income[series.AEP_KEYS])


def test_panel_guard_rejects_substituting_one_draw_with_a_duplicate(ledger):
    ledger.iloc[-1] = ledger.iloc[-2]
    with pytest.raises(ValueError, match="exactly once"):
        series.validate_panel(ledger)


def test_aep_verifier_detects_a_changed_value(ledger):
    expected = series.aep_series(ledger)[series.AEP_NAME]
    copied = expected.copy()
    copied.loc[0, "value"] += 1
    with pytest.raises(ValueError, match="AEP value differs"):
        series.compare_aep(copied, expected, tolerance=1e-8)


def test_nfip_standardizes_within_each_draw_and_uses_selected_years_for_correlation():
    counties, observed = [], []
    for name in series.COUNTIES_BY_FIPS.values():
        for year in series.YEARS:
            year_index = year - series.YEARS.start
            for draw in range(1, series.N_RUNS + 1):
                counties.append({"county": name, "run_id": draw, "year": year,
                                 "payout": 2 * (10 * draw + year_index), "claims": 2})
            observed.append({"county": name, "year": year, "observed_z": -year_index,
                             "correlation_year": year in (2013, 2017, 2020)})
    bands, metrics = series.nfip_series(pd.DataFrame(counties), pd.DataFrame(observed))
    assert len(bands) == 39 and metrics["n_years"].tolist() == [3, 3, 3]
    assert bands.query("year == 2011")["median"].iloc[0] == pytest.approx(-6 / (182 / 12) ** 0.5)
    assert metrics["pearson_r"].tolist() == pytest.approx([-1, -1, -1])

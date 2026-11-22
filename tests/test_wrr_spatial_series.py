"""Protect the distinct denominator and aggregation rules for the WRR maps."""

import pandas as pd
import pytest

from scripts.utilities import rebuild_wrr_spatial_series as maps


@pytest.fixture
def counts():
    rows = []
    for draw, (base_total, initial_fi, annual_total, annual_fi) in enumerate(
        [(10, 1, 30, 9), (20, 4, 20, 6), (100, 90, 110, 80)], start=1,
    ):
        rows.extend(
            (draw, "homeowner", "00000000001", year,
             initial_fi if year == 2020 else annual_fi,
             base_total if year == 2020 else annual_total)
            for year in maps.MAP_YEARS
        )
    return pd.DataFrame(rows, columns=[*maps.KEYS, "year", "insured", "population"])


def test_figure10_medians_use_paired_changes_and_frozen_population(counts):
    counts = pd.concat([counts, counts.assign(group="renter")], ignore_index=True)
    fi, _ = maps.summarize(counts, pd.Series({"00000000001": 7}))
    row = fi.iloc[0]
    assert row.fi_2020 == pytest.approx(.2)
    assert row.fi_2021 == pytest.approx(.8)
    assert row.delta_2021 == pytest.approx(.1)
    assert row.bivariate_2021 == 7.0
    assert row.flood_prone


def test_s7_percent_change_uses_population_medians_first(counts):
    counts = pd.concat([counts, counts.assign(group="renter")], ignore_index=True)
    _, s7 = maps.summarize(counts, pd.Series({"00000000001": 7}))
    assert s7.loc[0, "owner_dpct_2021"] == pytest.approx(50)
    assert s7.loc[0, "renter_dpct_2023"] == pytest.approx(50)


def test_spatial_rebuild_rejects_zero_reference_population(counts):
    counts.loc[0, ["population", "insured"]] = 0
    with pytest.raises(ValueError, match="positive 2020"):
        maps.summarize(counts, pd.Series({"00000000001": 7}))


def test_spatial_rebuild_rejects_nonrectangular_counts(counts):
    with pytest.raises(ValueError, match="every draw must have all four years"):
        maps.summarize(counts.iloc[1:], pd.Series({"00000000001": 7}))


def test_bivariate_threshold_equality():
    base = pd.Series([.29, .30, .69, .70, .70])
    change = pd.Series([-.04, -.03, .03, .031, -.031])
    assert maps.bivariate_code(base, change).tolist() == [1., 5., 5., 9., 3.]


def test_count_file_checks_year_and_duplicate_tracts(tmp_path):
    path = tmp_path / "tract_action.csv"
    tracts = pd.Index(["00000000001", "00000000002"])
    raw = pd.DataFrame({
        "year": [2020, 2020], "tract_geoid": tracts,
        "owner_n_FI": [1, 2], "owner_N_total": [10, 20],
        "renter_n_FI": [2, 3], "renter_N_total": [5, 5],
    })
    raw.to_csv(path, index=False)
    assert maps.read_action_counts(path, 2020, tracts).index.equals(tracts)
    with pytest.raises(ValueError, match="Invalid tract membership"):
        maps.read_action_counts(path, 2021, tracts)
    raw.loc[1, "tract_geoid"] = tracts[0]
    raw.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Invalid tract membership"):
        maps.read_action_counts(path, 2020, tracts)

"""Test the final manuscript's two displays of cumulative and annualized losses."""

import csv
from io import StringIO
import subprocess
import sys

import pandas as pd
import pytest

from scripts.utilities import build_wrr_annualized_loss_tables as tables


def test_all_final_table_values_match_the_two_submitted_precisions():
    result = tables.render_tables(tables.read_series(None), tables.read_initial_counts())

    def values(name):
        return [row[2:] for row in list(csv.reader(StringIO(result[name])))[1:]]

    assert values("Table_S7.csv") == [
        ["12.832", "6.685", "23.77", "12.39"], ["0.175", "0.119", "1.27", "0.86"],
        ["13.627", "12.354", "25.25", "22.89"], ["0.341", "0.325", "2.47", "2.36"],
    ]
    assert values("Table_R3.csv") == [
        ["$12.83 billion", "$6.69 billion", "23775", "12387"],
        ["$174.8 million", "$118.5 million", "1266", "858"],
        ["$13.63 billion", "$12.35 billion", "25248", "22889"],
        ["$340.6 million", "$325.3 million", "2466", "2355"],
    ]


def test_rebuilt_input_is_checked_at_all_coordinates(tmp_path):
    rebuilt = tables.read_series(None)
    rebuilt.sample(frac=1, random_state=2).to_csv(tmp_path / tables.CUMULATIVE_NAME, index=False)
    assert len(tables.read_series(tmp_path)) == 104
    row = rebuilt.loc[rebuilt["year"].eq(2012)].index[0]
    rebuilt.loc[row, "median"] += 1
    rebuilt.to_csv(tmp_path / tables.CUMULATIVE_NAME, index=False)
    with pytest.raises(ValueError, match="Rebuilt value differs"):
        tables.read_series(tmp_path)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "nonfinite", "inverted"])
def test_malformed_terminal_series_fails_closed(mutation):
    series = tables.read_series(None)
    chosen = series.loc[series["year"].eq(2023) & series["metric"].eq("actual")].index[0]
    if mutation == "missing":
        series = series.drop(index=chosen)
    elif mutation == "duplicate":
        series = pd.concat([series, series.loc[[chosen]]], ignore_index=True)
    else:
        series.loc[chosen, "median"] = float("nan") if mutation == "nonfinite" else 1e12
    with pytest.raises(ValueError):
        tables.render_tables(series, tables.read_initial_counts())


def test_cli_rejects_an_existing_destination_without_touching_it(tmp_path):
    output = tmp_path / "tables"
    command = [sys.executable, tables.__file__, "--out-dir", str(output)]
    subprocess.run(command, check=True, capture_output=True)
    saved = {file.name: file.read_bytes() for file in output.iterdir()}
    assert set(saved) == {"Table_S7.csv", "Table_R3.csv"}
    result = subprocess.run(command, check=False, capture_output=True)
    assert result.returncode != 0
    assert b"--out-dir must not exist" in result.stderr
    assert {file.name: file.read_bytes() for file in output.iterdir()} == saved

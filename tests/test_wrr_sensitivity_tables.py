"""Tests for the published paired threshold and decision-rule tables."""

from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from scripts.utilities import rebuild_wrr_sensitivity_tables as tables


def test_bundled_totals_reproduce_the_submitted_display():
    summary = tables.summarize(tables.read_bundled(), tables.read_initial_counts())
    tables.verify_display(summary)
    assert summary.groupby("table").size().to_dict() == {"S8": 8, "S9": 4}


def test_draw_pairing_subtracts_payout_and_uses_fixed_initial_counts():
    rows = []
    # The raw draw totals vary. Perfectly paired differences do not.
    counts = {"owner": 2, "renter": 3}
    for case in tables.CASES:
        for draw in reversed(tables.DRAWS):
            for group, count in counts.items():
                change = 0 if case == "uniform" else count * 1000
                rows.append((case, draw, group, 10000 * draw + 2 * change, 1000 * draw + change))
    totals = pd.DataFrame(rows, columns=tables.KEYS + tables.VALUES)
    summary = tables.summarize(totals.sample(frac=1, random_state=1), counts)
    expected = summary["metric"].map({"GUL": 2, "actual_loss": 1})
    for name in tables.STATS:
        pd.testing.assert_series_equal(summary[name], expected.astype(float), check_names=False)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "infinite", "excess_payout"])
def test_rejects_invalid_finance_totals(mutation):
    totals = tables.read_bundled()
    if mutation == "duplicate":
        totals = pd.concat([totals, totals.iloc[[0]]], ignore_index=True)
    elif mutation == "missing":
        totals = totals.iloc[1:]
    elif mutation == "infinite":
        totals.loc[0, "payout_usd"] = float("inf")
    else:
        totals.loc[0, "payout_usd"] = totals.loc[0, "gross_usd"] + 1
    with pytest.raises(ValueError):
        tables.summarize(totals, tables.read_initial_counts())


def test_rejects_totals_that_do_not_match_the_submitted_results():
    totals = tables.read_bundled()
    totals.loc[totals["case"].eq("triangular"), "gross_usd"] += 1e10
    summary = tables.summarize(totals, tables.read_initial_counts())
    with pytest.raises(ValueError, match="triangular"):
        tables.verify_display(summary)


def test_rejects_unpinned_bundle(tmp_path, monkeypatch):
    altered = tmp_path / "unverified.csv"
    altered.write_bytes(tables.DATA.read_bytes() + b"\n")
    monkeypatch.setattr(tables, "DATA", altered)
    with pytest.raises(ValueError, match="Bundled"):
        tables.read_bundled()


def test_cli_replays_from_bundle_and_refuses_overwrite(tmp_path):
    output = tmp_path / "tables"
    command = [sys.executable, str(Path(tables.__file__)), "--out-dir", str(output)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "36 displayed values" in result.stdout
    saved = {path.name: path.read_bytes() for path in output.iterdir()}
    assert set(saved) == {"Table_S8.csv", "Table_S9.csv"}
    assert b"S9,same_year,owner,GUL,-59.9,-63.1,-57.0\n" in saved["Table_S9.csv"]
    assert b"S8,truncated_normal,renter,actual_loss,3.1,2.9,3.4\n" in saved["Table_S8.csv"]
    repeated = subprocess.run(command, check=False, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert "must not exist" in repeated.stderr
    assert {path.name: path.read_bytes() for path in output.iterdir()} == saved

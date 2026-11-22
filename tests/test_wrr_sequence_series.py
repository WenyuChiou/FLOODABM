"""Tests for Figure S6 ensemble summaries and archive validation."""

import hashlib

import pandas as pd
import pytest

from scripts.utilities import rebuild_wrr_sequence_series as series


def test_observed_iqr_differs_from_shuffled_range_of_sequence_medians():
    rows = []
    for order, values in {"observed": [1, 3, 5], "permuted_01": [2, 4, 20],
                          "permuted_02": [5, 7, 50]}.items():
        for draw, gul in enumerate(values, start=1):
            rows.append((order, draw, "baseline", 2011, "owner", gul, .5 * gul))
            rows.append((order, draw, "worst", 2011, "owner", 1000, 1000))
    trajectory = pd.DataFrame(rows, columns=[*series.RUN_ROW, *series.LOSSES])
    plot = series.summarize_baseline(trajectory)
    gul = plot.loc[plot["metric"].eq("GUL")].set_index("sequence")
    assert tuple(gul.loc["Observed sequence", series.PLOT_VALUES]) == (3, 2, 4)
    assert tuple(gul.loc["Shuffled sequences", series.PLOT_VALUES]) == (5.5, 4, 7)


def test_archive_file_hash_rejects_changed_raw_contents(tmp_path):
    file = tmp_path / "finance_tract_2011.csv"
    expected = b"a,b\n1,2\n"
    sha = hashlib.sha256(expected).hexdigest()
    file.write_bytes(expected)
    assert series.verified_bytes(file, sha) == expected
    file.write_bytes(b"a,b\n1,3\n")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        series.verified_bytes(file, sha)


def test_reference_keys_must_match():
    row = pd.DataFrame({"year": [2011], "value": [1.0]})
    with pytest.raises(ValueError, match="sequence keys differ"):
        series.compare(row, row.assign(year=2012), ["year"], ["value"])

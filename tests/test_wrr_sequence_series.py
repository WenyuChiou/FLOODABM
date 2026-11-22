"""Tests for Figure S6 ensemble summaries and archive validation."""

from csv import DictReader
import hashlib
from pathlib import Path
import subprocess

import pandas as pd
import pytest

from scripts.utilities import rebuild_wrr_sequence_series as series
from scripts.utilities import run_wrr_flood_sequence as replay


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


def test_portable_checksum_coverage_and_observed_controls():
    hashes = series.read_public_hashes()
    assert len(hashes) == 220
    with (series.ROOT / "data/supplementary/wrr_mc50_finance_reference.csv").open(
        encoding="utf-8", newline="",
    ) as stream:
        benchmark = {(int(row["run_id"]), row["scenario"]): row["finance_sha256"]
                     for row in DictReader(stream)}
    assert {case[1:]: sha for case, sha in hashes.items() if case[0] == "observed"} == {
        case: sha for case, sha in benchmark.items() if case[0] in series.DRAWS
    }


def test_bundled_case_checksums_are_pinned(tmp_path, monkeypatch):
    modified = tmp_path / "portable_digests.csv"
    payload = series.PUBLIC_HASHES.read_bytes()
    modified.write_bytes(payload.replace(b"observed", b"tampered", 1))
    monkeypatch.setattr(series, "PUBLIC_HASHES", modified)
    with pytest.raises(ValueError, match="differ from the revised reference"):
        series.read_public_hashes()


def test_case_digest_normalizes_line_endings_and_detects_content_changes():
    payloads = {year: b"col\r\n1\r\n" for year in series.YEARS}
    normalized = {year: b"col\n1\n" for year in series.YEARS}
    assert series.case_digest(payloads) == series.case_digest(normalized)
    normalized[2011] = b"col\n2\n"
    assert series.case_digest(payloads) != series.case_digest(normalized)
    del normalized[2012]
    with pytest.raises(ValueError, match="each model year"):
        series.case_digest(normalized)


def test_case_uses_saved_draw_seeds_and_verifies_before_reuse(tmp_path, monkeypatch):
    case = ("permuted_10", 10, "worst")
    suite = tmp_path.resolve()
    payloads = {year: f"year\n{year}\n".encode() for year in series.YEARS}
    expected = series.case_digest(payloads)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        assert kwargs["check"] and kwargs["cwd"] == series.ROOT
        assert kwargs["timeout"] == 120
        assert kwargs["env"]["FLOODABM_POSTERIOR_IDX"] == "864"
        assert command[1:-1] == [
            "main.py", "--scenario", "worst", "--model-dir", "baseline", "--decision-seed", "90334",
            "--output-mode", "summary", "--no-plots", "--wrr-flood-order", "permuted_10", "--out-root",
        ]
        assert Path(command[-1]) == suite / "raw" / "permuted_10" / "draw_10" / "worst"
        for year, payload in payloads.items():
            path = suite / series.relative_finance_path(*case, year)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)

    monkeypatch.setattr(replay.subprocess, "run", fake_run)
    assert replay.run_case(suite, case, expected, 120, False) == "REPLAYED AND VERIFIED"
    with pytest.raises(FileExistsError):
        replay.run_case(suite, case, expected, 120, False)
    assert replay.run_case(suite, case, expected, 120, True) == "VERIFIED EXISTING"
    (suite / series.relative_finance_path(*case, 2011)).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="finance output differs"):
        replay.run_case(suite, case, expected, 120, True)
    assert len(calls) == 1


def test_case_timeout_leaves_log_but_cannot_be_reused(tmp_path, monkeypatch):
    case = ("observed", 1, "baseline")

    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(replay.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match=r"did not finish; see .*run.log"):
        replay.run_case(tmp_path.resolve(), case, "0" * 64, 120, False)
    assert (tmp_path / "raw/observed/draw_01/baseline/run.log").is_file()
    with pytest.raises(FileNotFoundError):
        replay.run_case(tmp_path.resolve(), case, "0" * 64, 120, True)


def test_confined_path_rejects_dot_dot(tmp_path):
    with pytest.raises(ValueError, match="stay within"):
        series.confined_path(tmp_path.resolve(), Path("../other"))


def test_second_csv_write_failure_exposes_no_result(tmp_path, monkeypatch):
    first = pd.DataFrame({"x": [1]})
    second = pd.DataFrame({"y": [2]})
    original = pd.DataFrame.to_csv

    def fail_second(table, path, **kwargs):
        if path.name == series.PLOT:
            path.write_bytes(b"partial")
            raise OSError("second write failed")
        original(table, path, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_second)
    out_dir = tmp_path / "rebuild"
    with pytest.raises(OSError, match="second write failed"):
        series.publish_tables(out_dir, first, second)
    assert not list(tmp_path.iterdir())


def test_second_publish_failure_rolls_back_first_without_clobber(tmp_path, monkeypatch):
    out_dir = tmp_path / "rebuild"
    first = pd.DataFrame({"x": [1]})
    second = pd.DataFrame({"y": [2]})
    original = series.os.link

    def conflict(source, dest):
        if dest.name == series.PLOT:
            dest.write_bytes(b"unrelated")
        original(source, dest)

    monkeypatch.setattr(series.os, "link", conflict)
    with pytest.raises(FileExistsError):
        series.publish_tables(out_dir, first, second)
    assert not (out_dir / series.TRAJECTORY).exists()
    assert (out_dir / series.PLOT).read_bytes() == b"unrelated"

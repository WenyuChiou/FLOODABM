"""Check the independent PNG renderer and the archived AEP data contract."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.utilities import render_wrr_aep_png as plot


def test_curve_simplification_retains_the_native_vector_positions():
    data = plot.read_series(None)
    assert len(data) == 2400
    left, homeowners = plot.native_curves(data, plot.PANELS[0])
    right, renters = plot.native_curves(data, plot.PANELS[1])
    assert left.count(b" l\n") == 241 + 178
    assert right.count(b" l\n") == 330 + 225
    assert b"58.434883 83.534023 m" in left
    assert b"782.394375 139.662924 l" in right
    assert round(100 * (1 - homeowners["Actual loss"] / homeowners["GUL"])) == 44
    assert round(100 * (1 - renters["Actual loss"] / renters["GUL"])) == 45


def test_rebuilt_curve_is_checked_even_at_interior_positions(tmp_path):
    data = plot.read_series(None)
    file = tmp_path / plot.AEP_NAME
    data.sample(frac=1, random_state=1).to_csv(file, index=False)
    assert len(plot.read_series(tmp_path)) == 2400
    data.loc[50, "value"] += .001
    data.to_csv(file, index=False)
    with pytest.raises(ValueError, match="AEP value differs"):
        plot.read_series(tmp_path)


def test_unverified_bundle_fails_closed(tmp_path, monkeypatch):
    extra = tmp_path / plot.AEP_NAME
    extra.write_bytes(plot.SERIES.read_bytes() + b"\n")
    monkeypatch.setattr(plot, "SERIES", extra)
    with pytest.raises(ValueError, match="Bundled AEP series"):
        plot.read_series(None)


def test_png_only_cli_matches_the_reference_and_refuses_overwrites(tmp_path):
    pytest.importorskip("pymupdf")
    windir = os.environ.get("WINDIR")
    if not windir or not all((Path(windir) / "Fonts" / name).is_file()
                            for name in ("times.ttf", "timesbd.ttf")):
        pytest.skip("The exact render requires locally installed Times New Roman")
    command = [sys.executable, plot.__file__, "--figure-dir", str(tmp_path)]
    result = subprocess.run(command, check=True, capture_output=True)
    assert b"exact accepted Figure 7 pixels (PNG only)" in result.stdout
    assert [f.name for f in tmp_path.iterdir()] == [plot.OUT]
    output = (tmp_path / plot.OUT).read_bytes()
    repeated = subprocess.run(command, check=False, capture_output=True)
    assert repeated.returncode != 0
    assert b"Refusing to overwrite" in repeated.stderr
    assert (tmp_path / plot.OUT).read_bytes() == output

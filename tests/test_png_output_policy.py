"""Check new plots are written directly as PNGs by public commands."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_posterior_query_produces_one_png_from_bundled_models(tmp_path):
    cmd = [
        sys.executable, str(ROOT / "calibration/phase2_bayesian_regression/posterior_predictive_query.py"),
        "--tp", "0.6", "--cp", "0.7", "--sp", "0.5", "--model", "owner_FI",
        "--out-dir", str(tmp_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["query_TP0.60_CP0.70_SP0.50.png"]
    with Image.open(tmp_path / names[0]) as image:
        assert image.format == "PNG"
        assert image.size == (1500, 1350)


def test_png_batch_preflights_aep_before_writing_other_plots(tmp_path):
    pytest.importorskip("pymupdf")
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/utilities/render_wrr_figures.py"),
        "--figure-dir", str(tmp_path / "output"), "--figures", "5", "7",
        "--font-dir", str(tmp_path / "absent-fonts"),
    ], check=False, capture_output=True)
    assert result.returncode != 0
    assert b"Figure 7:" in result.stderr
    assert not (tmp_path / "output").exists()


def test_figure_7_batch_route_writes_only_a_verified_png(tmp_path):
    pytest.importorskip("pymupdf")
    windir = os.environ.get("WINDIR")
    if not windir or not all((Path(windir) / "Fonts" / filename).is_file()
                            for filename in ("times.ttf", "timesbd.ttf")):
        pytest.skip("The exact render requires locally installed Times New Roman")
    out = tmp_path / "png"
    subprocess.run([
        sys.executable, str(ROOT / "scripts/utilities/render_wrr_figures.py"),
        "--figure-dir", str(out), "--figures", "7",
    ], check=True, capture_output=True)
    assert [p.name for p in out.iterdir()] == ["Figure_07.png"]
    subprocess.run([
        sys.executable, str(ROOT / "scripts/utilities/verify_wrr_figures.py"),
        "--figure-dir", str(out), "--figures", "7",
    ], check=True, capture_output=True)


def test_figure_6_batch_uses_only_csv_and_writes_only_a_verified_png(tmp_path):
    pytest.importorskip("pymupdf")
    windir = os.environ.get("WINDIR")
    if not windir or not all((Path(windir) / "Fonts" / filename).is_file()
                            for filename in ("times.ttf", "timesbd.ttf")):
        pytest.skip("The exact render requires locally installed Times New Roman")
    out = tmp_path / "png"
    subprocess.run([
        sys.executable, str(ROOT / "scripts/utilities/render_wrr_figures.py"),
        "--figure-dir", str(out), "--figures", "6",
    ], check=True, capture_output=True)
    assert [p.name for p in out.iterdir()] == ["Figure_06.png"]
    subprocess.run([
        sys.executable, str(ROOT / "scripts/utilities/verify_wrr_figures.py"),
        "--figure-dir", str(out), "--figures", "6",
    ], check=True, capture_output=True)


def test_figure_6_batch_refuses_incomplete_series_before_writing(tmp_path):
    pytest.importorskip("pymupdf")
    windir = os.environ.get("WINDIR")
    if not windir or not all((Path(windir) / "Fonts" / filename).is_file()
                            for filename in ("times.ttf", "timesbd.ttf")):
        pytest.skip("The exact render requires locally installed Times New Roman")
    data = tmp_path / "series"
    data.mkdir()
    bundled = ROOT / "data" / "figures"
    names = ["Figure5_nfip_validation_corrected_county_labels.csv",
             "Figure5_metrics_corrected_county_labels.csv", "Figure6_cumulative_loss.csv"]
    for name in names:
        shutil.copyfile(bundled / name, data / name)
    name = "Figure6_annual_finance.csv"
    (data / name).write_bytes(b"\n".join((bundled / name).read_bytes().splitlines()[:-1]) + b"\n")
    out = tmp_path / "png"
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/utilities/render_wrr_figures.py"),
        "--figure-dir", str(out), "--series-dir", str(data), "--figures", "5", "6",
    ], check=False, capture_output=True)
    assert result.returncode != 0
    assert b"Figure 6: Unexpected figure table row count" in result.stderr
    assert not out.exists()

from __future__ import annotations

import argparse
import csv
import hashlib
from io import StringIO

import pytest

from scripts.utilities import render_wrr_public_maps as maps


def mapped_series(offset=0):
    fi, population = StringIO(), StringIO()
    writer = csv.writer(fi)
    writer.writerow(["group", "tract_geoid", "bivariate_2021", "bivariate_2022", "bivariate_2023"])
    for group in ("homeowner", "renter"):
        for i in range(27):
            writer.writerow([group, f"{i:011d}", 1, 2, 3])
    writer = csv.writer(population)
    fields = [name for _, _, name in maps.spatial.POPULATION_PANELS.values()]
    writer.writerow(["tract_geoid", *fields])
    for i in range(27):
        writer.writerow([f"{i + offset:011d}", *(["0.000000000"] * 6)])
    return fi.getvalue().encode("utf-8-sig"), population.getvalue().encode("ascii")


def test_plot_tables_must_contain_same_tracts():
    fi, population = mapped_series()
    assert maps.population_tract_ids(fi, population) == tuple(f"{i:011d}" for i in range(27))
    _, other = mapped_series(offset=1)
    with pytest.raises(ValueError, match="different tracts"):
        maps.population_tract_ids(fi, other)


def test_pinned_input_may_be_renamed_but_not_changed(tmp_path):
    file = tmp_path / "some-layout.pagx"
    file.write_bytes(b"pinned")
    assert maps.load_layout(file, hashlib.sha256(b"pinned").hexdigest()) == b"pinned"
    with pytest.raises(ValueError, match="pinned revision"):
        maps.load_layout(file, hashlib.sha256(b"other").hexdigest())


def test_offline_cache_does_not_fall_back_to_network(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("network access was attempted with an explicit cache")

    monkeypatch.setattr("scripts.utilities.wrr_public_sources.build_opener", unexpected)
    with pytest.raises(FileNotFoundError):
        maps.public_input(tmp_path, "tracts.zip", "https://example.net/tracts.zip", 50)
    (tmp_path / "tracts.zip").write_bytes(b"test data")
    assert maps.public_input(tmp_path, "tracts.zip", "https://example.net/tracts.zip", 50) == b"test data"
    with pytest.raises(ValueError, match="input size"):
        maps.public_input(tmp_path, "tracts.zip", "https://example.net/tracts.zip", 2)


def test_output_directory_is_not_reused(tmp_path):
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        maps.run(argparse.Namespace(output_dir=tmp_path))


def test_wrong_layout_rejected_before_fetch(tmp_path, monkeypatch):
    layout = tmp_path / "wrong.pagx"
    blank = tmp_path / "Blank.aprx"
    executable = tmp_path / "python"
    for path in (layout, blank, executable):
        path.touch()

    def unexpected(*args, **kwargs):
        pytest.fail("source preparation was called for unverified layout bytes")

    monkeypatch.setattr(maps, "write_result_layers", unexpected)
    args = argparse.Namespace(
        output_dir=tmp_path / "output",
        blank_project=blank,
        arcgis_python=executable,
        figure10_layout=layout,
        s7_layout=layout,
    )
    with pytest.raises(ValueError, match="pinned revision"):
        maps.run(args)
    assert not args.output_dir.exists()


def test_wrong_blank_project_rejected_before_fetch(tmp_path, monkeypatch):
    blank = tmp_path / "archived.aprx"
    blank.write_bytes(b"not the stock project")
    monkeypatch.setattr(maps, "load_layout", lambda path, digest: b"approved layout")

    def unexpected(*args, **kwargs):
        pytest.fail("source preparation was called for unverified blank project bytes")

    monkeypatch.setattr(maps, "write_result_layers", unexpected)
    args = argparse.Namespace(
        output_dir=tmp_path / "output",
        blank_project=blank,
        arcgis_python=blank,
        figure10_layout=blank,
        s7_layout=blank,
    )
    with pytest.raises(ValueError, match="stock Blank.aprx"):
        maps.run(args)
    assert not args.output_dir.exists()


def test_bad_second_figure_leaves_no_final_output(tmp_path, monkeypatch):
    output = tmp_path / "output"
    (tmp_path / maps.spatial.LAYOUT_INPUT).write_bytes(b"raw main")
    monkeypatch.setattr(maps.raster, "finish_spatial_fi_map", lambda raw: b"accepted main")
    monkeypatch.setattr(
        maps.raster, "FIGURE10_FINAL_SHA", hashlib.sha256(b"accepted main").hexdigest()
    )

    def unequal(raw, output_file):
        raise ValueError("Native Figure S7 PNG differs")

    monkeypatch.setattr(maps.raster, "render_native_population_map", unequal)
    with pytest.raises(ValueError, match="S7 PNG differs"):
        maps.finish_pngs(tmp_path, output)
    assert not output.exists()


def test_both_pngs_written_after_verification(tmp_path, monkeypatch):
    output = tmp_path / "output"
    (tmp_path / maps.spatial.LAYOUT_INPUT).write_bytes(b"raw main")
    monkeypatch.setattr(maps.raster, "finish_spatial_fi_map", lambda raw: b"accepted main")
    monkeypatch.setattr(
        maps.raster, "FIGURE10_FINAL_SHA", hashlib.sha256(b"accepted main").hexdigest()
    )
    monkeypatch.setattr(
        maps.raster, "render_native_population_map", lambda raw, path: path.write_bytes(b"accepted S7")
    )
    maps.finish_pngs(tmp_path, output)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == {
        maps.spatial.OUTPUT: b"accepted main",
        maps.spatial.S7_OUTPUT: b"accepted S7",
    }


def test_second_copy_failure_leaves_no_partial_final_directory(tmp_path, monkeypatch):
    verified = tmp_path / "verified"
    verified.mkdir()
    for name in (maps.spatial.OUTPUT, maps.spatial.S7_OUTPUT):
        (verified / name).write_bytes(b"verified PNG")
    copy = maps.shutil.copyfileobj

    def fail_second(source, target):
        if maps.spatial.S7_OUTPUT in target.name:
            target.write(b"partial")
            raise OSError("simulated disk-full error")
        copy(source, target)

    monkeypatch.setattr(maps.shutil, "copyfileobj", fail_second)
    output = tmp_path / "output"
    with pytest.raises(OSError, match="disk-full"):
        maps.publish_pngs(verified, output)
    assert not output.exists()
    assert list(tmp_path.iterdir()) == [verified]

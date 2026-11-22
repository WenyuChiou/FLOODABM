import sys

import pytest

from scripts.utilities import export_wrr_spatial_fi_png as renderer


def test_rebuilt_classes_cover_all_six_panels_and_match_native_rows():
    geoids = [f"34013{i:06d}" for i in range(27)]
    header = "group,tract_geoid,bivariate_2021,bivariate_2022,bivariate_2023\n"
    records = "".join(
        f"{group},{geoid},{classes}\n"
        for group, classes in (("homeowner", "1,2,3"), ("renter", "7,8,9"))
        for geoid in reversed(geoids)
    )
    panels = renderer.expected_classes((header + records).encode())
    assert len(panels) == 6
    assert panels["homeowner", 2022] == dict.fromkeys(geoids, 2)
    assert panels["renter", 2023] == dict.fromkeys(geoids, 9)
    expected = panels["homeowner", 2022]
    rows = list(expected.items())
    renderer.verify_panel(rows[::-1], expected, "homeowner 2022")
    with pytest.raises(ValueError, match="incomplete"):
        renderer.verify_panel(rows[:-1], expected, "homeowner 2022")
    with pytest.raises(ValueError, match="mapped class"):
        renderer.verify_panel([*rows, rows[0]], expected, "homeowner 2022")
    with pytest.raises(ValueError, match="mapped class"):
        renderer.verify_panel([(rows[0][0], 2.5), *rows[1:]], expected, "homeowner 2022")
    with pytest.raises(ValueError, match="one bivariate class"):
        renderer.expected_classes((header + records + records.splitlines()[0] + "\n").encode())


def test_existing_output_is_never_opened_or_overwritten(tmp_path):
    existing = tmp_path / renderer.OUTPUT
    existing.write_bytes(b"existing output")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        renderer.render_project(tmp_path / "nonexistent.aprx", tmp_path / "nonexistent.csv",
                                tmp_path / "nonexistent.exe", tmp_path)
    assert existing.read_bytes() == b"existing output"


def test_wrong_project_is_rejected_before_launching_arcgis(tmp_path, monkeypatch):
    project = tmp_path / "Figure10_ready.aprx"
    project.write_bytes(b"not the revised project")
    destination = tmp_path / "output"
    monkeypatch.setattr(renderer.subprocess, "run", lambda *a, **kw: pytest.fail("Launched ArcGIS"))
    with pytest.raises(ValueError, match="does not match the revised Figure 10 input"):
        renderer.render_project(project, tmp_path / "series.csv", tmp_path / "python", destination)
    assert not destination.exists()


@pytest.mark.parametrize("concurrent_write", [False, True])
def test_native_worker_writes_its_page_exclusively(tmp_path, monkeypatch, concurrent_write):
    target = tmp_path / renderer.LAYOUT_INPUT
    monkeypatch.setattr(sys, "argv", ["export", "--export-full-page", "--project", "project.aprx",
                                      "--series", "series.csv", "--page-png", str(target)])

    def export(_project, _series, scratch):
        assert scratch.parent != target.parent
        scratch.write_bytes(b"ArcGIS output")
        if concurrent_write:
            target.write_bytes(b"concurrent output")

    monkeypatch.setattr(renderer, "export_native_layout", export)
    if concurrent_write:
        with pytest.raises(SystemExit) as error:
            renderer.main()
        assert error.value.code == 1
        assert target.read_bytes() == b"concurrent output"
    else:
        renderer.main()
        assert target.read_bytes() == b"ArcGIS output"

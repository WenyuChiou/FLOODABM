import hashlib

import pytest

from scripts.utilities import render_wrr_supporting_raster as renderer


def test_modified_source_stops_before_plotting(tmp_path, monkeypatch):
    path = tmp_path / "FigS01_tp_decay_calibration.png"
    path.write_bytes(b"not the pinned archive")
    monkeypatch.setattr(
        renderer, "relabel", lambda *_: pytest.fail("unexpected rendering")
    )
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="does not match the pinned archive"):
        renderer.render_checked(tmp_path, out, tmp_path, [1])
    assert not out.exists()


def test_existing_target_is_preserved(tmp_path):
    target = tmp_path / "FigS02_irene_depth_differences.png"
    target.write_bytes(b"keep")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        renderer.render_checked(tmp_path, tmp_path, tmp_path, [2])
    assert target.read_bytes() == b"keep"


def test_pngs_written_only_after_all_renders_match(tmp_path, monkeypatch):
    asset = b"synthetic source"
    image = b"synthetic png"
    asset_sha = hashlib.sha256(asset).hexdigest()
    image_sha = hashlib.sha256(image).hexdigest()
    for number in (1, 2):
        stem, _, ref = renderer.REFERENCES[number]
        (tmp_path / f"{stem}.png").write_bytes(asset)
        monkeypatch.setitem(
            renderer.REFERENCES,
            number,
            (stem, asset_sha, image_sha if number == 1 else ref),
        )
    monkeypatch.setattr(renderer, "relabel", lambda *_: image)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="S2 PNG differs"):
        renderer.render_checked(tmp_path, out, tmp_path, [1, 2])
    assert not out.exists()


def test_s7_checks_archived_source_name_and_uses_final_output_name(
    tmp_path, monkeypatch
):
    asset = b"pre-crop map"
    finished = b"finished map"
    (tmp_path / "FigS4_population_change.png").write_bytes(asset)
    name, _, _ = renderer.REFERENCES[7]
    monkeypatch.setitem(
        renderer.REFERENCES,
        7,
        (name, hashlib.sha256(asset).hexdigest(), hashlib.sha256(finished).hexdigest()),
    )
    monkeypatch.setattr(renderer, "finish_population_map", lambda data: finished)
    monkeypatch.setattr(
        renderer, "relabel", lambda *_: pytest.fail("unexpected font use")
    )
    out = tmp_path / "out"
    renderer.render_checked(tmp_path, out, tmp_path, [7])
    assert (out / "FigS07_population_change.png").read_bytes() == finished


def test_figure10_rejects_a_previously_cropped_or_alternate_map(tmp_path):
    (tmp_path / "Fig10_spatial_fi.png").write_bytes(b"wrong export")
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="not the pinned full-page map"):
        renderer.render_main_map(tmp_path, out)
    assert not out.exists()


def test_figure10_verifies_final_encoding_before_writing(tmp_path, monkeypatch):
    source = b"pinned map"
    (tmp_path / "Fig10_spatial_fi.png").write_bytes(source)
    monkeypatch.setattr(
        renderer, "FIGURE10_SOURCE_SHA", hashlib.sha256(source).hexdigest()
    )
    monkeypatch.setattr(
        renderer, "finish_spatial_fi_map", lambda _: b"incorrect encoder"
    )
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="PNG differs from the final archive"):
        renderer.render_main_map(tmp_path, out)
    assert not out.exists()

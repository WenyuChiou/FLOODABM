from io import BytesIO

from PIL import Image
import pytest

from scripts.utilities import render_wrr_vector_figures as renderer


def test_requires_only_the_components_for_the_selected_figure(tmp_path, monkeypatch):
    fixture = b"fixture for the one-input route"
    (tmp_path / renderer.DIAGNOSTICS).write_bytes(fixture)
    monkeypatch.setitem(renderer.COMPONENTS, renderer.DIAGNOSTICS, renderer.digest(fixture))

    assert renderer.read_components(tmp_path, [4]) == {renderer.DIAGNOSTICS: fixture}
    with pytest.raises(FileNotFoundError, match=renderer.CUMULATIVE):
        renderer.read_components(tmp_path, [4, 6])


def test_refuses_wrong_component_even_with_expected_filename(tmp_path):
    (tmp_path / renderer.FINANCE).write_bytes(b"not the pinned source")

    with pytest.raises(ValueError, match="component SHA-256 mismatch"):
        renderer.read_components(tmp_path, [7])


def test_checks_both_dimensions_and_rgba_pixels():
    image = Image.new("RGB", (3, 2), (16, 32, 64))
    with BytesIO() as output:
        image.save(output, "PNG")
        png = output.getvalue()
    expected = {"width": 3, "height": 2, "rgba_sha256": renderer.digest(image.convert("RGBA").tobytes())}

    renderer.verify_pixels(png, expected)
    with pytest.raises(ValueError, match="pixels or dimensions"):
        renderer.verify_pixels(png, {**expected, "width": 2})
    with pytest.raises(ValueError, match="pixels or dimensions"):
        renderer.verify_pixels(png, {**expected, "rgba_sha256": "0" * 64})

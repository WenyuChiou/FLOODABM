from pathlib import Path

import pytest
from PIL import Image

from scripts.utilities.verify_wrr_figures import fingerprint


def test_static_png_compression_is_not_part_of_fingerprint(tmp_path: Path) -> None:
    sample = Image.new("RGB", (2, 1), "red")
    uncompressed = tmp_path / "uncompressed.png"
    compressed = tmp_path / "compressed.png"
    sample.save(uncompressed, compress_level=0)
    sample.save(compressed, compress_level=9)
    assert uncompressed.read_bytes() != compressed.read_bytes()
    assert fingerprint(uncompressed) == fingerprint(compressed)


def test_animated_png_is_rejected_even_if_the_first_frame_matches(tmp_path: Path) -> None:
    frames = [Image.new("RGBA", (2, 1), color) for color in ("red", "blue")]
    animated = tmp_path / "animated.png"
    frames[0].save(animated, save_all=True, append_images=frames[1:], duration=200)
    with pytest.raises(OSError, match="Expected a static PNG"):
        fingerprint(animated)

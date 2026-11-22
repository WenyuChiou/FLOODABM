"""Check output PNG pixels against the revised WRR main figure exports."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "data" / "supplementary" / "wrr_main_figures_reference.json"


def fingerprint(path: Path) -> dict:
    with Image.open(path, formats=["PNG"]) as png:
        if getattr(png, "is_animated", False):
            raise OSError("Expected a static PNG, not an animated PNG")
        rgba = png.convert("RGBA")
        return {
            "width": rgba.width,
            "height": rgba.height,
            "rgba_sha256": hashlib.sha256(rgba.tobytes()).hexdigest(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--figures", type=int, nargs="+", choices=range(1, 12))
    args = parser.parse_args()
    manifest = json.loads(REFERENCE.read_text(encoding="utf-8"))
    names = [f"Figure_{n:02}.png" for n in range(1, 12)]
    if manifest.get("schema_version") != 1 or set(manifest.get("figures", {})) != set(names):
        raise ValueError("Incomplete or unsupported WRR figure reference")
    if args.figures is not None:
        names = [f"Figure_{n:02}.png" for n in sorted(set(args.figures))]
    failed = 0
    for name in names:
        try:
            actual = fingerprint(args.figure_dir / name)
        except OSError as exc:
            failed += 1
            print(f"FAIL {name}: {exc}")
            continue
        if actual != manifest["figures"][name]:
            failed += 1
            print(f"FAIL {name}: pixels or dimensions differ")
        else:
            print(f"PASS {name}: exact decoded pixels")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

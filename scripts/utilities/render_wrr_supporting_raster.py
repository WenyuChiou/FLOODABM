"""Reproduce Figure 10 and the accepted S1–S3 and S7 PNGs from archive layers.

The final label and layout edits preserve the underlying charts and maps.
This does not refit the behavioral model, calculate flood depths, or export
the GIS layout. The archive inputs and Windows Arial are not bundled.
"""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import os
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps


# Final standalone Supporting Information artwork and its pre-relabeling layers.
REFERENCES = {
    1: (
        "FigS01_tp_decay_calibration",
        "220c2e15b74d401d72b84ac7210113d8011295a3cf8aa540766c8e79b9b81379",
        "ed32c97d6762bff1bae6c8bd6407021a87fbc1ace6470351304d063ac023225e",
    ),
    2: (
        "FigS02_irene_depth_differences",
        "ac2fa7cebab7c280623fc35e4d1a3a5c27fd18cdc3635ed049248f8a1c918599",
        "46555fe8d716afa8ad20c69837153740916000dabc2edf32c1dc1f3b2c2507e7",
    ),
    3: (
        "FigS03_annual_flood_depth_maps",
        "e0d9b558502368c7e5985df3117b7d63f1f78d51a277e5d3df9d0cace688f399",
        "b1fee155116e2dca2e54a905acee32554b839d878c9472d73a10f0683418f1b7",
    ),
    7: (
        "FigS07_population_change",
        "af0ba660570f3977a7a1ba27bc7dfebdd507b34dfad98b0277823a1c7f031b47",
        "3441741345c7c4b42087c7d9c30dc035c786e3682de9d1f0035f06742176c50d",
    ),
}
SOURCE_NAMES = {7: "FigS4_population_change.png"}
FIGURE10_SOURCE_SHA = "685f5a7a2590376e50d6fbb7694697a8b1c91063f39ab80c66be39e52cec2653"
FIGURE10_FINAL_SHA = "0c82d883709f5dc3a52a154deda94f5e98d1b84a2e9a18cb8a40e894dc9a61aa"


def encode_png(im: Image.Image, *, optimize: bool = False) -> bytes:
    png = BytesIO()
    im.save(png, format="PNG", dpi=(300, 300), optimize=optimize)
    return png.getvalue()


def finish_population_map(source: bytes) -> bytes:
    """Apply the final S7 crop and translate the flood-prone legend entry."""
    with Image.open(BytesIO(source), formats=["PNG"]) as original:
        im = original.convert("RGB")
    # Coordinates belong to the pinned 8208 × 4555 raster. Limit modifications
    # to the blank legend margin; the population map panels are not rerendered.
    finished = im.crop((2, 2, 7945, 4553))
    finished.paste("white", (6700, 1510, 7943, 1750))
    finished.paste(im.crop((7248, 1512, 8208, 1752)), (6700, 1510))
    # The label and symbol move left together. The symbol also moves 27 px up.
    finished.paste("white", (6786, 1581, 6900, 1716))
    finished.paste(im.crop((7334, 1610, 7448, 1718)), (6786, 1581))
    return encode_png(finished)


def finish_spatial_fi_map(source: bytes) -> bytes:
    with Image.open(BytesIO(source), formats=["PNG"]) as original:
        # Trim the outer page margins of the pinned centered-key GIS export.
        artwork = original.convert("RGB").crop((1197, 843, 10635, 5238))
    return encode_png(ImageOps.expand(artwork, border=94, fill="white"), optimize=True)


def render_native_population_map(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {target}")
    with Image.open(source, formats=["PNG"]) as original:
        if original.size != (10200, 6600):
            raise ValueError("Unexpected full-page size for the native S7 project")
        # Keep the final legend and the six panels from the native page.
        artwork = original.crop((1197, 843, 8984, 5238)).convert("RGB")
    png = encode_png(ImageOps.expand(artwork, border=78, fill="white"))
    if hashlib.sha256(png).hexdigest() != REFERENCES[7][2]:
        raise ValueError("Native Figure S7 PNG differs from the revised submission")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(png)
    print(f"PASS {target}: exact accepted PNG bytes")


def render_main_map(source_dir: Path, output_dir: Path) -> None:
    target = output_dir / "Figure_10.png"
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {target}")
    source_path = source_dir / "Fig10_spatial_fi.png"
    source = source_path.read_bytes()
    if hashlib.sha256(source).hexdigest() != FIGURE10_SOURCE_SHA:
        raise ValueError(
            f"Figure 10 source is not the pinned full-page map: {source_path}"
        )
    png = finish_spatial_fi_map(source)
    if hashlib.sha256(png).hexdigest() != FIGURE10_FINAL_SHA:
        raise ValueError(
            "Figure 10 PNG differs from the final archive. Use the pinned Pillow build."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(png)
    print(f"PASS {target}: exact accepted PNG bytes")


def relabel(number: int, source: bytes, font_dir: Path) -> bytes:
    with Image.open(BytesIO(source), formats=["PNG"]) as original:
        im = original.convert("RGB")
    old = im.copy()
    d = ImageDraw.Draw(im)
    rects = []
    font = str(font_dir / "arial.ttf")
    bold = str(font_dir / "arialbd.ttf")

    def clear(box):
        d.rectangle((box[0], box[1], box[2] - 1, box[3] - 1), fill="white")
        rects.append(box)

    def txt(text, x, y, size, b=False, anchor="lt"):
        d.text(
            (x, y),
            text,
            font=ImageFont.truetype(bold if b else font, size),
            fill="black",
            anchor=anchor,
        )

    if number == 1:
        clear((0, 0, 1770, 90))
        txt("Bootstrap Parameter Uncertainty (N=100)", 960, 22, 55, anchor="mt")
        clear((0, 94, 154, 1168))
        for val, y in [
            ("1.0", 141),
            ("0.8", 370),
            ("0.6", 600),
            ("0.4", 830),
            ("0.2", 1059),
        ]:
            txt(val, 146, y, 48, anchor="rm")
        # Place the vertical label within its original outer margin.
        layer = Image.new("RGBA", (720, 70))
        ld = ImageDraw.Draw(layer)
        ld.text(
            (0, 0), "Relative TP(t)", font=ImageFont.truetype(font, 57), fill="black"
        )
        layer = layer.crop(layer.getbbox()).rotate(90, expand=True)
        im.paste(layer, (12, 630 - layer.height // 2), layer)
        clear((174, 1184, 1740, 1319))
        for i in range(9):
            txt(str(i * 5), 244 + i * (1426 / 8), 1195, 48, anchor="mt")
        txt("Years since Flood", 959, 1260, 55, anchor="mt")
        clear((1070, 107, 1725, 361))
        d.rectangle((1071, 108, 1723, 359), outline="#d6d6d6", width=3)
        for j, (label, color) in enumerate(
            [
                ("Homeowner 95% CI", old.getpixel((1280, 140))),
                ("Homeowner median", old.getpixel((1280, 193))),
                ("Renter 95% CI", old.getpixel((1280, 245))),
                ("Renter median", old.getpixel((1280, 299))),
            ]
        ):
            y = 126 + j * 59
            if j % 2 == 0:
                d.rectangle((1087, y + 6, 1170, y + 33), fill=color)
            else:
                d.line((1087, y + 21, 1170, y + 21), fill=color, width=8)
            txt(label, 1190, y, 42)
    elif number == 2:
        # Only the already-white legend, header margins and panel letters.
        clear((629, 120, 857, 582))
        txt("Legend", 742, 126, 52, True, anchor="mt")
        txt("Difference (m)", 742, 205, 34, anchor="mt")
        colors = [old.getpixel((642, y)) for y in [272, 329, 387, 444, 501, 559]]
        labels = ["< −0.5", "−0.5–0", "0–0.5", "0.5–1.5", "1.5–3.0", "> 3.0"]
        for j, (label, color) in enumerate(zip(labels, colors)):
            y = 262 + j * 53
            d.rectangle((635, y + 8, 669, y + 28), fill=color)
            txt(label, 678, y, 44)
        for text, x in [("(a)", 0), ("(b)", 755)]:
            clear((x, 0, x + 100, 65))
            txt(text, x + 10, 14, 36, True)
    else:
        # Existing legend only: keep the map panels, north arrow and scale geometry.
        clear((1716, 125, 2028, 650))
        txt("Legend", 1874, 150, 52, True, anchor="mt")
        txt("Simulation", 1874, 219, 42, anchor="mt")
        txt("domain", 1874, 266, 42, anchor="mt")
        d.rectangle((1730, 315, 1790, 342), outline="#e95656", width=3)
        txt("Flood depth (m)", 1874, 356, 40, anchor="mt")
        colors = [old.getpixel((1750, y)) for y in [392, 436, 480, 524, 568]]
        for j, (label, color) in enumerate(
            zip(["0–1.5", "1.6–3.0", "3.1–4.5", "4.6–6.0", "> 6.0"], colors)
        ):
            y = 413 + j * 44
            d.rectangle((1730, y, 1785, y + 26), fill=color)
            txt(label, 1800, y - 3, 40)

    diff = ImageChops.difference(old, im)
    for box in rects:
        diff.paste((0, 0, 0), box)
    if diff.getbbox() is not None:
        raise ValueError(f"S{number} changed pixels outside the approved label regions")
    return encode_png(im)


def render_checked(
    source_dir: Path, output_dir: Path, font_dir: Path, figures: list[int]
) -> None:
    names = {n: REFERENCES[n][0] + ".png" for n in figures}
    targets = {n: output_dir / name for n, name in names.items()}
    for target in targets.values():
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {target}")

    sources = {}
    for n, name in names.items():
        path = source_dir / SOURCE_NAMES.get(n, name)
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != REFERENCES[n][1]:
            raise ValueError(
                f"S{n} source raster does not match the pinned archive: {path}"
            )
        sources[n] = data

    outputs = {}
    for n, data in sources.items():
        png = finish_population_map(data) if n == 7 else relabel(n, data, font_dir)
        if hashlib.sha256(png).hexdigest() != REFERENCES[n][2]:
            raise ValueError(
                f"S{n} PNG differs from the final archive. Use the pinned Pillow build"
                + (" and original Windows Arial fonts." if n != 7 else ".")
            )
        outputs[n] = png

    output_dir.mkdir(parents=True, exist_ok=True)
    for n, png in outputs.items():
        with targets[n].open("xb") as output:
            output.write(png)
        print(f"PASS {targets[n]}: exact accepted PNG bytes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--components",
        required=True,
        type=Path,
        help="Directory containing the pinned pre-relabeling PNGs",
    )
    parser.add_argument("--figure-dir", required=True, type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--figures",
        nargs="+",
        type=int,
        choices=tuple(REFERENCES),
    )
    selection.add_argument("--main-map", type=int, choices=(10,))
    parser.add_argument(
        "--font-dir",
        type=Path,
        default=Path(os.environ.get("SystemRoot", "C:/Windows")) / "Fonts",
    )
    args = parser.parse_args()
    try:
        if args.main_map == 10:
            render_main_map(args.components, args.figure_dir)
        else:
            render_checked(
                args.components,
                args.figure_dir,
                args.font_dir,
                sorted(set(args.figures or (1, 2, 3))),
            )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

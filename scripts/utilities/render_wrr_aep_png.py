"""Redraw the submitted Figure 7 PNG from the verified exceedance series."""

import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path

import matplotlib as mpl
from matplotlib.path import Path as PlotPath
import numpy as np
import pandas as pd

if __package__:
    from .rebuild_wrr_finance_series import AEP_NAME, ROOT, compare_aep
    from .render_wrr_vector_figures import RGB_BLUE, RGB_RED, verify_pixels
    from .verify_wrr_figures import REFERENCE
else:
    from rebuild_wrr_finance_series import AEP_NAME, ROOT, compare_aep
    from render_wrr_vector_figures import RGB_BLUE, RGB_RED, verify_pixels
    from verify_wrr_figures import REFERENCE


SERIES = ROOT / "data" / "figures" / AEP_NAME
SERIES_HASH = "a58a77f9ce8d86915acc59f5c5a309f6b0b5b3fd5b9afbc82c76e4c9d9707f12"
OUT = "Figure_07.png"
PAGE_SIZE = (851.9381103515625, 329.5286865234375)
# The Matplotlib path stream used these axes in its original point coordinates.
# The final figure crops the bottom of that page before adding its labels.
SOURCE_HEIGHT, SOURCE_CUT = 889.0286865234375, 589.5
PLOT_WIDTH, PLOT_HEIGHT, PLOT_FLOOR = 300.305, 223.8695588235, 41.825
TOP, BOTTOM = 43.834137, 267.703674
GRID = (.6901198, .6902419, .6901045)
DARK_GRAY = (.2999771, .3000382, .2999771)

# Both panels retain distinct linear y-axis scales. The diagonal connector
# geometry is a fixed layout annotation, not a line sampled from the model.
PANELS = (
    ("homeowner", 57.934375, 3.782192116191215, 26.641881467817942,
     "278.845786 232.187256 m 303.274762 213.988475 327.703737 195.789693 352.132713 177.590912 c"),
    ("renter", 482.089375, 81.597622283838135, 38.151801081470303,
     "703.95994 233.077488 m 728.059492 215.941849 752.159044 198.80621 776.258596 181.670571 c"),
)


def read_series(series_dir: Path | None) -> pd.DataFrame:
    raw = SERIES.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(raw).hexdigest() != SERIES_HASH:
        raise ValueError("Bundled AEP series differs from the revised Figure 7 reference")
    expected = pd.read_csv(BytesIO(raw), encoding="utf-8-sig", float_precision="round_trip")
    if series_dir is None:
        return expected
    data = pd.read_csv(series_dir / AEP_NAME, encoding="utf-8-sig", float_precision="round_trip")
    if not set(expected.columns).issubset(data.columns):
        raise ValueError("Missing cumulative exceedance coordinates or loss values")
    compare_aep(data, expected, tolerance=1e-8)
    return data


def native_curves(data: pd.DataFrame, panel: tuple) -> tuple[bytes, dict[str, float]]:
    group, left, scale, intercept, connector = panel
    commands, endpoint = [], {}
    for metric, rgb in (("GUL", "0.8392156863 0.1568627451 0.1568627451"),
                        ("Actual loss", "0 0.4666666667 0.7137254902")):
        rows = data.loc[data["group"].eq(group) & data["metric"].eq(metric)].sort_values("aep", ascending=False)
        points = np.column_stack((left + PLOT_WIDTH * (1 - rows["aep"]),
                                  intercept + scale * rows["value"] / 1000))
        with mpl.rc_context({"path.simplify": True, "path.simplify_threshold": 0.111111111111}):
            path = PlotPath(points).cleaned(simplify=True, curves=False).vertices[:-1]
        # Retain the backend's repeated terminal endpoint. Removing it changes
        # the projecting cap on the clipped curve and affects the last pixels.
        clipping = f"{left} {PLOT_FLOOR} {PLOT_WIDTH} {PLOT_HEIGHT} re W n"
        commands.append(f"q {clipping} 2 J 1 j 2 w {rgb} RG")
        commands.extend(f"{x:.6f} {y:.6f} {'m' if i == 0 else 'l'}" for i, (x, y) in enumerate(path))
        commands.append("S Q")
        endpoint[metric] = float(rows["value"].iloc[-1])
    commands.append(f"q 1 J 1 j 0.6 w 0.4 G {connector} S Q")
    return ("\n".join(commands) + "\n").encode("ascii"), endpoint


def legend_underlay(page, left: float) -> None:
    """Retain the visible left edge of the earlier legend beneath its replacement."""
    shape = page.new_shape()
    x0, x1 = left + 5.5, left + 92.11875
    y0, y1, radius = 49.334137, 81.552872, 2.2
    corner = radius / 3
    shape.draw_line((x0 + radius, y1), (x1 - radius, y1))
    shape.draw_bezier((x1 - radius, y1), (x1 - corner, y1), (x1, y1 - corner), (x1, y1 - radius))
    shape.draw_line((x1, y1 - radius), (x1, y0 + radius))
    shape.draw_bezier((x1, y0 + radius), (x1, y0 + corner), (x1 - corner, y0), (x1 - radius, y0))
    shape.draw_line((x1 - radius, y0), (x0 + radius, y0))
    shape.draw_bezier((x0 + radius, y0), (x0 + corner, y0), (x0, y0 + corner), (x0, y0 + radius))
    shape.draw_line((x0, y0 + radius), (x0, y1 - radius))
    shape.draw_bezier((x0, y1 - radius), (x0, y1 - corner), (x0 + corner, y1), (x0 + radius, y1))
    shape.finish(width=1, color=(.7999237, .8000305, .7998932), fill=(1, 1, 1),
                 stroke_opacity=.92, fill_opacity=.92)
    shape.commit()


def render_png(pdf, data: pd.DataFrame, fonts: tuple[bytes, bytes]) -> bytes:
    """Use a temporary in-memory drawing surface; return only a PNG."""
    with pdf.open() as canvas:
        pw, ph = PAGE_SIZE
        page = canvas.new_page(width=pw, height=ph)
        names = ("WCRegular", "WCBold")
        metrics = [pdf.Font(fontbuffer=font) for font in fonts]
        for name, font in zip(names, fonts, strict=True):
            page.insert_font(fontname=name, fontbuffer=font)

        def label(text, size, x, y, *, bold=False, align=0, rotate=0, color=(0, 0, 0)):
            if not rotate:
                x -= metrics[bold].text_length(text, fontsize=size) * align
            page.insert_text((x, y), text, fontsize=size, fontname=names[bold], color=color, rotate=rotate)

        for i, panel in enumerate(PANELS):
            left, right, scale, offset, panel_x, ylabel_x = (
                (57.934376, 358.23938, 3.78219215, 282.8868054, 27.903875, 20.934376),
                (482.089386, 782.394348, 81.59762265, 271.3768858, 452.058868, 442.089386),
            )[i]
            center = (left + right) / 2
            xs = np.linspace(right, left, 6)
            ylabels = np.arange(.5, 2.6, .5) if i else np.arange(10, 61, 10)
            ys = offset - scale * ylabels
            with pdf.open() as grid:
                plot = grid.new_page(width=pw, height=ph)
                style = dict(color=GRID, width=.8, dashes="[2.96 1.28] 0", stroke_opacity=.2, lineJoin=1)
                plot.draw_rect((left, TOP, right, BOTTOM), color=None, fill=(1, 1, 1))
                for x in xs:
                    plot.draw_line((x, BOTTOM), (x, TOP), **style)
                for y in ys:
                    plot.draw_line((left, y), (right, y), **style)
                view = pdf.Rect(left, TOP, right, BOTTOM)
                page.show_pdf_page(view, grid, 0, clip=view)
            with pdf.open() as vector:
                source = vector.new_page(width=pw, height=SOURCE_HEIGHT)
                stream, endpoint = native_curves(data, panel)
                xref = vector.get_new_xref()
                vector.update_object(xref, "<<>>")
                vector.update_stream(xref, stream)
                source.set_contents(xref)
                clip = pdf.Rect(0, SOURCE_CUT, pw, source.rect.height)
                page.show_pdf_page(pdf.Rect(0, 10, pw, clip.height + 10), vector, 0, clip=clip)
            for x, mark in zip(xs, np.linspace(0, 1, 6), strict=True):
                page.draw_line((x, BOTTOM), (x, BOTTOM + 3.5), width=.8)
                label(f"{mark:.1f}", 15, x, 286.031799, align=.5)
            for y, mark in zip(ys, ylabels, strict=True):
                page.draw_line((left - 3.5, y), (left, y), width=.8)
                label(f"{mark:.1f}" if i else f"{mark:.0f}", 15, left - 7, y + 5.164063, align=1)
            page.draw_line((left, TOP), (left, BOTTOM), width=.8, lineCap=2)
            page.draw_line((left, BOTTOM), (right, BOTTOM), width=.8, lineCap=2)
            hi = offset - scale * endpoint["GUL"] / 1000
            lo = offset - scale * endpoint["Actual loss"] / 1000
            hh = hi + (7.765732 if i else 7.946365)
            ll = lo - (7.75885 if i else 7.952881)
            x = left + .985 * (right - left)
            stroke = dict(color=DARK_GRAY, width=1.5, lineCap=1, lineJoin=1, closePath=False)
            page.draw_bezier((x, ll), (x, (ll * 2 + hh) / 3), (x, (ll + hh * 2) / 3), (x, hh), **stroke)
            for y, cap in ((hh, 4.8), (ll, -4.8)):
                page.draw_polyline(((x - 2.4, y + cap), (x, y), (x + 2.4, y + cap)), **stroke)
            label("(b)" if i else "(a)", 20, panel_x, 31.035583, bold=True)
            label("Renter" if i else "Homeowner", 20, center, 35.834137, bold=True, align=.5)
            label("Annual loss per household ($K)", 18, ylabel_x, 252.127441, rotate=90)
            label("Aggregate exceedance probability", 18, center, 311.313049, align=.5)
            for key, y, color in (("GUL", hi, RGB_RED), ("Actual loss", lo, RGB_BLUE)):
                label(f"${endpoint[key] / 1000:.{i}f}K", 15, right + 5, y + 2.6953125, bold=True, color=color)
            legend_underlay(page, left)
            page.draw_rect((left + 5.5, 48, left + 119, 91), color=(.8, .8, .8), fill=(1, 1, 1), width=.6)
            for text, y, color in (("GUL", 62, RGB_RED), ("Actual loss", 81, RGB_BLUE)):
                page.draw_line((left + 10, y - 4), (left + 29, y - 4), color=color, width=1.8)
                label(text, 15, left + 34, y)
            difference = endpoint["GUL"] - endpoint["Actual loss"]
            pct = 100 * difference / endpoint["GUL"]
            notes = (f"Protection margin = {pct:.0f}% of GUL", f"Dollar difference = ${difference / 1000:.{i}f}K")
            cx = 287 + left - 57.934376
            half = max(metrics[1].text_length(text, fontsize=14) for text in notes) / 2 + 4
            page.draw_rect((cx - half, 45, cx + half, 81), color=(.4, .4, .4), fill=(1, 1, 1), width=.6)
            for text, y in zip(notes, (59, 75), strict=True):
                label(text, 14, cx, y, bold=True, align=.5, color=(.2, .2, .2))
        return page.get_pixmap(dpi=600, alpha=False).tobytes("png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", required=True, type=Path, help="Destination for the checked PNG")
    parser.add_argument("--series-dir", type=Path, help="Folder with the regenerated Figure 7 CSV")
    parser.add_argument("--font-dir", type=Path, help="Folder with times.ttf and timesbd.ttf (Windows Fonts by default)")
    args = parser.parse_args()
    output = args.figure_dir / OUT
    if output.exists() or output.is_symlink():
        parser.error(f"Refusing to overwrite: {output}")
    try:
        png = verified_png(args.series_dir, args.font_dir)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            stream.write(png)
        print(f"PASS {output}: exact accepted Figure 7 pixels (PNG only)")
    except (ImportError, KeyError, OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


def verified_png(series_dir: Path | None = None, font_dir: Path | None = None) -> bytes:
    """Build and validate the PNG before the dedicated or batch CLI writes it."""
    import pymupdf as pdf

    if pdf.VersionBind != "1.26.7":
        raise ValueError("Use PyMuPDF 1.26.7 from requirements-vector-figures.txt")
    if font_dir is None:
        windows = os.environ.get("WINDIR")
        if not windows:
            raise ValueError("Supply --font-dir with the local regular and bold Times New Roman fonts")
        font_dir = Path(windows) / "Fonts"
    fonts = tuple((font_dir / name).read_bytes() for name in ("times.ttf", "timesbd.ttf"))
    png = render_png(pdf, read_series(series_dir), fonts)
    expected = json.loads(REFERENCE.read_text(encoding="utf-8"))["figures"][OUT]
    verify_pixels(png, expected)
    return png


if __name__ == "__main__":
    main()

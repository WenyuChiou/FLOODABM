"""Reproduce revised WRR Figures 4, 6, and 7 from pinned study-archive PDFs."""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path

from PIL import Image

if __package__:
    from .verify_wrr_figures import REFERENCE
else:
    from verify_wrr_figures import REFERENCE


DIAGNOSTICS = "Fig04_flood_inudation.pdf"  # Original archive spelling.
CUMULATIVE = "Fig06_cumulative_loss.pdf"
FINANCE = "Fig07_finance_per_household.pdf"
COMPONENTS = {
    DIAGNOSTICS: "873ab5251246fb02903da2bab0a472d18dfbb85a68cd3ee2ddf4941bfe40e4f0",
    CUMULATIVE: "f1aff8c3cc7f18f0df1e757f9e2653b979555479cd0b2d294e31ddd6d149874d",
    FINANCE: "aedcd2169991b3ce099d63db7079aa32a4d4e1715c2e274f0fa8b442662d18d0",
}
INPUTS = {4: (DIAGNOSTICS,), 6: (CUMULATIVE, FINANCE), 7: (FINANCE,)}
FONT_HASHES = {
    "WCTimes": "931c5de5c70401d9324d5014c123802b4fb753000360ceb2f56c589403cd58c5",
    "WCTimesBold": "54fbe2c70af7c85a97bed0573227e3ccc4b2486012e3ca2a40c6bc77065846f5",
}
CUT = 589.5  # Split the review-era finance page at this PDF coordinate, in points.
RGB_RED = (0.835294, 0.152941, 0.156863)
RGB_BLUE = (0, 0.466667, 0.713725)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_components(directory: Path, figures: list[int]) -> dict[str, bytes]:
    """Validate all selected inputs before painting or writing anything."""
    loaded = {}
    for name in sorted({name for number in figures for name in INPUTS[number]}):
        data = (directory / name).read_bytes()
        if digest(data) != COMPONENTS[name]:
            raise ValueError(f"Archive component SHA-256 mismatch: {name}")
        loaded[name] = data
    return loaded


def spans(page):
    return [
        span
        for block in page.get_text("dict")["blocks"]
        for line in block.get("lines", [])
        for span in line["spans"]
    ]


def embedded_fonts(finance):
    fonts = {}
    for xref, ext, _, _, alias, _, _ in finance[0].get_fonts(full=True):
        if ext == "ttf" and alias in FONT_HASHES:
            data = finance.extract_font(xref)[3]
            if digest(data) != FONT_HASHES[alias]:
                raise ValueError(f"Embedded font SHA-256 mismatch: {alias}")
            fonts[alias] = data
    if fonts.keys() != FONT_HASHES.keys():
        raise ValueError("The finance component is missing the expected embedded fonts")
    return fonts


def render_sq1(pdf, cumulative, finance, bold: bytes) -> bytes:
    """Combine the cumulative panel pair with the four financial panels."""
    with pdf.open() as annual, pdf.open() as result:
        width = finance[0].rect.width
        page = annual.new_page(width=width, height=CUT)
        page.show_pdf_page(page.rect, finance, 0, clip=pdf.Rect(0, 0, width, CUT))
        original = spans(page)
        labels = []
        for old, new in zip(("(a)", "(b)", "(c)", "(d)"), ("(c)", "(d)", "(e)", "(f)")):
            match = [s for s in original if s["text"] == old and s["bbox"][1] < CUT]
            if len(match) != 1:
                raise ValueError(f"Expected one annual panel label {old}, found {len(match)}")
            s = match[0]
            page.add_redact_annot(pdf.Rect(s["bbox"]) + (-1, -1, 4, 1), fill=(1, 1, 1))
            labels.append((s["origin"], s["size"], new))
        page.apply_redactions(images=0, graphics=0, text=0)
        page.insert_font(fontname="WCTimesBoldSQ1", fontbuffer=bold)
        for origin, size, text in labels:
            page.insert_text(origin, text, fontsize=size, fontname="WCTimesBoldSQ1")

        top, gap, bottom = 6, 14, 8
        height = cumulative[0].rect.height * (width / cumulative[0].rect.width)
        financial_y = top + height + gap
        page = result.new_page(width=width, height=financial_y + CUT + bottom)
        page.show_pdf_page(pdf.Rect(0, top, width, top + height), cumulative, 0)
        page.show_pdf_page(pdf.Rect(0, financial_y, width, financial_y + CUT), annual, 0)
        return page.get_pixmap(dpi=300, alpha=False).tobytes("png")


def curve_paths(page):
    return [
        (d["items"], d["color"], d["width"], d["rect"])
        for d in page.get_drawings()
        if len(d["items"]) > 20 and d["color"] is not None
        and max(d["color"]) - min(d["color"]) > 0.3
    ]


def render_sq2(pdf, finance, fonts: dict[str, bytes]) -> bytes:
    """Crop the AEP pair and apply the submitted typography and margin labels."""
    clip = pdf.Rect(0, CUT, finance[0].rect.width, finance[0].rect.height)
    with pdf.open() as result:
        page = result.new_page(width=clip.width, height=clip.height + 30)
        page.show_pdf_page(pdf.Rect(0, 10, clip.width, clip.height + 10), finance, 0, clip=clip)
        curves = curve_paths(page)
        if len(curves) != 4:
            raise ValueError("Expected four source AEP curves")
        original = spans(page)
        boxes = [
            d["rect"] for d in page.get_drawings()
            if d["type"] == "f" and d["fill"] == (1.0, 1.0, 1.0)
            and 25 < d["rect"].width < 35 and 13 < d["rect"].height < 15
        ]
        if len(boxes) != 4:
            raise ValueError("Expected four original AEP endpoint-label boxes")
        for s in original:
            page.add_redact_annot(pdf.Rect(s["bbox"]) + (-0.2, -0.2, 0.2, 0.2), fill=False)
        page.apply_redactions(images=0, graphics=0, text=0)
        if page.get_text().strip():
            raise ValueError("Could not remove the previous AEP typography")
        for box in boxes:
            page.add_redact_annot(box + (-0.1, -0.1, 0.1, 0.1), fill=False)
        page.apply_redactions(images=0, graphics=1, text=0)

        page.insert_font(fontname="WCRegular", fontbuffer=fonts["WCTimes"])
        page.insert_font(fontname="WCBold", fontbuffer=fonts["WCTimesBold"])
        regular = pdf.Font(fontbuffer=fonts["WCTimes"])
        bold = pdf.Font(fontbuffer=fonts["WCTimesBold"])

        def put(text, size, origin, *, boldface=False, color=(0, 0, 0), align=0, rotate=0):
            font = bold if boldface else regular
            x, y = origin
            if not rotate:
                x -= font.text_length(text, fontsize=size) * align
            page.insert_text(
                (x, y), text, fontsize=size, fontname="WCBold" if boldface else "WCRegular",
                color=color, rotate=rotate,
            )

        for s in original:
            text = s["text"].replace("\u00a0", " ")
            rect = pdf.Rect(s["bbox"])
            x, y = s["origin"]
            left, right = (482.089386, 782.394348) if x > 425 else (57.934376, 358.23938)
            center = (left + right) / 2
            if text in ("GUL", "Actual loss") or text.startswith(("Tail gap", "Covered")):
                continue
            if text in ("(e)", "(f)"):
                put("(a)" if text == "(e)" else "(b)", 20, (x, y), boldface=True)
            elif text in ("Homeowner", "Renter"):
                put(text, 20, (center, y), boldface=True, align=0.5)
            elif text.startswith("Annual loss"):
                length = regular.text_length(text, fontsize=18)
                put(text, 18, (x - 11, (rect.y0 + rect.y1 + length) / 2), rotate=90)
            elif text.startswith("Aggregate exceedance"):
                put(text, 18, (center, y + 12), align=0.5)
            elif text.startswith("$"):
                colors = {"$51K": RGB_RED, "$2.2K": RGB_RED, "$28K": RGB_BLUE, "$1.2K": RGB_BLUE}
                put(text, 15, (right + 5, y), boldface=True, color=colors[text])
            elif rect.y0 > 265:
                put(text, 15, ((rect.x0 + rect.x1) / 2, y + 3), align=0.5)
            else:
                put(text, 15, (rect.x1, y + 1), align=1)

        for left, offset, amount, pct in [(57.934376, 0, "$23K", 44), (482.089386, 424.15501, "$1.0K", 45)]:
            legend = pdf.Rect(left + 5.5, 48, left + 119, 91)
            page.draw_rect(legend, color=(0.8, 0.8, 0.8), fill=(1, 1, 1), width=0.6)
            for text, y, color in [("GUL", 62, RGB_RED), ("Actual loss", 81, RGB_BLUE)]:
                page.draw_line((left + 10, y - 4), (left + 29, y - 4), color=color, width=1.8)
                put(text, 15, (left + 34, y))
            labels = [f"Protection margin = {pct}% of GUL", f"Dollar difference = {amount}"]
            center = 272 + offset + 15
            half = max(bold.text_length(text, fontsize=14) for text in labels) / 2 + 4
            box = pdf.Rect(center - half, 45, center + half, 81)
            page.draw_rect(box, color=(0.4, 0.4, 0.4), fill=(1, 1, 1), width=0.6)
            for text, y in zip(labels, (59, 75)):
                put(text, 14, (center, y), boldface=True, color=(0.2, 0.2, 0.2), align=0.5)

        if curve_paths(page) != curves:
            raise ValueError("An AEP vector curve changed during the typography update")
        return page.get_pixmap(dpi=600, alpha=False).tobytes("png")


def render(pdf, number: int, components: dict[str, bytes]) -> bytes:
    if number == 4:
        with pdf.open(stream=components[DIAGNOSTICS], filetype="pdf") as diagnostic:
            return diagnostic[0].get_pixmap(dpi=600, alpha=False).tobytes("png")
    with pdf.open(stream=components[FINANCE], filetype="pdf") as finance:
        fonts = embedded_fonts(finance)
        if number == 7:
            return render_sq2(pdf, finance, fonts)
        with pdf.open(stream=components[CUMULATIVE], filetype="pdf") as cumulative:
            return render_sq1(pdf, cumulative, finance, fonts["WCTimesBold"])


def verify_pixels(png: bytes, expected: dict) -> None:
    with Image.open(BytesIO(png), formats=["PNG"]) as image:
        rgba = image.convert("RGBA")
        actual = {"width": rgba.width, "height": rgba.height, "rgba_sha256": digest(rgba.tobytes())}
    if actual != expected:
        raise ValueError("Rendered pixels or dimensions differ from the accepted WRR reference")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", type=Path, required=True, help="Directory containing the archived PDFs")
    parser.add_argument("--figure-dir", type=Path, required=True, help="PNG output directory; never overwrites")
    parser.add_argument("--figures", nargs="+", type=int, choices=sorted(INPUTS), default=sorted(INPUTS))
    args = parser.parse_args()
    figures = sorted(set(args.figures))
    targets = {number: args.figure_dir / f"Figure_{number:02d}.png" for number in figures}
    try:
        for target in targets.values():
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"Refusing to overwrite: {target}")
        components = read_components(args.components, figures)
        from pymupdf import VersionBind, VersionFitz
        import pymupdf as pdf

        if (VersionBind, VersionFitz) != ("1.26.7", "1.26.12"):
            raise ValueError("Install requirements-vector-figures.txt for the pinned renderer")
        manifest = json.loads(REFERENCE.read_text(encoding="utf-8"))["figures"]
        outputs = {}
        for number, target in targets.items():
            png = render(pdf, number, components)
            try:
                verify_pixels(png, manifest[target.name])
            except ValueError as error:
                raise ValueError(f"{target.name}: {error}") from error
            outputs[target] = png
        args.figure_dir.mkdir(parents=True, exist_ok=True)
        for target, png in outputs.items():
            with target.open("xb") as file:
                file.write(png)
            print(f"PASS {target.name}: accepted pixels reproduced from pinned PDF components")
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()

"""Reproduce Figure 10 from its archived ArcGIS project and rebuilt CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
from io import StringIO
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


PROJECT_SHA = "1a6b4af9f708773831dba833b52e26cb4fdb71619ab481cd1efdf04f81d88d1f"
SERIES_SHA = "a5fd14743512deaa4fea828413376007a2c8948d8ca75ebbc1201b6949471be3"
LAYOUT_INPUT = "Fig10_spatial_fi.png"
OUTPUT = "Figure_10.png"
PANELS = {
    f"Figure10 {prefix} {year}": (group, year, f"{prefix}BIV{year % 100}")
    for prefix, group in (("H", "homeowner"), ("R", "renter"))
    for year in (2021, 2022, 2023)
}


def verified_input(path: Path, digest: str) -> bytes:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"{path.name} does not match the revised Figure 10 input")
    return data


def expected_classes(series: bytes) -> dict[tuple[str, int], dict[str, float]]:
    reader = csv.DictReader(StringIO(series.decode("utf-8-sig"), newline=""))
    panels: dict[tuple[str, int], dict[str, float]] = {}
    for row in reader:
        group = row["group"]
        geoid = row["tract_geoid"]
        if group not in ("homeowner", "renter") or len(geoid) != 11 or not geoid.isascii() or not geoid.isdecimal():
            raise ValueError("Expected homeowner/renter records with 11-digit tract GEOIDs")
        for year in (2021, 2022, 2023):
            values = panels.setdefault((group, year), {})
            cell = float(row[f"bivariate_{year}"])
            if geoid in values or cell not in range(1, 10):
                raise ValueError("Expected one bivariate class 1–9 per tenure, year and tract")
            values[geoid] = cell
    if len(panels) != 6 or any(len(values) != 27 for values in panels.values()):
        raise ValueError("Expected 27 tracts in each of the six panels")
    if any(set(values) != set(panels["homeowner", 2021]) for values in panels.values()):
        raise ValueError("All panels must contain the same tract GEOIDs")
    return panels


def verify_panel(rows, expected: dict[str, float], panel: str) -> None:
    seen = set()
    for geoid, cell in rows:
        if geoid in seen or geoid not in expected or cell != expected[geoid]:
            raise ValueError(f"{panel}: a mapped class does not match the revised data")
        seen.add(geoid)
    if seen != expected.keys():
        raise ValueError(f"{panel}: incomplete set of tracts")


def export_native_layout(project_path: Path, series_path: Path, layout_png: Path) -> None:
    original = verified_input(project_path, PROJECT_SHA)
    expected = expected_classes(verified_input(series_path, SERIES_SHA))
    import arcpy

    project = arcpy.mp.ArcGISProject(str(project_path))
    info = arcpy.GetInstallInfo()
    layouts = project.listLayouts("Layout17")
    if len(layouts) != 1:
        raise ValueError("Expected one approved Layout17")
    layout = layouts[0]
    frames = [frame for frame in layout.listElements("MAPFRAME_ELEMENT") if frame.visible]
    if len(frames) != len(PANELS):
        raise ValueError("Expected six visible map frames")
    seen = set()
    for frame in frames:
        layers = [layer for layer in frame.map.listLayers() if layer.visible]
        if any(layer.isBroken or layer.isWebLayer or layer.isBasemapLayer for layer in layers):
            raise ValueError(f"{frame.name}: missing or non-local map source")
        mapped = [layer for layer in layers if layer.name in PANELS]
        if len(mapped) != 1 or mapped[0].name in seen:
            raise ValueError(f"{frame.name}: unexpected or repeated result layer")
        layer = mapped[0]
        group, year, column = PANELS[layer.name]
        if list(layer.symbology.renderer.fields) != [column]:
            raise ValueError(f"{layer.name}: unexpected renderer field")
        with arcpy.da.SearchCursor(layer, ["GEOID", column]) as rows:
            verify_panel(rows, expected[group, year], layer.name)
        seen.add(layer.name)
    if seen != PANELS.keys():
        raise ValueError("A revised result panel is missing")
    print(f"ArcGIS Pro {info['Version']}: verified all 162 plotted bivariate classes", flush=True)

    # The archived .aprx precedes the 45 px downward legend-marker shift at 300 dpi.
    markers = [el for el in layout.listElements("GRAPHIC_ELEMENT") if el.name == "Patch 4"]
    if len(markers) != 1 or abs(markers[0].elementPositionY - 14.93) > 1e-6:
        raise ValueError("Unexpected flood-prone legend-marker position")
    markers[0].elementPositionY = 14.78
    layout.exportToPNG(str(layout_png), resolution=300, clip_to_elements=False)
    if project_path.read_bytes() != original:
        raise ValueError("The source project changed during export")


def render_project(project: Path, series: Path, arcgis_python: Path, output_dir: Path) -> None:
    if __package__:
        from .render_wrr_supporting_raster import render_main_map
    else:
        from render_wrr_supporting_raster import render_main_map

    target = output_dir / OUTPUT
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {target}")
    verified_input(project, PROJECT_SHA)
    verified_input(series, SERIES_SHA)
    with TemporaryDirectory(prefix="wrr-figure10-") as folder:
        layout_png = Path(folder) / LAYOUT_INPUT
        argv = [
            str(arcgis_python.resolve(strict=True)), "-X", "utf8", str(Path(__file__).resolve()),
            "--export-full-page", "--project", str(project.resolve()),
            "--series", str(series.resolve()), "--page-png", str(layout_png),
        ]
        subprocess.run(argv, check=True, timeout=180)
        render_main_map(Path(folder), output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True, help="Archived Figure10_ready.aprx")
    parser.add_argument("--series", type=Path, required=True, help="Verified Figure10_spatial_fi.csv")
    parser.add_argument("--arcgis-python", type=Path, help="Executable from an installed ArcGIS Pro environment")
    parser.add_argument("--figure-dir", type=Path, help="Directory for Figure_10.png")
    parser.add_argument("--export-full-page", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--page-png", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.export_full_page:
            if not args.page_png or args.page_png.name != LAYOUT_INPUT or args.page_png.exists() or args.page_png.is_symlink():
                parser.error("Native export requires a new full-page PNG path")
            with TemporaryDirectory(prefix="wrr-figure10-native-") as folder:
                scratch_png = Path(folder) / LAYOUT_INPUT
                export_native_layout(args.project, args.series, scratch_png)
                with scratch_png.open("rb") as source, args.page_png.open("xb") as destination:
                    shutil.copyfileobj(source, destination)
        elif args.figure_dir is None or args.arcgis_python is None:
            parser.error("--arcgis-python and --figure-dir are required")
        else:
            render_project(args.project, args.series, args.arcgis_python, args.figure_dir)
    except (ImportError, OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Figure 10: {error}\n")


if __name__ == "__main__":
    main()

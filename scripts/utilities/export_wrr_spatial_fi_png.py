"""Reproduce revised Figure 10 or Supporting Figure S7 from ArcGIS and rebuilt data."""

from __future__ import annotations

import argparse
import csv
import hashlib
from io import StringIO
import math
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


PROJECT_SHA = "1a6b4af9f708773831dba833b52e26cb4fdb71619ab481cd1efdf04f81d88d1f"
SERIES_SHA = "a5fd14743512deaa4fea828413376007a2c8948d8ca75ebbc1201b6949471be3"
LAYOUT_INPUT = "Fig10_spatial_fi.png"
OUTPUT = "Figure_10.png"
S7_PROJECT_SHA = "dd6d3bf1932ec3a6bd8ce5434014b80c66fa2f91f29f62a9ef46e81094ccbc27"
S7_SERIES_SHA = "1f172276a493d186764c5acc3e390b249d3262f834014b7f14c96e0ec337ba89"
S7_LAYOUT_INPUT = "native_population_layout.png"
S7_OUTPUT = "FigS07_population_change.png"
PANELS = {
    f"Figure10 {prefix} {year}": (group, year, f"{prefix}BIV{year % 100}")
    for prefix, group in (("H", "homeowner"), ("R", "renter"))
    for year in (2021, 2022, 2023)
}
POPULATION_PANELS = {
    f"Population {prefix} {year}": (group, year, f"{prefix}_dpct_{year}")
    for prefix, group in (("owner", "homeowner"), ("renter", "renter"))
    for year in (2021, 2022, 2023)
}


def verified_input(path: Path, digest: str, figure: str = "10") -> bytes:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"{path.name} does not match the revised Figure {figure} input")
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


def expected_population(series: bytes) -> dict[tuple[str, int], dict[str, str]]:
    reader = csv.DictReader(StringIO(series.decode("ascii"), newline=""))
    panels = {(group, year): {} for group, year, _ in POPULATION_PANELS.values()}
    seen = set()
    for row in reader:
        geoid = row["tract_geoid"]
        if len(geoid) != 11 or not geoid.isascii() or not geoid.isdecimal() or geoid in seen:
            raise ValueError("Expected one row per 11-digit tract GEOID")
        seen.add(geoid)
        for group, year, column in POPULATION_PANELS.values():
            raw = row[column]
            value = float(raw)
            if not math.isfinite(value) or f"{value:.9f}" != raw:
                raise ValueError(f"{column}: expected a finite population change at nine decimals")
            panels[group, year][geoid] = raw
    if len(seen) != 27:
        raise ValueError("Expected 27 tracts in each of the six panels")
    return panels


def verify_panel(rows, expected, panel: str, metric: str = "class") -> None:
    seen = set()
    for geoid, cell in rows:
        if geoid in seen or geoid not in expected or cell != expected[geoid]:
            raise ValueError(f"{panel}: a mapped {metric} does not match the revised data")
        seen.add(geoid)
    if seen != expected.keys():
        raise ValueError(f"{panel}: incomplete set of tracts")


def export_native_layout(project_path: Path, series_path: Path, layout_png: Path, figure: str = "10") -> None:
    population = figure == "S7"
    original = verified_input(project_path, S7_PROJECT_SHA if population else PROJECT_SHA, figure)
    series = verified_input(series_path, S7_SERIES_SHA if population else SERIES_SHA, figure)
    expected = expected_population(series) if population else expected_classes(series)
    panels = POPULATION_PANELS if population else PANELS
    layout_name = "Layout171" if population else "Layout17"
    import arcpy

    project = arcpy.mp.ArcGISProject(str(project_path))
    info = arcpy.GetInstallInfo()
    layouts = project.listLayouts(layout_name)
    if len(layouts) != 1:
        raise ValueError(f"Expected one approved {layout_name}")
    layout = layouts[0]
    frames = [frame for frame in layout.listElements("MAPFRAME_ELEMENT") if frame.visible]
    if len(frames) != len(panels):
        raise ValueError("Expected six visible map frames")
    seen = set()
    for frame in frames:
        layers = [layer for layer in frame.map.listLayers() if layer.visible]
        if any(layer.isBroken or layer.isWebLayer or layer.isBasemapLayer for layer in layers):
            raise ValueError(f"{frame.name}: missing or non-local map source")
        mapped = [layer for layer in layers if layer.name in panels]
        if len(mapped) != 1 or mapped[0].name in seen:
            raise ValueError(f"{frame.name}: unexpected or repeated result layer")
        layer = mapped[0]
        group, year, column = panels[layer.name]
        binding = (
            [layer.getDefinition("V3").renderer.field]
            if population else list(layer.symbology.renderer.fields)
        )
        if binding != [column]:
            raise ValueError(f"{layer.name}: unexpected renderer field")
        with arcpy.da.SearchCursor(layer, ["GEOID", column]) as rows:
            if population:
                numbers = ((geoid, None if cell is None else f"{cell:.9f}") for geoid, cell in rows)
                verify_panel(numbers, expected[group, year], layer.name, "population value")
            else:
                verify_panel(rows, expected[group, year], layer.name)
        seen.add(layer.name)
    if seen != panels.keys():
        raise ValueError("A revised result panel is missing")
    metric = "population changes at nine decimals" if population else "bivariate classes"
    print(f"ArcGIS Pro {info['Version']}: verified all 162 plotted {metric}", flush=True)

    # The archived .aprx precedes the 45 px downward legend-marker shift at 300 dpi.
    # The archived S7 project already includes its final legend positions.
    if not population:
        markers = [el for el in layout.listElements("GRAPHIC_ELEMENT") if el.name == "Patch 4"]
        if len(markers) != 1 or abs(markers[0].elementPositionY - 14.93) > 1e-6:
            raise ValueError("Unexpected flood-prone legend-marker position")
        markers[0].elementPositionY = 14.78
    layout.exportToPNG(str(layout_png), resolution=300, clip_to_elements=False)
    if project_path.read_bytes() != original:
        raise ValueError("The source project changed during export")


def render_project(project: Path, series: Path, arcgis_python: Path, output_dir: Path, figure: str = "10") -> None:
    if __package__:
        from .render_wrr_supporting_raster import render_main_map, render_native_population_map
    else:
        from render_wrr_supporting_raster import render_main_map, render_native_population_map

    population = figure == "S7"
    target = output_dir / (S7_OUTPUT if population else OUTPUT)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {target}")
    verified_input(project, S7_PROJECT_SHA if population else PROJECT_SHA, figure)
    verified_input(series, S7_SERIES_SHA if population else SERIES_SHA, figure)
    with TemporaryDirectory(prefix="wrr-spatial-map-") as folder:
        layout_png = Path(folder) / (S7_LAYOUT_INPUT if population else LAYOUT_INPUT)
        argv = [
            str(arcgis_python.resolve(strict=True)), "-X", "utf8", str(Path(__file__).resolve()),
            "--export-full-page", "--figure", figure, "--project", str(project.resolve()),
            "--series", str(series.resolve()), "--page-png", str(layout_png),
        ]
        subprocess.run(argv, check=True, timeout=180)
        if population:
            render_native_population_map(layout_png, target)
        else:
            render_main_map(Path(folder), output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=("10", "S7"), default="10", help="Map to export (default: 10)")
    parser.add_argument("--project", type=Path, required=True, help="Archived editable ArcGIS project")
    parser.add_argument("--series", type=Path, required=True, help="Verified rebuilt CSV for this map")
    parser.add_argument("--arcgis-python", type=Path, help="Executable from an installed ArcGIS Pro environment")
    parser.add_argument("--figure-dir", type=Path, help="Directory for the final PNG")
    parser.add_argument("--export-full-page", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--page-png", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.export_full_page:
            layout_input = S7_LAYOUT_INPUT if args.figure == "S7" else LAYOUT_INPUT
            if not args.page_png or args.page_png.name != layout_input or args.page_png.exists() or args.page_png.is_symlink():
                parser.error("Native export requires a new full-page PNG path")
            with TemporaryDirectory(prefix="wrr-spatial-map-native-") as folder:
                scratch_png = Path(folder) / layout_input
                export_native_layout(args.project, args.series, scratch_png, args.figure)
                with scratch_png.open("rb") as source, args.page_png.open("xb") as destination:
                    shutil.copyfileobj(source, destination)
        elif args.figure_dir is None or args.arcgis_python is None:
            parser.error("--arcgis-python and --figure-dir are required")
        else:
            render_project(args.project, args.series, args.arcgis_python, args.figure_dir, args.figure)
    except (ImportError, OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Figure {args.figure}: {error}\n")


if __name__ == "__main__":
    main()

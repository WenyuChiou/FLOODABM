"""Reproduce the revised Figure 10 and Supporting Figure S7 PNGs from public GIS data."""

from __future__ import annotations

import argparse
from io import BytesIO
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from geopandas import GeoDataFrame
    from shapely import Geometry
    from shapely.geometry import Polygon

if __package__:
    from . import export_wrr_spatial_fi_png as spatial
    from . import render_wrr_supporting_raster as raster
    from .wrr_public_sources import fema_geometry, public_input, read_limited, verify_digest
else:
    import export_wrr_spatial_fi_png as spatial
    import render_wrr_supporting_raster as raster
    from wrr_public_sources import fema_geometry, public_input, read_limited, verify_digest


TRACT_ARCHIVE_NAME = "tl_2022_34_tract.zip"
TRACT_ARCHIVE_SHA = "bec4d351275e6cc3c7613fd845303a771dd9094ba9e20154394d5ece955d945f"
TRACT_GEOMETRY_SHA = "5e5d531dda32bd3a8b25c4a349b61cd497e36eb7bd4f703912ae48995c900cbf"
CENSUS_BASE = "https://www2.census.gov/geo/tiger/TIGER2022"
LAYOUT_PINS = (
    "5d22a43820352827b4ed6d32bd1b56071b2cbd334d3e51d05c539970a53f4f04",
    "4cbec7e5729b528ba08ff33146624f5154a8f256c56d17991088e04a25aed943",
)
BLANK_PROJECT_SHA = "650d45b59cf8dbbe009758702983a063e686131aa2f41f6e928ffcb4091bf44d"
COUNTY_WATER_PINS = {
    "LINEARWATER": {
        "34013": "bc14fe94d970433c8527c05e71e8298f530d9174432126f5e8a9db9ab436a00c",
        "34027": "a994c17001bfe58f9f2a15cc2da6d474e27f78573903b4b39523224aaf45f7d8",
        "34031": "e978416d81bcc0fff3802f689a9e4a59ff81e3bc8df9938f3ac5cf88b8f0f13f",
    },
    "AREAWATER": {
        "34013": "aacbb6cd1253e67e296c2b593f8186d823ccd5703be872e6259991a69467dde3",
        "34027": "d0b9ede08a66fdeb6c4193360d54ecc7bd89fa72fe7b46dd7b0a740495a837c0",
        "34031": "626c6e5d7cf5c449fa7f5863d0ae3337b8e3ea680e16b5673a06b6551522f8a2",
    },
}
# These are public TIGER feature IDs selected by the revised layout.
WATER_IDS = {
    "LINEARWATER": frozenset({
        "1106092963031", "1106092994550", "110457972725", "1104469409412",
        "1104469409654", "110457972729", "110457972828", "110457972822",
        "110457972826", "110716402465", "110716402461", "1105606597266",
        "1106092974452", "1106092963033", "110457972720", "110457972736",
        "1106092994633", "1106092994634", "110457972738", "1106092994632",
        "1106092962232", "1106092961902", "1106092961903",
    }),
    "AREAWATER": frozenset({
        "1106087133783", "1105577465050", "1104469409674", "1104486330139",
        "1105577465049", "1106087133782", "1106087092040",
    }),
}
SERIES_NAMES = ("Figure10_spatial_fi.csv", "FigureS7_population_change.csv")


def load_layout(path: Path, digest: str) -> bytes:
    """Allow only the two verified layout-only files, regardless of filename."""
    data = read_limited(path, 4_000_000)
    verify_digest(data, digest, path.name)
    return data


def population_tract_ids(fi: bytes, population: bytes) -> tuple[str, ...]:
    """Validate the two plot schemas and require the same 27 tract IDs."""
    classes = spatial.expected_classes(fi)
    changes = spatial.expected_population(population)
    fi_ids = set(classes["homeowner", 2021])
    if any(fi_ids != set(values) for values in changes.values()):
        raise ValueError("Figure 10 and S7 series contain different tracts")
    return tuple(sorted(fi_ids))


def public_water(cache: Path | None, category: str) -> GeoDataFrame:
    """Load only the selected public river features from pinned county archives."""
    import geopandas as gpd
    import pandas as pd

    field = "LINEARID" if category == "LINEARWATER" else "HYDROID"
    wanted = WATER_IDS[category]
    parts = []
    for county, digest in COUNTY_WATER_PINS[category].items():
        name = f"tl_2022_{county}_{category.lower()}.zip"
        data = public_input(cache, name, f"{CENSUS_BASE}/{category}/{name}", 10_000_000)
        verify_digest(data, digest, name)
        table = gpd.read_file(BytesIO(data), columns=[field, "FULLNAME"]).to_crs(4326)
        parts.append(table.loc[table[field].isin(wanted)].copy())
    public = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=4326)
    if set(public[field]) != wanted:
        raise ValueError(f"{category}: public feature IDs differ")
    return public


def polygon_parts(geometry: Geometry) -> Iterator[Polygon]:
    """Yield nonempty polygons, dropping boundary-only overlay fragments."""
    if geometry.geom_type == "Polygon" and not geometry.is_empty:
        yield geometry
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from polygon_parts(part)


def write_public_geometry(work: Path, cache: Path | None, tracts: tuple[str, ...]) -> None:
    """Write the verified public FEMA, line and polygon layers into scratch."""
    import geopandas as gpd
    import shapely

    fema = fema_geometry(cache, tracts)
    fema.to_file(work / "public27.shp", index=False)
    lines = public_water(cache, "LINEARWATER")
    if len(lines) != 36:
        raise ValueError("Expected the 36 public river source rows")
    # ArcGIS handles the line clip using its native coordinate tolerance.
    lines.to_file(work / "publiclines.shp", index=False)
    area = public_water(cache, "AREAWATER")
    grid = 1.0e-8
    mask = shapely.union_all([shapely.make_valid(geom) for geom in fema.geometry], grid_size=grid)
    rows = []
    for gid, group in area.groupby("HYDROID", sort=True):
        geometry = shapely.union_all(
            [shapely.make_valid(geom) for geom in group.geometry], grid_size=grid
        )
        intersection = shapely.intersection(geometry, mask, grid_size=grid)
        polygons = shapely.union_all(list(polygon_parts(intersection)), grid_size=grid)
        if polygons.is_empty:
            raise ValueError(f"Empty public river polygon {gid}")
        rows.append({"HYDROID": gid, "FULLNAME": group.FULLNAME.iloc[0], "geometry": polygons})
    if len(rows) != 7:
        raise ValueError("Expected seven clipped river polygons")
    gpd.GeoDataFrame(rows, crs=4326).to_file(work / "public_area.shp", index=False)


def write_result_layers(
    work: Path, cache: Path | None, tracts: tuple[str, ...], fi: bytes, population: bytes
) -> None:
    """Write the 27 public tract polygons and the six bivariate DBF fields."""
    import geopandas as gpd
    from zipfile import ZipFile

    data = public_input(
        cache, TRACT_ARCHIVE_NAME, f"{CENSUS_BASE}/TRACT/{TRACT_ARCHIVE_NAME}", 20_000_000
    )
    verify_digest(data, TRACT_ARCHIVE_SHA, TRACT_ARCHIVE_NAME)
    with ZipFile(BytesIO(data)) as archive:
        verify_digest(archive.read("tl_2022_34_tract.shp"), TRACT_GEOMETRY_SHA, "Census shapefile")
    frame = gpd.read_file(BytesIO(data), columns=["GEOID"])
    # The source occurrence order controls edge antialiasing in the accepted map.
    frame = frame.loc[frame.GEOID.isin(tracts)].reset_index(drop=True)
    if len(frame) != 27 or set(frame.GEOID) != set(tracts) or frame.crs.to_epsg() != 4269:
        raise ValueError("Unexpected geometry or CRS for the revised 27 study tracts")
    if not frame.geometry.is_valid.all():
        raise ValueError("A selected Census tract has invalid geometry")
    out = work / "series"
    out.mkdir()
    (out / SERIES_NAMES[0]).write_bytes(fi)
    (out / SERIES_NAMES[1]).write_bytes(population)
    frame.to_file(out / "public_27.shp", index=False)
    classes = spatial.expected_classes(fi)
    for group, year, column in spatial.PANELS.values():
        frame[column] = frame.GEOID.map(classes[group, year]).astype("float64")
    frame.to_file(out / "Fig10_FI.shp", index=False)


def publish_pngs(verified: Path, output: Path) -> None:
    """Publish two completed files without overwriting an existing output."""
    names = (spatial.OUTPUT, spatial.S7_OUTPUT)
    # Stage beside the destination so Windows can rename the complete pair on
    # one filesystem. ArcGIS Pro is only available on Windows; the POSIX
    # fallback also supports testing the publishing gate without ArcGIS.
    with TemporaryDirectory(prefix=".wrr-publish-", dir=output.parent) as temp:
        ready = Path(temp) / "ready"
        ready.mkdir()
        for name in names:
            with (verified / name).open("rb") as src, (ready / name).open("xb") as dst:
                shutil.copyfileobj(src, dst)
        if os.name == "nt":
            # Windows rename fails if the destination already exists.
            ready.rename(output)
        else:
            output.mkdir(exist_ok=False)
            try:
                for name in names:
                    os.link(ready / name, output / name)
            except BaseException:
                for name in names:
                    target = output / name
                    if target.exists() and not target.is_symlink() and target.samefile(ready / name):
                        target.unlink()
                output.rmdir()
                raise
    for name in names:
        print(f"PASS {output / name}: exact revised PNG bytes", flush=True)


def finish_pngs(work: Path, output: Path) -> None:
    """Verify both final images before publishing the output pair."""
    first = raster.finish_spatial_fi_map((work / spatial.LAYOUT_INPUT).read_bytes())
    verify_digest(first, raster.FIGURE10_FINAL_SHA, spatial.OUTPUT)
    verified = work / "verified"
    verified.mkdir()
    (verified / spatial.OUTPUT).write_bytes(first)
    raster.render_native_population_map(work / spatial.S7_LAYOUT_INPUT, verified / spatial.S7_OUTPUT)
    publish_pngs(verified, output)


def run(args: argparse.Namespace) -> None:
    """Validate explicit inputs, rebuild public layers, and run ArcGIS separately."""
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if not args.output_dir.parent.is_dir():
        raise ValueError("Output directory's parent must already exist")
    executable = args.arcgis_python.resolve(strict=True)
    blank = args.blank_project.resolve(strict=True)
    if not executable.is_file() or not blank.is_file():
        raise ValueError("Provide the ArcGIS Python executable and an ArcGIS blank project file")
    layouts = [load_layout(p, want) for p, want in zip(
        (args.figure10_layout, args.s7_layout), LAYOUT_PINS, strict=True
    )]
    blank_bytes = read_limited(blank, 4_000_000)
    verify_digest(blank_bytes, BLANK_PROJECT_SHA, "ArcGIS 3.2 stock Blank.aprx")
    fi, population = (read_limited(args.series_dir / name, 100_000) for name in SERIES_NAMES)
    verify_digest(fi, spatial.SERIES_SHA, SERIES_NAMES[0])
    verify_digest(population, spatial.S7_SERIES_SHA, SERIES_NAMES[1])
    tracts = population_tract_ids(fi, population)
    with TemporaryDirectory(prefix="wrr-public-maps-", dir=args.scratch_root) as scratch:
        work = Path(scratch)
        write_result_layers(work, args.public_source_dir, tracts, fi, population)
        write_public_geometry(work, args.public_source_dir, tracts)
        local_layouts = [work / name for name in ("Figure10.pagx", "FigureS7.pagx")]
        for path, data in zip(local_layouts, layouts, strict=True):
            path.write_bytes(data)
        local_blank = work / "Blank.aprx"
        local_blank.write_bytes(blank_bytes)
        worker = Path(__file__).resolve().with_name("_wrr_public_map_arcgis.py")
        command = [executable, "-X", "utf8", worker, work, *local_layouts, local_blank]
        subprocess.run(command, check=True, timeout=240)
        finish_pngs(work, args.output_dir)


def main() -> None:
    """Command-line entrypoint for both exact revised spatial figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series-dir", type=Path, required=True, help="Verified rebuilt map CSVs")
    parser.add_argument("--figure10-layout", type=Path, required=True, help="Pinned Figure 10 PAGX")
    parser.add_argument("--s7-layout", type=Path, required=True, help="Pinned Supporting Figure S7 PAGX")
    parser.add_argument("--arcgis-python", type=Path, required=True, help="ArcGIS Pro Python executable")
    parser.add_argument("--blank-project", type=Path, required=True, help="Pinned stock ArcGIS Pro 3.2 Blank.aprx")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for the two PNGs")
    parser.add_argument("--public-source-dir", type=Path, help="Read pinned downloads from an offline cache")
    parser.add_argument("--scratch-root", type=Path, help="Existing directory for disposable intermediate files")
    args = parser.parse_args()
    try:
        run(args)
    except (ImportError, KeyError, OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Revised spatial maps: {error}\n")


if __name__ == "__main__":
    main()

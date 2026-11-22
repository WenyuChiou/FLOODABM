"""Build the two WRR ArcGIS page exports from public feature data and PAGX files.

This worker is launched with the Python 3.9 environment bundled with ArcGIS
Pro 3.2.  Its command-line interface is intentionally positional because the
portable caller supplies and validates all four paths.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import shutil
import sys
from tempfile import mkdtemp, TemporaryDirectory
from typing import Union

import arcpy

from export_wrr_spatial_fi_png import (
    PANELS,
    POPULATION_PANELS,
    expected_classes,
    expected_population,
    verify_panel,
)


FIGURE10_PAGX_SHA256 = "5d22a43820352827b4ed6d32bd1b56071b2cbd334d3e51d05c539970a53f4f04"
S7_PAGX_SHA256 = "4cbec7e5729b528ba08ff33146624f5154a8f256c56d17991088e04a25aed943"
BLANK_PROJECT_SHA256 = "650d45b59cf8dbbe009758702983a063e686131aa2f41f6e928ffcb4091bf44d"
FIGURE10_PAGE = "Fig10_spatial_fi.png"
S7_PAGE = "native_population_layout.png"
MARKER_IDS = (
    "34013021601",
    "34013021802",
    "34027040101",
    "34027040102",
    "34027040200",
    "34027040803",
    "34027040805",
    "34031154006",
    "34031223802",
    "34031246003",
    "34031246300",
)
FIG10_SIDECARS = (".shp", ".shx", ".dbf", ".prj", ".cpg")


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise ValueError(f"Missing {label}: {path}")
    return path.resolve()


def _pinned_pagx(path: Path, expected: str, label: str) -> Path:
    path = _require_file(path, label)
    body = path.read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    if digest != expected:
        raise ValueError(f"{label} SHA-256 changed: {digest}")
    forbidden = (b"onedrive", b".codex", b"/scratch/", b"token=", b"://", b"homefs")
    if any(fragment in body.lower() for fragment in forbidden):
        raise ValueError(f"{label} contains an external or private source reference")
    return path


def _count(dataset: Union[Path, str]) -> int:
    return int(arcpy.management.GetCount(str(dataset))[0])


def _factory_code(dataset: Union[Path, str]) -> int:
    return int(arcpy.Describe(str(dataset)).spatialReference.factoryCode)


def _prepare_native_clip(root: Path, build: Path) -> Path:
    """Clip public Census water lines using ArcGIS' unmodified defaults."""
    public_tracts = _require_file(root / "public27.shp", "public tract shapefile")
    public_lines = _require_file(root / "publiclines.shp", "public river-line shapefile")
    public_area = _require_file(root / "public_area.shp", "public river-area shapefile")
    if (_count(public_tracts), _count(public_lines), _count(public_area)) != (27, 36, 7):
        raise ValueError("Expected 27 public tracts, 36 public river rows, and 7 public river polygons")
    if (_factory_code(public_tracts), _factory_code(public_lines), _factory_code(public_area)) != (4326, 4326, 4326):
        raise ValueError("The public map feature inputs must use WGS 1984")

    native_gdb = build / "native.gdb"
    arcpy.management.CreateFileGDB(str(build), native_gdb.name)
    tracts = Path(arcpy.management.CopyFeatures(str(public_tracts), str(native_gdb / "tracts"))[0])
    lines = Path(arcpy.management.CopyFeatures(str(public_lines), str(native_gdb / "lines"))[0])

    # Deliberately use the 27 separate tract polygons and ArcGIS' default XY
    # tolerance.  Dissolving the mask or supplying a tolerance changes the
    # native geometry that the approved layouts use.
    clipped = Path(arcpy.analysis.Clip(str(lines), str(tracts), str(native_gdb / "river"))[0])
    if _count(clipped) != 36:
        raise ValueError("The default ArcGIS clip did not preserve the 36 selected river rows")
    spatial_reference = arcpy.Describe(str(clipped)).spatialReference
    if int(spatial_reference.factoryCode) != 4326:
        raise ValueError("The clipped river must retain its WGS 1984 spatial reference")
    print(
        "NATIVE_DEFAULT_CLIP",
        _count(clipped),
        spatial_reference.XYTolerance,
        spatial_reference.XYResolution,
        flush=True,
    )
    return clipped


def _load_markers(root: Path) -> dict[str, object]:
    marker_set = set(MARKER_IDS)
    found = {}
    with arcpy.da.SearchCursor(str(root / "public27.shp"), ["Tract", "SHAPE@"]) as rows:
        for geoid, geometry in rows:
            if geoid in marker_set:
                if geoid in found:
                    raise ValueError(f"Duplicate public marker tract: {geoid}")
                found[geoid] = geometry
    if set(found) != marker_set:
        raise ValueError("The public tract layer does not contain all 11 marker tracts")
    return found


def _create_base_gdb(root: Path, local: Path, clipped_river: Path, markers: dict[str, object]) -> Path:
    gdb = local / "floodmap.gdb"
    arcpy.management.CreateFileGDB(str(local), gdb.name)
    river = Path(arcpy.management.CopyFeatures(str(clipped_river), str(gdb / "river"))[0])
    area = Path(
        arcpy.management.CopyFeatures(str(root / "public_area.shp"), str(gdb / "passaic_river_polygon"))[0]
    )
    if (_count(river), _count(area)) != (36, 7):
        raise ValueError("The local river layers have unexpected feature counts")

    # Inherit the native clipped river spatial reference, including its XY
    # resolution/tolerance, instead of constructing a bare EPSG:4326 object.
    river_reference = arcpy.Describe(str(river)).spatialReference
    marker_fc = Path(
        arcpy.management.CreateFeatureclass(
            str(gdb), "FD_2023", "POLYGON", spatial_reference=river_reference
        )[0]
    )
    arcpy.management.AddField(str(marker_fc), "CensusTrac", "TEXT", field_length=11)
    expected_order = []
    with arcpy.da.InsertCursor(str(marker_fc), ["CensusTrac", "SHAPE@"]) as rows:
        for geoid in MARKER_IDS:
            rows.insertRow([geoid, markers[geoid]])
            rows.insertRow([geoid, markers[geoid]])
            expected_order.extend((geoid, geoid))
    with arcpy.da.SearchCursor(
        str(marker_fc), ["CensusTrac"], sql_clause=(None, "ORDER BY OBJECTID")
    ) as rows:
        actual_order = [row[0] for row in rows]
    if actual_order != expected_order:
        raise ValueError("Marker rows were not written twice in sorted tract order")
    if _factory_code(marker_fc) != _factory_code(river):
        raise ValueError("The marker layer did not inherit the clipped river spatial reference")
    return gdb


def _copy_figure10_series(root: Path, local: Path) -> tuple[Path, str, str]:
    source = root / "series" / "Fig10_FI.shp"
    workspace = local / "joined"
    workspace.mkdir()
    for extension in FIG10_SIDECARS:
        item = _require_file(source.with_suffix(extension), f"Figure 10 {extension} sidecar")
        shutil.copy2(str(item), str(workspace / item.name))
    if _count(workspace / source.name) != 27 or _factory_code(workspace / source.name) != 4269:
        raise ValueError("Expected 27 NAD83 Figure 10 tract rows")
    return workspace, source.name, "Shapefile"


def _build_population_series(root: Path, local: Path, population: dict) -> tuple[Path, str, str]:
    source = _require_file(root / "series" / "public_27.shp", "public NAD83 tract shapefile")
    if _count(source) != 27 or _factory_code(source) != 4269:
        raise ValueError("Expected 27 public NAD83 population tract rows")
    workspace = local / "population.gdb"
    arcpy.management.CreateFileGDB(str(local), workspace.name)
    dataset = "Tract_population"
    feature_class = Path(arcpy.management.CopyFeatures(str(source), str(workspace / dataset))[0])
    fields = [f"{prefix}_dpct_{year}" for prefix in ("owner", "renter") for year in (2021, 2022, 2023)]
    for field in fields:
        arcpy.management.AddField(str(feature_class), field, "DOUBLE")
    with arcpy.da.UpdateCursor(str(feature_class), ["GEOID", *fields]) as rows:
        seen = set()
        for row in rows:
            geoid = row[0]
            if geoid in seen or geoid not in population["homeowner", 2021]:
                raise ValueError("Unexpected population tract GEOID")
            seen.add(geoid)
            values = []
            for prefix in ("owner", "renter"):
                group = "homeowner" if prefix == "owner" else "renter"
                for year in (2021, 2022, 2023):
                    values.append(float(population[group, year][geoid]))
            rows.updateRow([geoid, *values])
    if seen != population["homeowner", 2021].keys():
        raise ValueError("Population feature data do not cover all canonical tracts")
    return workspace, dataset, "FileGDB"


def _set_connection(layer, dataset: str, workspace: Path, factory: str) -> None:
    cim = layer.getDefinition("V3")
    connection = arcpy.cim.CreateCIMObjectFromClassName("CIMStandardDataConnection", "V3")
    connection.workspaceConnectionString = f"DATABASE={workspace}"
    connection.workspaceFactory = factory
    connection.dataset = dataset
    connection.datasetType = "esriDTFeatureClass"
    cim.featureTable.dataConnection = connection
    layer.setDefinition(cim)


def _local_source(layer, local: Path) -> bool:
    try:
        return Path(layer.dataSource).resolve().is_relative_to(local.resolve())
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return False


def _import_and_bind(
    blank: Path,
    pagx: Path,
    local: Path,
    gdb: Path,
    layout_name: str,
    result_workspace: Path,
    result_dataset: str,
    result_factory: str,
    result_prefix: str,
) -> tuple[object, object, Path]:
    project_path = local / "layout.aprx"
    arcpy.mp.ArcGISProject(str(blank)).saveACopy(str(project_path))
    project = arcpy.mp.ArcGISProject(str(project_path))
    project.importDocument(str(pagx))
    project.homeFolder = str(local)
    project.defaultGeodatabase = str(gdb)
    project.updateDatabases([{"databasePath": str(gdb), "isDefaultDatabase": True}], True)
    project.updateFolderConnections(
        [{"connectionString": str(local), "alias": "Project folder", "isHomeFolder": True}]
    )

    layouts = project.listLayouts(layout_name)
    if len(layouts) != 1:
        raise ValueError(f"Expected one imported {layout_name}")
    layout = layouts[0]
    frames = [frame for frame in layout.listElements("MAPFRAME_ELEMENT") if frame.visible]
    if len(frames) != 6 or any(frame.map is None for frame in frames):
        raise ValueError("Expected six visible map frames with maps")
    used_maps = {frame.map.name for frame in frames}
    if len(used_maps) != 6:
        raise ValueError("Expected six distinct panel maps")

    for map_item in list(project.listMaps()):
        if map_item.name not in used_maps:
            project.deleteItem(map_item)
            continue
        if len(map_item.listLayers()) != 4 or map_item.listTables():
            raise ValueError(f"{map_item.name}: expected four layers and no tables")
        for layer in map_item.listLayers():
            if layer.name == "mark":
                dataset, workspace, factory = "FD_2023", gdb, "FileGDB"
            elif layer.name in ("river", "passaic_river_polygon"):
                dataset, workspace, factory = layer.name, gdb, "FileGDB"
            elif layer.name.startswith(result_prefix):
                dataset, workspace, factory = result_dataset, result_workspace, result_factory
            else:
                raise ValueError(f"{map_item.name}: unexpected layer {layer.name}")
            _set_connection(layer, dataset, workspace, factory)
            if (
                not layer.visible
                or layer.isBroken
                or layer.isWebLayer
                or layer.isBasemapLayer
                or not layer.supports("DATASOURCE")
                or not _local_source(layer, local)
            ):
                raise ValueError(f"{map_item.name}/{layer.name}: broken or external active layer")

    if len(project.listMaps()) != 6 or len(project.listLayouts()) != 1 or project.listBrokenDataSources():
        raise ValueError("The cleaned project contains extra maps/layouts or broken sources")
    project.save()
    return project, layout, project_path


def _verify_layout(layout, expected: dict, population: bool, local: Path) -> None:
    panels = POPULATION_PANELS if population else PANELS
    seen = set()
    verified_values = 0
    for frame in [item for item in layout.listElements("MAPFRAME_ELEMENT") if item.visible]:
        layers = [layer for layer in frame.map.listLayers() if layer.visible]
        if any(
            layer.isBroken
            or layer.isWebLayer
            or layer.isBasemapLayer
            or not _local_source(layer, local)
            for layer in layers
        ):
            raise ValueError(f"{frame.name}: broken or external active layer")
        mapped = [layer for layer in layers if layer.name in panels]
        if len(mapped) != 1 or mapped[0].name in seen:
            raise ValueError(f"{frame.name}: unexpected or repeated result layer")
        layer = mapped[0]
        group, year, column = panels[layer.name]
        binding = [layer.getDefinition("V3").renderer.field] if population else list(layer.symbology.renderer.fields)
        if binding != [column]:
            raise ValueError(f"{layer.name}: unexpected renderer field")
        with arcpy.da.SearchCursor(layer, ["GEOID", column]) as rows:
            values = list(rows)
        verified_values += len(values)
        if population:
            normalized = ((geoid, None if value is None else f"{value:.9f}") for geoid, value in values)
            verify_panel(normalized, expected[group, year], layer.name, "population value")
        else:
            verify_panel(values, expected[group, year], layer.name)
        seen.add(layer.name)
    if seen != panels.keys() or verified_values != 162:
        raise ValueError("Expected all six result panels and exactly 162 mapped values")


def _reopen_local_project(project_path: Path, local: Path, gdb: Path, layout_name: str):
    """Reopen saved bytes and recheck that no imported source path survived."""
    project = arcpy.mp.ArcGISProject(str(project_path))
    if Path(project.homeFolder).resolve() != local.resolve():
        raise ValueError("The reopened project home folder is not local")
    if Path(project.defaultGeodatabase).resolve() != gdb.resolve():
        raise ValueError("The reopened project default geodatabase is not local")
    if any(
        not Path(item["connectionString"]).resolve().is_relative_to(local.resolve())
        for item in project.folderConnections
    ):
        raise ValueError("The reopened project contains an external folder connection")
    if any(
        not Path(item["databasePath"]).resolve().is_relative_to(local.resolve())
        for item in project.databases
    ):
        raise ValueError("The reopened project contains an external database connection")
    layouts = project.listLayouts(layout_name)
    if len(layouts) != 1 or len(project.listLayouts()) != 1 or len(project.listMaps()) != 6:
        raise ValueError("The reopened project did not preserve the cleaned layout and maps")
    layout = layouts[0]
    for frame in [item for item in layout.listElements("MAPFRAME_ELEMENT") if item.visible]:
        if len(frame.map.listLayers()) != 4 or frame.map.listTables():
            raise ValueError(f"{frame.name}: reopened map structure changed")
        for layer in frame.map.listLayers():
            if (
                not layer.visible
                or layer.isBroken
                or layer.isWebLayer
                or layer.isBasemapLayer
                or not _local_source(layer, local)
            ):
                raise ValueError(f"{frame.name}/{layer.name}: reopened source is broken or external")
    if project.listBrokenDataSources():
        raise ValueError("The reopened project contains a broken data source")
    return project, layout


def _exclusive_export(layout, destination: Path, temporary: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    scratch = temporary / destination.name
    layout.exportToPNG(str(scratch), resolution=300, clip_to_elements=False)
    with scratch.open("rb") as source, destination.open("xb") as target:
        shutil.copyfileobj(source, target)


def _build_figure(
    root: Path,
    build: Path,
    blank: Path,
    pagx: Path,
    clipped_river: Path,
    markers: dict[str, object],
    population: bool,
    expected: dict,
) -> None:
    tag = "FigureS7" if population else "Figure10"
    local = build / tag
    local.mkdir()
    gdb = _create_base_gdb(root, local, clipped_river, markers)
    if population:
        result_workspace, result_dataset, result_factory = _build_population_series(root, local, expected)
        layout_name, prefix, output_name = "Layout171", "Population ", S7_PAGE
    else:
        result_workspace, result_dataset, result_factory = _copy_figure10_series(root, local)
        layout_name, prefix, output_name = "Layout17", "Figure10 ", FIGURE10_PAGE
    project, layout, project_path = _import_and_bind(
        blank,
        pagx,
        local,
        gdb,
        layout_name,
        result_workspace,
        result_dataset,
        result_factory,
        prefix,
    )
    del layout
    del project
    project, layout = _reopen_local_project(project_path, local, gdb, layout_name)
    _verify_layout(layout, expected, population, local)
    if population:
        print("ArcGIS: verified all 162 plotted population changes at nine decimals", flush=True)
    else:
        patches = [item for item in layout.listElements("GRAPHIC_ELEMENT") if item.name == "Patch 4"]
        if len(patches) != 1:
            raise ValueError("Expected one Figure 10 Patch 4 legend marker")
        patches[0].elementPositionY = 14.78
        print("ArcGIS: verified all 162 plotted bivariate classes", flush=True)
    with TemporaryDirectory(prefix="page-", dir=str(local)) as page_folder:
        _exclusive_export(layout, root / output_name, Path(page_folder))
    if project.listBrokenDataSources() or not project_path.is_file():
        raise ValueError(f"{tag}: project failed its final local-source check")
    print("EXPORTED_PUBLIC_LAYOUT", tag, root / output_name, flush=True)


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit(
            "usage: _wrr_public_map_arcgis.py WORK_DIR FIGURE10_PAGX S7_PAGX BLANK_APRX"
        )
    root = Path(sys.argv[1]).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"WORK_DIR is not a directory: {root}")
    arcgis_version = str(arcpy.GetInstallInfo().get("Version", ""))
    if sys.version_info[:2] != (3, 9) or arcgis_version.split(".")[:2] != ["3", "2"]:
        raise ValueError(f"Expected ArcGIS Pro 3.2 with Python 3.9; got {arcgis_version}, {sys.version.split()[0]}")
    figure10_pagx = _pinned_pagx(Path(sys.argv[2]), FIGURE10_PAGX_SHA256, "Figure 10 PAGX")
    s7_pagx = _pinned_pagx(Path(sys.argv[3]), S7_PAGX_SHA256, "S7 PAGX")
    blank = _require_file(Path(sys.argv[4]), "installed blank APRX")
    if hashlib.sha256(blank.read_bytes()).hexdigest() != BLANK_PROJECT_SHA256:
        raise ValueError("Blank.aprx must be the pinned stock ArcGIS Pro 3.2 template")
    if (root / FIGURE10_PAGE).exists() or (root / S7_PAGE).exists():
        raise FileExistsError("Refusing to overwrite an existing raw page PNG")

    figure10_csv = _require_file(root / "series" / "Figure10_spatial_fi.csv", "Figure 10 canonical CSV")
    population_csv = _require_file(root / "series" / "FigureS7_population_change.csv", "S7 canonical CSV")
    figure10_expected = expected_classes(figure10_csv.read_bytes())
    population_expected = expected_population(population_csv.read_bytes())
    markers = _load_markers(root)

    # ArcGIS can retain File GDB table handles until its Python process exits.
    # Keep this caller-owned directory so the portable parent can remove it
    # after the subprocess has terminated instead of failing here on Windows.
    build = Path(mkdtemp(prefix="wrr-public-arcgis-", dir=str(root)))
    print("ARCGIS_BUILD_DIR", build, flush=True)
    clipped_river = _prepare_native_clip(root, build)
    _build_figure(
        root, build, blank, figure10_pagx, clipped_river, markers, False, figure10_expected
    )
    _build_figure(root, build, blank, s7_pagx, clipped_river, markers, True, population_expected)


if __name__ == "__main__":
    main()

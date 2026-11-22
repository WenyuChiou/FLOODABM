"""Rebuild the accepted model or Figure 1 SFHA area tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
from io import BytesIO, StringIO
from pathlib import Path

if __package__:
    from .wrr_public_sources import fema_geometry, read_limited, verify_digest
else:
    from wrr_public_sources import fema_geometry, read_limited, verify_digest


TRACTS_SHA256 = "07ac664f8c171459e8625fd12a0b065c3f7aad6ecf4ad907c2f5d2d9e808d3b1"
SFHA_SHA256 = "9bb2422c77a91ac9c7abf5e3358b2753aaafc2e1cb7ebda4f2a372de6664dfc0"
MODEL_CSV = "sfha_area_share_by_tract.csv"
MAP_CSV = "sfha_tract_area_coverage.csv"
MODEL_SHA256 = "09603907a1ab7aa66b128d939df73dadfbb77b49c93203ae2e546c3ab7dbb818"
MAP_SHA256 = "047b82516a8a23dcf80ffcd98917cec097fc06a00964119539c033781c94cd1e"
MODEL_REFERENCE = Path(__file__).resolve().parents[2] / "config" / MODEL_CSV


def verified_source(path: Path, digest: str) -> BytesIO:
    snapshot = path.read_bytes()
    if hashlib.sha256(snapshot).hexdigest() != digest:
        raise ValueError(f"Unexpected input SHA-256: {path.name}. Use the archived layer revision.")
    return BytesIO(snapshot)


def area_table(tracts, sfha, *, epsg: int):
    """Compute one row per tract in the specified metre-based projection."""
    if tracts.crs is None or sfha.crs is None:
        raise ValueError("Both layers must define a coordinate reference system")
    geoids = tracts["Tract"].astype(str)
    if geoids.duplicated().any() or not geoids.str.fullmatch(r"[0-9]{11}").all():
        raise ValueError("Expected unique 11-digit tract GEOIDs")
    projected = tracts.to_crs(epsg)
    # Union before intersection so overlapping flood-zone polygons count once.
    # The archived inputs are pinned; retain the original union operation.
    flood_union = sfha.to_crs(epsg).geometry.union_all()
    if flood_union.is_empty or not flood_union.is_valid:
        raise ValueError("The projected SFHA union is empty or invalid")
    result = geoids.to_frame(name="tract_geoid")
    result["tract_area_m2"] = projected.geometry.area
    result["sfha_area_m2"] = projected.geometry.intersection(flood_union).area
    result["sfha_area_share"] = result.sfha_area_m2 / result.tract_area_m2
    if not result.tract_area_m2.gt(0).all() or not result.sfha_area_share.between(0, 1).all():
        raise ValueError("Tract areas must be positive and covered shares must lie in [0, 1]")
    return result.sort_values("tract_geoid").reset_index(drop=True)


def encode_map_table(table) -> bytes:
    with StringIO() as buffer:
        output = csv.writer(buffer, lineterminator="\n")
        output.writerow(("tract_geoid", "sfha_area_percent", "tract_area_m2", "sfha_covered_area_m2"))
        for row in table.itertuples(index=False):
            output.writerow((row.tract_geoid, round(100 * row.sfha_area_m2 / row.tract_area_m2, 6),
                             round(row.tract_area_m2, 3), round(row.sfha_area_m2, 3)))
        return buffer.getvalue().encode("utf-8")


def verified_tables(tracts_path: Path, sfha_path: Path) -> dict[str, bytes]:
    tracts_source = verified_source(tracts_path, TRACTS_SHA256)
    sfha_source = verified_source(sfha_path, SFHA_SHA256)
    import geopandas as gpd

    # GDAL must parse the verified snapshots, not reopen paths or GeoPackage sidecars.
    tracts = gpd.read_file(tracts_source, columns=["Tract"], engine="pyogrio")
    sfha = gpd.read_file(sfha_source, columns=[], engine="pyogrio")
    if len(tracts) != 27 or len(sfha) != 1428:
        raise ValueError("Unexpected number of features in the archived SFHA overlay inputs")
    model = area_table(tracts, sfha, epsg=26918)
    figure = area_table(tracts, sfha, epsg=5070)
    outputs = {
        MODEL_CSV: model.to_csv(index=False, lineterminator="\n").encode("utf-8"),
        MAP_CSV: encode_map_table(figure),
    }
    for name, digest in ((MODEL_CSV, MODEL_SHA256), (MAP_CSV, MAP_SHA256)):
        if hashlib.sha256(outputs[name]).hexdigest() != digest:
            raise ValueError(f"{name} does not match the accepted CSV. Check the pinned GIS environment.")
    return outputs


def verified_public_map_table(sfha_path: Path, source_dir: Path | None) -> bytes:
    """Recompute only Figure 1 with pinned FEMA tracts and archived SFHA zones."""
    import geopandas as gpd

    sfha_source = verified_source(sfha_path, SFHA_SHA256)
    sfha = gpd.read_file(sfha_source, columns=[], engine="pyogrio")
    if len(sfha) != 1428:
        raise ValueError("Unexpected number of features in the archived SFHA overlay")

    # Git may change line endings, but the scientific fields and rows are pinned.
    reference = read_limited(MODEL_REFERENCE, 10_000).replace(b"\r\n", b"\n")
    verify_digest(reference, MODEL_SHA256, MODEL_CSV)
    rows = csv.DictReader(StringIO(reference.decode("utf-8")))
    ids = tuple(sorted(row["tract_geoid"] for row in rows))
    tracts = fema_geometry(source_dir, ids)
    result = encode_map_table(area_table(tracts, sfha, epsg=5070))
    verify_digest(result, MAP_SHA256, MAP_CSV)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tracts", type=Path, help="Archived study_tracts_svi.gpkg; rebuild both tables")
    mode.add_argument("--public-tracts", action="store_true", help="Use public FEMA tracts; Figure 1 only")
    parser.add_argument("--public-source-dir", type=Path, help="Offline FEMA GeoJSON cache for --public-tracts")
    parser.add_argument("--sfha", type=Path, required=True, help="Archived nj_nfhl_sfha_study_tracts.geojson")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for the verified CSV output")
    args = parser.parse_args()
    if args.public_source_dir is not None and not args.public_tracts:
        parser.error("--public-source-dir requires --public-tracts")
    try:
        if args.output_dir.exists() or args.output_dir.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
        if args.public_tracts:
            tables = {MAP_CSV: verified_public_map_table(args.sfha, args.public_source_dir)}
        else:
            tables = verified_tables(args.tracts, args.sfha)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        for name, content in tables.items():
            path = args.output_dir / name
            with path.open("xb") as output:
                output.write(content)
            print(f"PASS {path}: matches the 27-tract reference")
    except (ImportError, OSError, ValueError) as error:
        parser.exit(1, f"WRR SFHA areas: {error}\n")


if __name__ == "__main__":
    main()

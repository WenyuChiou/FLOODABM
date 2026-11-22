import hashlib
from io import BytesIO
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request

import pytest

from scripts.utilities import rebuild_wrr_sfha_areas as rebuild
from scripts.utilities import wrr_public_sources as sources


def test_overlay_unions_overlaps_before_calculating_tract_share():
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    tracts = gpd.GeoDataFrame(
        {"Tract": ["34013000200", "34013000100"]},
        geometry=[box(4, 0, 8, 4), box(0, 0, 4, 4)],
        crs=5070,
    )
    zones = gpd.GeoDataFrame(
        geometry=[box(0, 0, 2, 3), box(1, 0, 4, 3)], crs=5070
    )

    table = rebuild.area_table(tracts, zones, epsg=5070)

    assert list(table.tract_geoid) == ["34013000100", "34013000200"]
    assert list(table.sfha_area_m2) == [12, 0]
    assert list(table.sfha_area_share) == [0.75, 0]
    assert rebuild.encode_map_table(table) == (
        b"tract_geoid,sfha_area_percent,tract_area_m2,sfha_covered_area_m2\n"
        b"34013000100,75.0,16.0,12.0\n"
        b"34013000200,0.0,16.0,0.0\n"
    )


def test_invalid_geoid_or_missing_crs_is_rejected():
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    tract = box(0, 0, 1, 1)
    zones = gpd.GeoDataFrame(geometry=[tract], crs=5070)
    tracts = gpd.GeoDataFrame(
        {"Tract": ["34013000100", "34013000100"]}, geometry=[tract, tract]
    )
    with pytest.raises(ValueError, match="coordinate reference system"):
        rebuild.area_table(tracts, zones, epsg=5070)
    with pytest.raises(ValueError, match="unique 11-digit tract GEOIDs"):
        rebuild.area_table(tracts.set_crs(5070), zones, epsg=5070)


def test_modified_source_fails_without_creating_outputs(tmp_path, monkeypatch, capsys):
    modified = tmp_path / "study_tracts_svi.gpkg"
    modified.write_bytes(b"not the archived revision")
    destination = tmp_path / "areas"
    monkeypatch.setattr(
        "sys.argv",
        [
            "rebuild_wrr_sfha_areas.py",
            "--tracts", str(modified),
            "--sfha", str(tmp_path / "nj_nfhl_sfha_study_tracts.geojson"),
            "--output-dir", str(destination),
        ],
    )
    with pytest.raises(SystemExit) as error:
        rebuild.main()
    assert error.value.code == 1
    assert "Unexpected input SHA-256: study_tracts_svi.gpkg" in capsys.readouterr().err
    assert not destination.exists()


def test_gdal_receives_verified_snapshots_even_if_original_path_changes(tmp_path, monkeypatch):
    gpd = pytest.importorskip("geopandas")
    tracts, sfha = tmp_path / "tracts.gpkg", tmp_path / "sfha.geojson"
    tracts.write_bytes(b"verified tracts")
    sfha.write_bytes(b"verified SFHA")
    monkeypatch.setattr(rebuild, "TRACTS_SHA256", hashlib.sha256(tracts.read_bytes()).hexdigest())
    monkeypatch.setattr(rebuild, "SFHA_SHA256", hashlib.sha256(sfha.read_bytes()).hexdigest())
    received = []

    def read_file(snapshot, **kwargs):
        tracts.write_bytes(b"changed on disk before GDAL parsing")
        assert isinstance(snapshot, BytesIO)
        received.append(snapshot.read())
        return []

    monkeypatch.setattr(gpd, "read_file", read_file)
    with pytest.raises(ValueError, match="Unexpected number of features"):
        rebuild.verified_tables(tracts, sfha)
    assert received == [b"verified tracts", b"verified SFHA"]


def test_existing_destination_is_not_overwritten(tmp_path, monkeypatch, capsys):
    existing = tmp_path / rebuild.MODEL_CSV
    existing.write_bytes(b"do not overwrite")
    monkeypatch.setattr(
        "sys.argv",
        [
            "rebuild_wrr_sfha_areas.py",
            "--tracts", str(Path("unused.gpkg")),
            "--sfha", str(Path("unused.geojson")),
            "--output-dir", str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit) as error:
        rebuild.main()
    assert error.value.code == 1
    assert "Refusing to overwrite" in capsys.readouterr().err
    assert existing.read_bytes() == b"do not overwrite"


@pytest.mark.parametrize("matches_reference", [True, False])
@pytest.mark.parametrize("public_nfhl", [True, False])
def test_public_route_writes_only_the_exact_map_table(
    tmp_path, monkeypatch, matches_reference, public_nfhl,
):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    area = box(0, 0, 4, 4)
    zones = gpd.GeoDataFrame(geometry=[area] * 1428, crs=5070)
    ids = tuple(f"34013{i:06d}" for i in range(27))
    tracts = gpd.GeoDataFrame({"Tract": list(ids)}, geometry=[area] * 27, crs=5070)
    reference = b"tract_geoid\n" + b"".join(x.encode() + b"\n" for x in ids)
    model_reference = tmp_path / "shipped.csv"
    model_reference.write_bytes(reference.replace(b"\n", b"\r\n"))
    monkeypatch.setattr(rebuild, "MODEL_REFERENCE", model_reference)
    monkeypatch.setattr(rebuild, "MODEL_SHA256", hashlib.sha256(reference).hexdigest())
    wanted = b"tract_geoid,sfha_area_percent,tract_area_m2,sfha_covered_area_m2\n"
    wanted += b"".join(x.encode() + b",100.0,16.0,16.0\n" for x in ids)
    monkeypatch.setattr(
        rebuild, "MAP_SHA256", hashlib.sha256(wanted if matches_reference else b"wrong").hexdigest()
    )
    nfhl = tmp_path / "pinned_nfhl.geojson"
    nfhl.write_bytes(b"archived flood data")
    monkeypatch.setattr(rebuild, "SFHA_SHA256", hashlib.sha256(nfhl.read_bytes()).hexdigest())

    def read_file(snapshot, **_):
        assert isinstance(snapshot, BytesIO)
        assert snapshot.read() == b"archived flood data"
        return zones

    def fema_geometry(cache, geoids):
        assert cache is None and geoids == ids
        return tracts

    monkeypatch.setattr(gpd, "read_file", read_file)
    monkeypatch.setattr(rebuild, "fema_geometry", fema_geometry)
    if public_nfhl:
        monkeypatch.setattr(rebuild, "njdep_nfhl_payload", lambda _: b"archived flood data")
        flood_args = ["--public-nfhl"]
    else:
        flood_args = ["--sfha", str(nfhl)]
    output = tmp_path / "areas"
    monkeypatch.setattr(
        "sys.argv", ["rebuild", "--public-tracts", *flood_args, "--output-dir", str(output)]
    )
    if matches_reference:
        rebuild.main()
        assert sorted(x.name for x in output.iterdir()) == [rebuild.MAP_CSV]
        assert (output / rebuild.MAP_CSV).read_bytes() == wanted
    else:
        with pytest.raises(SystemExit) as error:
            rebuild.main()
        assert error.value.code == 1
        assert not output.exists()
    assert model_reference.read_bytes() == reference.replace(b"\n", b"\r\n")


def test_modified_public_route_flood_input_does_not_contact_provider(tmp_path, monkeypatch, capsys):
    pytest.importorskip("geopandas")
    nfhl = tmp_path / "nfhl.geojson"
    nfhl.write_bytes(b"new NFHL download")
    output = tmp_path / "areas"

    def no_query(*_):
        pytest.fail("Public geometries must not be queried after NFHL verification fails")

    monkeypatch.setattr(rebuild, "fema_geometry", no_query)
    monkeypatch.setattr(
        "sys.argv", ["rebuild", "--public-tracts", "--sfha", str(nfhl), "--output-dir", str(output)]
    )
    with pytest.raises(SystemExit) as error:
        rebuild.main()
    assert error.value.code == 1
    assert "Unexpected input SHA-256: nfhl.geojson" in capsys.readouterr().err
    assert not output.exists()


def test_shared_fema_cache_checks_geometry_independent_of_feature_order(tmp_path, monkeypatch):
    pytest.importorskip("geopandas")
    from shapely import normalize
    from shapely.geometry import box, mapping

    ids = tuple(f"34013{i:06d}" for i in range(27))
    geom = box(0, 0, 1, 1)
    digest = hashlib.sha256()
    for geoid in ids:
        digest.update(geoid.encode("ascii"))
        digest.update(normalize(geom).wkb)
    monkeypatch.setattr(sources, "FEMA_GEOMETRY_SHA", digest.hexdigest())
    features = [
        {"type": "Feature", "properties": {"Tract": geoid}, "geometry": mapping(geom)}
        for geoid in reversed(ids)
    ]
    cache = tmp_path / "fema_tracts.geojson"

    def cache_features():
        cache.write_text(json.dumps({"type": "FeatureCollection", "features": features}))

    cache_features()
    assert tuple(sources.fema_geometry(tmp_path, ids).Tract) == ids
    features[0]["geometry"] = mapping(box(0, 0, 1, 2))
    cache_features()
    with pytest.raises(ValueError, match="does not match the verified revision"):
        sources.fema_geometry(tmp_path, ids)
    features[0]["geometry"] = {"type": "InvalidPolygon", "coordinates": []}
    cache_features()
    with pytest.raises(ValueError, match="invalid tract or geometry schema"):
        sources.fema_geometry(tmp_path, ids)


@pytest.mark.parametrize(
    "url", ["https://other.example/data", "http://source.example/data", "https://source.example/data"]
)
def test_public_download_rejects_redirect_before_requesting_another_location(url):
    handler = sources.RejectPublicSourceRedirect()
    with pytest.raises(ValueError, match="Unexpected public source redirect"):
        handler.redirect_request(Request("https://source.example/data"), None, 302, "Found", {}, url)


def test_njdep_public_query_replays_the_original_envelope_and_zone_filter(monkeypatch):
    snapshot = b"mock pinned August 2026 response"

    def checked_request(cache, name, uri, limit):
        assert cache is None
        assert name == "nj_nfhl_sfha_study_tracts.geojson"
        assert limit == 16_000_000
        parsed = urlsplit(uri)
        assert parsed.scheme == "https"
        assert parsed.netloc == "mapsdep.nj.gov"
        assert parsed.path == "/arcgis/rest/services/Features/Hydrography/MapServer/43/query"
        query = {key: values[0] for key, values in parse_qs(parsed.query).items()}
        assert json.loads(query.pop("geometry")) == {
            "xmin": -8283243.078172955, "ymin": 4988017.55332962,
            "xmax": -8259463.001313929, "ymax": 5012955.030759927,
            "spatialReference": {"wkid": 3857},
        }
        assert query == {
            "geometryType": "esriGeometryEnvelope", "inSR": "3857",
            "spatialRel": "esriSpatialRelIntersects",
            "where": "FLD_ZONE IN ('A','AE','AH','AO','AR','A99','V','VE')",
            "outFields": "OBJECTID,FLD_ZONE,ZONE_SUBTY", "returnGeometry": "true",
            "outSR": "4326", "resultRecordCount": "2000", "f": "geojson",
        }
        return snapshot

    monkeypatch.setattr(sources, "NJDEP_NFHL_SHA256", hashlib.sha256(snapshot).hexdigest())
    monkeypatch.setattr(sources, "public_input", checked_request)
    assert sources.njdep_nfhl_payload(None) == snapshot


def test_changed_njdep_cache_cannot_publish_a_figure1_table(tmp_path, monkeypatch, capsys):
    pytest.importorskip("geopandas")
    cache = tmp_path / sources.NJDEP_NFHL_FILE
    cache.write_bytes(b"changed public flood data")
    output = tmp_path / "areas"

    def no_query(*_):
        pytest.fail("A changed NFHL cache must fail before any tract query")

    monkeypatch.setattr(rebuild, "fema_geometry", no_query)
    monkeypatch.setattr(sources, "build_opener", no_query)
    monkeypatch.setattr(
        "sys.argv", [
            "rebuild", "--public-tracts", "--public-nfhl",
            "--public-source-dir", str(tmp_path), "--output-dir", str(output),
        ],
    )
    with pytest.raises(SystemExit) as error:
        rebuild.main()
    assert error.value.code == 1
    assert "NJDEP NFHL bytes differ from the pinned revision" in capsys.readouterr().err
    assert not output.exists()

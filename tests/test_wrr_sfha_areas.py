import hashlib
from io import BytesIO
from pathlib import Path

import pytest

from scripts.utilities import rebuild_wrr_sfha_areas as rebuild


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

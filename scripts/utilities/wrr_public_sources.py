"""Pinned public GIS inputs shared by the revised WRR reproduction tools."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING, IO, Mapping
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

if TYPE_CHECKING:
    from email.message import Message
    from geopandas import GeoDataFrame


FEMA_QUERY = (
    "https://services.arcgis.com/XG15cJAlne2vxtgt/ArcGIS/rest/services/"
    "FEMA_HurricaneWind_Tract/FeatureServer/0/query"
)
FEMA_GEOMETRY_SHA = "3c432a1d16673f7a9e8dfd7ccafb627b82c84d28dbdfffa6cb8a3d629222dc71"


class RejectPublicSourceRedirect(HTTPRedirectHandler):
    """Read directly from the pinned provider URL, without following redirects."""

    def redirect_request(
        self, req: Request, fp: IO[bytes] | None, code: int, msg: str,
        headers: Message | Mapping[str, str], newurl: str,
    ) -> Request | None:
        raise ValueError("Unexpected public source redirect")


def read_limited(path: Path, limit: int) -> bytes:
    """Read a bounded local input."""
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"{path.name} exceeds the expected input size")
    return data


def verify_digest(data: bytes, expected: str, label: str) -> None:
    """Reject bytes that do not match this revision's input or output."""
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"{label}: SHA-256 differs from the pinned revision")


def public_input(cache: Path | None, name: str, url: str, limit: int) -> bytes:
    """Read a supplied cache file or download it from the recorded public provider."""
    if cache is not None:
        return read_limited(cache / name, limit)
    requested = urlsplit(url)
    if requested.scheme != "https" or not requested.netloc:
        raise ValueError(f"{name}: expected an HTTPS public source")
    opener = build_opener(RejectPublicSourceRedirect)
    with opener.open(url, timeout=30) as response:
        final = urlsplit(response.geturl())
        if final.scheme != requested.scheme or final.netloc != requested.netloc:
            raise ValueError(f"{name}: unexpected source redirect")
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"{name}: response exceeds the expected input size")
    return data


def fema_geometry(cache: Path | None, tracts: tuple[str, ...]) -> GeoDataFrame:
    """Validate the public FEMA geometry by ID and normalized WKB, not JSON order."""
    import geopandas as gpd
    import shapely
    from shapely.errors import ShapelyError
    from shapely.geometry import shape

    if (
        len(tracts) != 27
        or tuple(sorted(set(tracts))) != tracts
        or not all(re.fullmatch(r"[0-9]{11}", tract) for tract in tracts)
    ):
        raise ValueError("Expected 27 unique, sorted, 11-digit revised tract GEOIDs")
    ids = ",".join(f"'{tract}'" for tract in tracts)
    params = {
        "f": "geojson", "where": f"Tract IN ({ids})", "outFields": "Tract",
        "returnGeometry": "true", "outSR": 4326, "returnZ": "false", "returnM": "false",
    }
    source = public_input(
        cache, "fema_tracts.geojson", FEMA_QUERY + "?" + urlencode(params), 2_000_000
    )
    data = json.loads(source)
    if not isinstance(data, dict) or data.get("type") != "FeatureCollection" or "error" in data:
        raise ValueError("FEMA response did not contain the expected feature collection")
    try:
        features = sorted(data["features"], key=lambda item: item["properties"]["Tract"])
        if tuple(item["properties"]["Tract"] for item in features) != tracts:
            raise ValueError("FEMA changed the revised tract set")
        digest = hashlib.sha256()
        for feature in features:
            digest.update(feature["properties"]["Tract"].encode("ascii"))
            digest.update(shapely.normalize(shape(feature["geometry"])).wkb)
    except (AttributeError, KeyError, ShapelyError, TypeError) as error:
        raise ValueError("FEMA response has an invalid tract or geometry schema") from error
    if digest.hexdigest() != FEMA_GEOMETRY_SHA:
        raise ValueError("Public FEMA geometry does not match the verified revision")
    return gpd.GeoDataFrame.from_features(features, crs=4326)

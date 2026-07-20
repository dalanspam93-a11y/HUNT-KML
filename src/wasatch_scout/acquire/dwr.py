"""Utah DWR Open Data Hub layers (crucial range, migration corridors, unit
boundary) and ownership/SITLA parcels, via ArcGIS REST FeatureServer "query".

NOT reachable from this sandbox -- dwr-data-utahdnr.hub.arcgis.com and
gis.utah.gov (UGRC, for ownership) are both blocked by the outbound network
policy here. `query_featureserver` is a generic, correct implementation of
the standard Esri REST query pattern and will work unmodified wherever these
hosts are reachable (your machine, or a less-restricted session) -- just
fill in the *_service_url fields in config.yaml (see the comments there for
how to find them) and rerun. Until then, drop manually-exported GeoJSON at
the *_file paths in config and this module will use those instead.
"""
from __future__ import annotations

import logging

import geopandas as gpd
import requests

from ..aoi import AOI
from ..config import resolve

log = logging.getLogger(__name__)

WGS84 = "EPSG:4326"


def query_featureserver(service_url: str, aoi: AOI) -> gpd.GeoDataFrame | None:
    """Query an Esri FeatureServer/MapServer layer, clipped to the AOI bbox, as GeoJSON."""
    if not service_url:
        return None
    min_lon, min_lat, max_lon, max_lat = aoi.bbox_wgs84
    params = {
        "where": "1=1",
        "outFields": "*",
        "f": "geojson",
        "geometry": f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outSR": "4326",
    }
    try:
        r = requests.get(f"{service_url}/query", params=params, timeout=60)
        r.raise_for_status()
        gdf = gpd.read_file(r.text)  # geopandas can parse a GeoJSON string via a file-like/text path
        gdf = gdf.set_crs(WGS84, allow_override=True)
        log.info("Fetched %d feature(s) from %s", len(gdf), service_url)
        return gdf
    except Exception as e:
        log.warning("Query failed for %s: %s", service_url, e)
        return None


def _load_cached_or_query(cfg: dict, service_url_key: str, file_key: str, aoi: AOI, label: str) -> gpd.GeoDataFrame | None:
    cache_path = resolve(cfg, file_key)
    if cache_path.exists():
        log.info("%s: loaded cache %s", label, cache_path)
        return gpd.read_file(cache_path).to_crs(WGS84)

    node = cfg
    for part in service_url_key.split("."):
        node = node[part]
    service_url = node

    gdf = query_featureserver(service_url, aoi) if service_url else None
    if gdf is None:
        log.warning(
            "%s: no service_url configured (or query failed) and no cache at %s. "
            "This layer will be treated as absent -- see config.yaml comments to fill "
            "in the service URL, or manually export the layer to that path.",
            label, cache_path,
        )
        return None

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(cache_path, driver="GeoJSON")
    return gdf


def _load_species_habitat(cfg: dict, file_key: str, aoi: AOI, label: str) -> gpd.GeoDataFrame | None:
    """Load a real DWR species-habitat export (SEASON/VALUE/SPECIES/COMMENTS schema),
    keep only VALUE == 'crucial' polygons, clip to the AOI. "substantial" is a real,
    lower-tier DWR classification -- not used for the overlap bonus."""
    path = resolve(cfg, file_key)
    if not path.exists():
        log.warning("%s: no file at %s -- this species' dwr_overlap term will be dropped.", label, path)
        return None
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(WGS84)
    gdf = gdf.to_crs(WGS84)
    if "VALUE" in gdf.columns:
        before = len(gdf)
        gdf = gdf[gdf["VALUE"].astype(str).str.lower() == "crucial"]
        log.info("%s: %d/%d features are VALUE=crucial", label, len(gdf), before)
    aoi_poly = aoi.boundary_wgs84.geometry.union_all()
    gdf = gdf[gdf.intersects(aoi_poly)]
    log.info("%s: %d crucial feature(s) intersect AOI", label, len(gdf))
    return gdf


def _load_migration_corridor(cfg: dict, aoi: AOI) -> gpd.GeoDataFrame | None:
    """Real DWR migration-corridor export (Type=high/medium/low use, area_name=herd,
    Species). Keeps high+medium use only -- low use is a real but weak signal, same
    reasoning as dropping 'substantial' in _load_species_habitat."""
    path = resolve(cfg, "dwr_habitat.migration_corridor_file")
    if not path.exists():
        log.warning("DWR migration corridor: no file at %s -- this term will be dropped.", path)
        return None
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(WGS84)
    gdf = gdf.to_crs(WGS84)
    if "Type" in gdf.columns:
        before = len(gdf)
        gdf = gdf[gdf["Type"].astype(str).str.lower().isin(["high use", "medium use"])]
        log.info("DWR migration corridor: %d/%d features are high/medium use", len(gdf), before)
    aoi_poly = aoi.boundary_wgs84.geometry.union_all()
    gdf = gdf[gdf.intersects(aoi_poly)]
    log.info("DWR migration corridor: %d feature(s) intersect AOI", len(gdf))
    return gdf


def acquire_dwr_habitat(cfg: dict, aoi: AOI) -> dict[str, gpd.GeoDataFrame | None]:
    deer = _load_species_habitat(cfg, "dwr_habitat.deer_habitat_file", aoi, "DWR mule deer crucial habitat")
    elk = _load_species_habitat(cfg, "dwr_habitat.elk_habitat_file", aoi, "DWR elk crucial habitat")
    migration = _load_migration_corridor(cfg, aoi)
    return {"deer_habitat": deer, "elk_habitat": elk, "migration_corridor": migration}


def acquire_ownership(cfg: dict, aoi: AOI) -> gpd.GeoDataFrame | None:
    return _load_cached_or_query(
        cfg, "ownership.ownership_service_url", "ownership.boundary_ownership_file", aoi, "Ownership/SITLA parcels"
    )

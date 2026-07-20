"""Area of interest: unit boundary (+buffer) resolution.

Prefers an authoritative boundary_file (GeoJSON exported from OnX/DWR Hunt
Planner). Falls back to an explicitly-unofficial bbox from config so the
pipeline can still run end-to-end when that export hasn't been supplied yet.
"""
from __future__ import annotations

import logging

import geopandas as gpd
from shapely.geometry import box

from .config import resolve

log = logging.getLogger(__name__)

WGS84 = "EPSG:4326"
MILES_TO_M = 1609.34


class AOI:
    def __init__(self, boundary_wgs84: gpd.GeoDataFrame, is_authoritative: bool, working_crs: str):
        self.boundary_wgs84 = boundary_wgs84
        self.is_authoritative = is_authoritative
        self.working_crs = working_crs

    @property
    def bbox_wgs84(self) -> tuple[float, float, float, float]:
        return tuple(self.boundary_wgs84.total_bounds)

    def working_gdf(self) -> gpd.GeoDataFrame:
        return self.boundary_wgs84.to_crs(self.working_crs)


def load_aoi(cfg: dict) -> AOI:
    boundary_path = resolve(cfg, "aoi.boundary_file")
    buffer_miles = cfg["aoi"]["buffer_miles"]
    working_crs = cfg["aoi"]["working_crs"]

    if boundary_path.exists():
        unit = gpd.read_file(boundary_path)
        if unit.crs is None:
            unit = unit.set_crs(WGS84)
        unit = unit.to_crs(working_crs)
        buffered = unit.geometry.buffer(buffer_miles * MILES_TO_M)
        aoi_gdf = gpd.GeoDataFrame(geometry=[buffered.union_all()], crs=working_crs).to_crs(WGS84)
        log.info("AOI: loaded authoritative unit boundary from %s (+%.1f mi buffer)", boundary_path, buffer_miles)
        return AOI(aoi_gdf, is_authoritative=True, working_crs=working_crs)

    log.warning(
        "AOI: %s not found — using UNOFFICIAL fallback bbox from config.yaml "
        "(aoi.fallback_bbox_unofficial). This is NOT the authoritative unit boundary. "
        "Export the real boundary from OnX/DWR Hunt Planner before trusting any output.",
        boundary_path,
    )
    bb = cfg["aoi"]["fallback_bbox_unofficial"]
    geom = box(bb["min_lon"], bb["min_lat"], bb["max_lon"], bb["max_lat"])
    aoi_gdf = gpd.GeoDataFrame(geometry=[geom], crs=WGS84)
    # buffer in working CRS then back to WGS84, same as the authoritative path
    buffered = aoi_gdf.to_crs(working_crs).geometry.buffer(buffer_miles * MILES_TO_M)
    aoi_gdf = gpd.GeoDataFrame(geometry=[buffered.union_all()], crs=working_crs).to_crs(WGS84)
    return AOI(aoi_gdf, is_authoritative=False, working_crs=working_crs)

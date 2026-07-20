"""Motorized-legal access points + trailheads that seed the effort surface.

Per the brief, this layer is meant to come FROM OnX/GOHUNT (export trailheads
and MVUM-legal routes as points/lines) rather than be rebuilt from scratch.
This module loads user-supplied exports if present; otherwise it falls back
to a small, loudly-labeled demo seed set (real-world canyon-mouth trailhead
coordinates on the Wasatch Front) so the effort-surface algorithm can be
validated end-to-end before the real export is available. Demo seeds must
NOT be used to generate zones you rely on.
"""
from __future__ import annotations

import logging

import geopandas as gpd
from shapely.geometry import Point

from ..aoi import AOI
from ..config import resolve

log = logging.getLogger(__name__)

WGS84 = "EPSG:4326"

# Real-world, approximate canyon-mouth / trailhead coordinates on the Salt Lake/Utah
# County Wasatch Front, used ONLY as placeholder seeds when no OnX/GOHUNT export is
# present. Coordinates are eyeballed from general knowledge, not survey-grade, and the
# set is not exhaustive -- do not treat as a real access-point inventory.
_DEMO_FOOT_SEEDS = [
    ("Big Cottonwood Canyon mouth", -111.7658, 40.6199),
    ("Little Cottonwood Canyon mouth", -111.7614, 40.5735),
    ("Mill Creek Canyon mouth", -111.7583, 40.6813),
    ("American Fork Canyon mouth", -111.6994, 40.4308),
    ("Corner Canyon trailhead (Draper)", -111.8175, 40.5145),
    ("Provo Canyon / South Fork", -111.5950, 40.2727),
]
_DEMO_ATV_SEEDS = [
    ("Mill Creek Canyon Rd (seasonal, ATV-legal upper)", -111.7100, 40.6900),
    ("American Fork Canyon Rd (MVUM ATV routes)", -111.6300, 40.4550),
]


def _demo_gdf(points: list[tuple[str, float, float]]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"name": [p[0] for p in points]},
        geometry=[Point(p[1], p[2]) for p in points],
        crs=WGS84,
    )


def _load_or_demo(cfg: dict, path_key: str, demo_points, label: str) -> tuple[gpd.GeoDataFrame, bool]:
    path = resolve(cfg, path_key)
    if path.exists():
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            gdf = gdf.set_crs(WGS84)
        log.info("%s: loaded %d seed(s) from %s", label, len(gdf), path)
        return gdf.to_crs(WGS84), True
    log.warning(
        "%s: %s not found. Using %d DEMO seed point(s) for pipeline validation only "
        "-- export real trailheads/routes from OnX or GOHUNT to %s before trusting output.",
        label, path, len(demo_points), path,
    )
    return _demo_gdf(demo_points), False


def load_access_seeds(cfg: dict, aoi: AOI) -> dict:
    """Returns dict with foot/atv GeoDataFrames (WGS84) and authoritative flags, clipped to AOI."""
    foot_gdf, foot_real = _load_or_demo(cfg, "access_seeds.foot_seed_file", _DEMO_FOOT_SEEDS, "Foot access seeds")
    atv_gdf, atv_real = ({}, False)
    if cfg["run"]["atv_available"]:
        atv_gdf, atv_real = _load_or_demo(cfg, "access_seeds.atv_seed_file", _DEMO_ATV_SEEDS, "ATV access seeds")
    else:
        atv_gdf = _demo_gdf([])

    aoi_poly = aoi.boundary_wgs84.geometry.union_all()
    foot_gdf = foot_gdf[foot_gdf.intersects(aoi_poly.buffer(0.05))]  # small slop for near-boundary trailheads
    if len(atv_gdf):
        atv_gdf = atv_gdf[atv_gdf.intersects(aoi_poly.buffer(0.05))]

    return {
        "foot": foot_gdf,
        "atv": atv_gdf,
        "foot_authoritative": foot_real,
        "atv_authoritative": atv_real,
    }

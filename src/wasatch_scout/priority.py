"""Combine habitat score + effort budget + ownership mask into ranked priority
zones: high habitat quality, hard for the crowd to reach (within your own
effort budget), on land you can legally hunt.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import shapes as rio_shapes
from scipy.ndimage import label, find_objects
from shapely.geometry import shape as shapely_shape
from shapely.ops import unary_union

from . import ownership as own
from .config import resolve

log = logging.getLogger(__name__)

ACRES_PER_CELL_FACTOR = 0.000247105  # m^2 -> acres


@dataclass
class Zone:
    rank: int
    zone_id: int
    geometry: object  # shapely polygon, working CRS
    acres: float
    mean_habitat: float
    mean_foot_minutes: float
    min_foot_minutes: float
    dominant_landform: str
    dwr_overlap_pct: float
    ownership_flags: list[str]
    nearest_foot: dict | None
    nearest_atv: dict | None
    centroid_lonlat: tuple[float, float]
    priority_score: float = 0.0


def _load(path: Path):
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        nodata = ds.nodata
        transform = ds.transform
        crs = ds.crs
    return arr, nodata, transform, crs


def _nearest_access(cell_row: int, cell_col: int, minutes_arr: np.ndarray, nearest_seed_arr: np.ndarray,
                     seed_meta: list[dict], elev: np.ndarray) -> dict | None:
    m = minutes_arr[cell_row, cell_col]
    seed_idx = nearest_seed_arr[cell_row, cell_col]
    if m < 0 or seed_idx < 0 or seed_idx >= len(seed_meta):
        return None
    seed = seed_meta[seed_idx]
    vert_gain_m = float(elev[cell_row, cell_col] - elev[seed["row"], seed["col"]])
    return {
        "name": seed["name"],
        "one_way_minutes": round(float(m), 1),
        "vert_gain_m": round(vert_gain_m, 0),
        "lon": seed["lon"],
        "lat": seed["lat"],
    }


def build_priority_zones(cfg: dict, habitat_path: Path, foot_minutes_path: Path, foot_nearest_seed_path: Path,
                          foot_seeds_path: Path, ownership_path: Path, landform_path: Path, dem_path: Path,
                          atv_minutes_path: Path | None = None, atv_nearest_seed_path: Path | None = None,
                          atv_seeds_path: Path | None = None) -> list[Zone]:
    pcfg = cfg["priority"]
    max_effort = cfg["run"]["max_effort_minutes"]

    habitat, hab_nodata, transform, crs = _load(habitat_path)
    foot_minutes, foot_nodata, _, _ = _load(foot_minutes_path)
    nearest_seed, _, _, _ = _load(foot_nearest_seed_path)
    ownership_mask, _, _, _ = _load(ownership_path)
    landform, landform_nodata, _, _ = _load(landform_path)
    with rasterio.open(dem_path) as ds:
        elev = ds.read(1).astype("float64")

    # Union across every species' overlap raster present (dwr_overlap_elk.tif,
    # dwr_overlap_deer.tif, ...) -- a zone counts as DWR-overlapping if it overlaps
    # for any scored species.
    dwr_overlap = None
    for p in sorted(habitat_path.parent.glob("dwr_overlap_*.tif")):
        arr, _, _, _ = _load(p)
        dwr_overlap = arr if dwr_overlap is None else np.maximum(dwr_overlap, arr)

    foot_seeds = json.loads(foot_seeds_path.read_text())

    atv_minutes = atv_nearest = atv_seeds = None
    if atv_minutes_path and atv_minutes_path.exists():
        atv_minutes, _, _, _ = _load(atv_minutes_path)
        atv_nearest, _, _, _ = _load(atv_nearest_seed_path)
        atv_seeds = json.loads(atv_seeds_path.read_text())

    candidate = (
        (habitat != hab_nodata) & (habitat >= pcfg["habitat_score_threshold"]) &
        (foot_minutes != foot_nodata) & (foot_minutes >= 0) & (foot_minutes <= max_effort) &
        (ownership_mask != own.EXCLUDED)
    )
    n_candidate = int(candidate.sum())
    log.info("Candidate cells passing habitat/effort/ownership filters: %d (%.2f%% of AOI)",
              n_candidate, 100 * n_candidate / candidate.size)
    if n_candidate == 0:
        log.warning("No cells passed the priority filters -- lower priority.habitat_score_threshold "
                    "or raise run.max_effort_minutes in config.yaml and rerun Phase 2.")
        return []

    structure = np.ones((3, 3), dtype="uint8")  # 8-connectivity
    labeled, n_labels = label(candidate, structure=structure)
    log.info("Clustered into %d connected component(s) before size filtering", n_labels)
    slices = find_objects(labeled)

    pixel_area_m2 = abs(transform.a * transform.e)
    landform_names = {0: "drainage bottom", 1: "side slope", 3: "bench/saddle/ridge"}

    import geopandas as gpd

    zones = []
    for lid in range(1, n_labels + 1):
        box = slices[lid - 1]
        if box is None:
            continue
        row_off, col_off = box[0].start, box[1].start
        local_mask = labeled[box] == lid
        ncells = int(local_mask.sum())
        if ncells < pcfg["min_zone_cells"]:
            continue

        local_habitat = habitat[box]
        local_minutes = foot_minutes[box]
        local_landform = landform[box]
        local_ownership = ownership_mask[box]

        mean_hab = float(local_habitat[local_mask].mean())
        mean_min = float(local_minutes[local_mask].mean())
        min_min = float(local_minutes[local_mask].min())
        acres = ncells * pixel_area_m2 * ACRES_PER_CELL_FACTOR

        lf_vals, lf_counts = np.unique(local_landform[local_mask], return_counts=True)
        dominant_lf_code = int(lf_vals[np.argmax(lf_counts)]) if len(lf_vals) else -1
        dominant_lf = landform_names.get(dominant_lf_code, "mixed")

        ownership_flags = []
        own_vals = set(np.unique(local_ownership[local_mask]).tolist())
        if own.SITLA in own_vals:
            ownership_flags.append("SITLA")
        if own.UNKNOWN in own_vals:
            ownership_flags.append("OWNERSHIP UNVERIFIED")

        rows, cols = np.where(local_mask)
        best_idx = np.argmin(local_minutes[local_mask])
        best_row, best_col = int(rows[best_idx]) + row_off, int(cols[best_idx]) + col_off

        nearest_foot = _nearest_access(best_row, best_col, foot_minutes, nearest_seed, foot_seeds, elev)
        nearest_atv = None
        if atv_minutes is not None:
            nearest_atv = _nearest_access(best_row, best_col, atv_minutes, atv_nearest, atv_seeds, elev)

        local_transform = transform * transform.translation(col_off, row_off)
        polys = [shapely_shape(geom) for geom, val in rio_shapes(local_mask.astype("uint8"), mask=local_mask, transform=local_transform) if val == 1]
        geom = polys[0] if len(polys) == 1 else unary_union(polys)

        centroid_working = geom.centroid
        centroid_lonlat = gpd.GeoSeries([centroid_working], crs=crs).to_crs("EPSG:4326").iloc[0]

        dwr_overlap_pct = float(100 * dwr_overlap[box][local_mask].mean()) if dwr_overlap is not None else 0.0

        zones.append(Zone(
            rank=0, zone_id=lid, geometry=geom, acres=round(acres, 1), mean_habitat=round(mean_hab, 3),
            mean_foot_minutes=round(mean_min, 1), min_foot_minutes=round(min_min, 1),
            dominant_landform=dominant_lf, dwr_overlap_pct=dwr_overlap_pct, ownership_flags=ownership_flags,
            nearest_foot=nearest_foot, nearest_atv=nearest_atv,
            centroid_lonlat=(centroid_lonlat.x, centroid_lonlat.y),
        ))

    # priority_score rewards good habitat AND being hard to reach (up to the budget)
    for z in zones:
        effort_fraction = min(z.mean_foot_minutes / max_effort, 1.0)
        z.priority_score = z.mean_habitat * (0.5 + 0.5 * effort_fraction)

    zones.sort(key=lambda z: z.priority_score, reverse=True)
    zones = zones[: pcfg["top_n"]]
    for i, z in enumerate(zones, start=1):
        z.rank = i

    log.info("Kept top %d priority zone(s) after ranking", len(zones))
    return zones

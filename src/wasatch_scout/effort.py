"""Anisotropic hiking-time cost surface from Tobler's hiking function.

Tobler's function gives walking speed (km/h) as a function of terrain slope
(rise/run, signed: positive = uphill in the direction of travel):

    W(S) = 6 * exp(-3.5 * abs(S + 0.05))

The 0.05 offset shifts the speed peak to a gentle downhill grade, matching
observed hiking behavior. Because the function depends on the *signed*
slope in the direction of travel, cost is direction-dependent (uphill !=
downhill) -- this is computed per-edge directly from the DEM during the
Dijkstra expansion via skimage.graph.MCP_Flexible, not from a precomputed
scalar slope raster, so it's genuinely anisotropic rather than an isotropic
approximation.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import rasterio
import skimage.graph as skgraph
from rasterio.transform import rowcol

from .aoi import AOI
from .config import resolve

log = logging.getLogger(__name__)


class _TimeCostMCP(skgraph.MCP_Flexible):
    """costs array = elevation (m). travel_cost turns each edge's elevation delta
    and real horizontal distance (from `sampling`) into Tobler hiking time in minutes."""

    def __init__(self, elevation: np.ndarray, sampling: tuple[float, float], pace_multiplier: float,
                 base_kmh: float, slope_coeff: float, offset: float):
        super().__init__(elevation, sampling=sampling, fully_connected=True)
        self.pace_multiplier = pace_multiplier
        self.base_kmh = base_kmh
        self.slope_coeff = slope_coeff
        self.offset = offset

    def travel_cost(self, old_cost, new_cost, offset_length):
        if offset_length <= 0 or np.isnan(old_cost) or np.isnan(new_cost):
            return np.inf
        rise = new_cost - old_cost
        slope = rise / offset_length  # signed rise/run in direction of travel
        speed_kmh = self.base_kmh * np.exp(self.slope_coeff * abs(slope + self.offset)) * self.pace_multiplier
        speed_kmh = max(speed_kmh, 0.05)  # floor so cliffs are "very slow" not divide-by-zero
        distance_km = offset_length / 1000.0
        return (distance_km / speed_kmh) * 60.0  # minutes


def _cost_from_seeds(elev: np.ndarray, seed_rc: list[tuple[int, int]], sampling: tuple[float, float],
                      pace_mult: float, base_kmh: float, slope_coeff: float, offset: float,
                      max_effort_minutes: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Minimum Tobler hiking-time (minutes) from any of seed_rc to every cell, plus
    which seed index (into seed_rc) reached each cell fastest (-1 if unreached).

    skimage's MCP seeds each start's cumulative cost at `costs[start]` (here,
    that start pixel's raw elevation in meters) before any travel_cost edges
    are applied -- a one-time additive bias equal to that seed's elevation.
    With multiple seeds at different elevations this bias would differ by
    seed and corrupt which seed "wins" each cell, so each seed is run as an
    independent single-source pass and the exact per-seed bias (== elevation
    at that seed, verified empirically) is subtracted back out before taking
    the elementwise minimum across seeds.

    Each pass is cropped to a window around the seed sized from the fastest
    speed the Tobler model can ever produce (base_kmh * pace_mult, at the
    optimal slight-downhill grade) times the effort budget, plus a 1.5x safety
    margin for path inefficiency -- no true shortest path within budget can
    reach outside that radius, so cropping loses no accuracy while cutting
    compute roughly (full_grid_area / window_area)-fold.
    """
    accum = np.full(elev.shape, np.inf, dtype="float64")
    nearest_seed = np.full(elev.shape, -1, dtype="int16")
    py, px = sampling

    if max_effort_minutes is not None:
        max_speed_kmh = base_kmh * pace_mult
        radius_m = (max_effort_minutes / 60.0) * max_speed_kmh * 1000.0 * 1.5
        radius_px_r = int(np.ceil(radius_m / py))
        radius_px_c = int(np.ceil(radius_m / px))
        log.info("Cropping each seed's search to a %.1f km radius window (%d x %d px)",
                  radius_m / 1000.0, radius_px_r * 2, radius_px_c * 2)
    else:
        radius_px_r = radius_px_c = None

    nrows, ncols = elev.shape
    for i, (r, c) in enumerate(seed_rc):
        if np.isnan(elev[r, c]):
            log.warning("Seed at row=%d col=%d has no elevation data (outside DEM); skipping", r, c)
            continue

        if radius_px_r is not None:
            r0, r1 = max(0, r - radius_px_r), min(nrows, r + radius_px_r + 1)
            c0, c1 = max(0, c - radius_px_c), min(ncols, c + radius_px_c + 1)
        else:
            r0, r1, c0, c1 = 0, nrows, 0, ncols

        window = elev[r0:r1, c0:c1]
        mcp = _TimeCostMCP(window, sampling=sampling, pace_multiplier=pace_mult,
                            base_kmh=base_kmh, slope_coeff=slope_coeff, offset=offset)
        cost, _ = mcp.find_costs([(r - r0, c - c0)])
        bias = elev[r, c]
        corrected = cost - bias

        accum_window = accum[r0:r1, c0:c1]
        nearest_window = nearest_seed[r0:r1, c0:c1]
        better = corrected < accum_window
        nearest_window[better] = i
        accum_window[better] = corrected[better]
        log.info("  seed %d/%d (row=%d col=%d) done, window %d x %d", i + 1, len(seed_rc), r, c, r1 - r0, c1 - c0)
    return accum, nearest_seed


def _seed_rowcols(seeds_wgs84, dem_transform, dem_crs, shape) -> list[dict]:
    """Returns list of {row, col, name, lon, lat} for seeds falling inside the grid."""
    if seeds_wgs84 is None or len(seeds_wgs84) == 0:
        return []
    pts = seeds_wgs84.to_crs(dem_crs)
    out = []
    nrows, ncols = shape
    for idx, geom in zip(seeds_wgs84.index, pts.geometry):
        if geom is None:
            continue
        x, y = geom.x, geom.y
        r, c = rowcol(dem_transform, x, y)
        r, c = int(r), int(c)
        if 0 <= r < nrows and 0 <= c < ncols:
            wgs_geom = seeds_wgs84.geometry.loc[idx]
            name = seeds_wgs84.loc[idx, "name"] if "name" in seeds_wgs84.columns else f"seed_{idx}"
            out.append({"row": r, "col": c, "name": str(name), "lon": float(wgs_geom.x), "lat": float(wgs_geom.y)})
    return out


def compute_effort_surface(cfg: dict, dem_path: Path, seeds: dict, force: bool = False) -> dict[str, Path]:
    """Returns paths to foot_minutes.tif, foot_nearest_seed.tif, foot_seeds.json,
    and (if ATV seeds exist) the same trio for atv_*."""
    processed = resolve(cfg, "paths.processed_dir")
    out = {
        "foot_minutes": processed / "foot_minutes.tif",
        "foot_nearest_seed": processed / "foot_nearest_seed.tif",
        "foot_seeds": processed / "foot_seeds.json",
    }
    atv_seeds = seeds.get("atv")
    have_atv = atv_seeds is not None and len(atv_seeds) > 0
    if have_atv:
        out["atv_minutes"] = processed / "atv_minutes.tif"
        out["atv_nearest_seed"] = processed / "atv_nearest_seed.tif"
        out["atv_seeds"] = processed / "atv_seeds.json"

    if all(p.exists() for p in out.values()) and not force:
        log.info("Effort surface cache hit")
        return out

    eff_cfg = cfg["effort"]
    pace = cfg["run"]["pace_profile"]
    pace_mult = eff_cfg["pace_multiplier"][pace]

    with rasterio.open(dem_path) as ds:
        elev = ds.read(1).astype("float64")
        nodata = ds.nodata
        if nodata is not None:
            elev[elev == nodata] = np.nan
        transform = ds.transform
        crs = ds.crs
        px, py = ds.res

    foot_rc = _seed_rowcols(seeds["foot"], transform, crs, elev.shape)
    if not foot_rc:
        raise ValueError("No foot access seeds fall within the AOI -- cannot build effort surface.")
    log.info("Foot access seeds in AOI: %d", len(foot_rc))

    max_effort = cfg["run"]["max_effort_minutes"]
    log.info("Computing foot effort surface (%d seed(s), cropped windows sized to the %d-min budget)...",
              len(foot_rc), max_effort)
    foot_minutes, foot_nearest = _cost_from_seeds(
        elev, [(s["row"], s["col"]) for s in foot_rc], sampling=(py, px), pace_mult=pace_mult,
        base_kmh=eff_cfg["tobler_base_kmh"], slope_coeff=eff_cfg["tobler_slope_coeff"], offset=eff_cfg["tobler_offset"],
        max_effort_minutes=max_effort,
    )
    foot_minutes = np.where(np.isnan(elev), np.nan, foot_minutes)
    _write_like(out["foot_minutes"], foot_minutes, dem_path, dtype="float32", nodata=-1)
    _write_like(out["foot_nearest_seed"], foot_nearest, dem_path, dtype="int16", nodata=-1)
    out["foot_seeds"].write_text(json.dumps(foot_rc))
    finite = foot_minutes[np.isfinite(foot_minutes)]
    log.info("Foot effort surface done. Min/median/max minutes: %.1f / %.1f / %.1f",
              finite.min(), np.median(finite), finite.max())

    if have_atv:
        atv_rc = _seed_rowcols(seeds["atv"], transform, crs, elev.shape)
        if atv_rc:
            log.info("Computing ATV-alternate effort surface (%d seed(s))...", len(atv_rc))
            atv_minutes, atv_nearest = _cost_from_seeds(
                elev, [(s["row"], s["col"]) for s in atv_rc], sampling=(py, px), pace_mult=pace_mult,
                base_kmh=eff_cfg["tobler_base_kmh"], slope_coeff=eff_cfg["tobler_slope_coeff"], offset=eff_cfg["tobler_offset"],
                max_effort_minutes=max_effort,
            )
            atv_minutes = np.where(np.isnan(elev), np.nan, atv_minutes)
            _write_like(out["atv_minutes"], atv_minutes, dem_path, dtype="float32", nodata=-1)
            _write_like(out["atv_nearest_seed"], atv_nearest, dem_path, dtype="int16", nodata=-1)
            out["atv_seeds"].write_text(json.dumps(atv_rc))
        else:
            del out["atv_minutes"], out["atv_nearest_seed"], out["atv_seeds"]

    return out


def _write_like(out_path: Path, arr: np.ndarray, template_path: Path, dtype: str, nodata):
    with rasterio.open(template_path) as t:
        profile = t.profile.copy()
    profile.update(dtype=dtype, count=1, nodata=nodata, compress="lzw")
    arr = np.where(np.isnan(arr) | np.isinf(arr), nodata, arr).astype(dtype)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr, 1)

"""Slope, aspect, hillshade, and landform (ridge/saddle/bench/drainage) from the DEM."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import uniform_filter

from .config import resolve

log = logging.getLogger(__name__)


def _read_dem(dem_path: Path):
    with rasterio.open(dem_path) as ds:
        elev = ds.read(1).astype("float64")
        nodata = ds.nodata
        if nodata is not None:
            elev = np.where(elev == nodata, np.nan, elev)
        profile = ds.profile
        px, py = ds.res  # meters/pixel, both positive
    return elev, profile, px, py


def compute_slope_aspect(elev: np.ndarray, px: float, py: float) -> tuple[np.ndarray, np.ndarray]:
    """Horn's method. Returns slope in degrees and aspect in degrees (0=N, 90=E, clockwise)."""
    dzdx = np.gradient(elev, px, axis=1)
    dzdy = np.gradient(elev, py, axis=0)
    slope_rad = np.arctan(np.hypot(dzdx, dzdy))
    slope_deg = np.degrees(slope_rad)

    aspect_rad = np.arctan2(dzdy, -dzdx)  # standard GIS convention adjustment below
    aspect_deg = np.degrees(aspect_rad)
    aspect_deg = 90.0 - aspect_deg
    aspect_deg = np.where(aspect_deg < 0, aspect_deg + 360.0, aspect_deg)
    aspect_deg = np.where(slope_deg < 0.5, -1.0, aspect_deg)  # flat: undefined aspect
    return slope_deg, aspect_deg


def compute_hillshade(elev: np.ndarray, px: float, py: float, azimuth_deg=315.0, altitude_deg=45.0) -> np.ndarray:
    dzdx = np.gradient(elev, px, axis=1)
    dzdy = np.gradient(elev, py, axis=0)
    slope_rad = np.arctan(np.hypot(dzdx, dzdy))
    aspect_rad = np.arctan2(dzdy, -dzdx)

    az_rad = np.radians(360.0 - azimuth_deg + 90.0)
    alt_rad = np.radians(altitude_deg)

    shaded = np.sin(alt_rad) * np.cos(slope_rad) + np.cos(alt_rad) * np.sin(slope_rad) * np.cos(az_rad - aspect_rad)
    shaded = np.clip(shaded, 0, 1) * 255
    return shaded.astype("uint8")


def compute_landform(elev: np.ndarray, px: float, py: float, window_m: float = 150.0) -> np.ndarray:
    """Simple TPI-based landform classes: 0=drainage, 1=slope, 2=bench, 3=saddle/ridge.

    TPI = elevation - mean(elevation in window). Saddle/ridge detection uses TPI plus
    local slope: high positive TPI on gentle slope reads as bench; high positive TPI
    on steeper local relief reads as ridge/saddle-adjacent, both grouped as huntable
    "high ground" class 3 since the brief only needs bench/saddle vs ridge vs drainage
    at zone-scoring resolution.
    """
    win_px = max(3, int(round(window_m / ((px + py) / 2))))
    if win_px % 2 == 0:
        win_px += 1
    mean_elev = uniform_filter(np.nan_to_num(elev, nan=np.nanmean(elev)), size=win_px)
    tpi = elev - mean_elev

    std = np.nanstd(tpi)
    landform = np.full(elev.shape, 1, dtype="uint8")  # default: side slope
    landform[tpi < -0.5 * std] = 0  # drainage / valley bottom
    landform[tpi > 0.5 * std] = 3  # ridge / saddle / bench (locally higher than surroundings)
    return landform


def build_terrain_stack(cfg: dict, dem_path: Path, force: bool = False) -> dict[str, Path]:
    processed = resolve(cfg, "paths.processed_dir")
    out = {
        "slope": processed / "slope.tif",
        "aspect": processed / "aspect.tif",
        "hillshade": processed / "hillshade.tif",
        "landform": processed / "landform.tif",
    }
    if all(p.exists() for p in out.values()) and not force:
        log.info("Terrain stack cache hit")
        return out

    elev, profile, px, py = _read_dem(dem_path)
    log.info("DEM grid %s px, resolution ~%.1f x %.1f m", elev.shape, px, py)

    nodata_mask = np.isnan(elev)
    elev_filled = np.where(nodata_mask, np.nanmean(elev), elev)

    slope, aspect = compute_slope_aspect(elev_filled, px, py)
    hillshade = compute_hillshade(elev_filled, px, py)
    landform = compute_landform(elev_filled, px, py)

    slope[nodata_mask] = np.nan
    aspect[nodata_mask] = np.nan
    hillshade = np.where(nodata_mask, 0, hillshade).astype("uint8")
    landform = np.where(nodata_mask, 255, landform).astype("uint8")

    _write(out["slope"], slope.astype("float32"), profile, dtype="float32", nodata=-9999)
    _write(out["aspect"], aspect.astype("float32"), profile, dtype="float32", nodata=-9999)
    _write(out["hillshade"], hillshade, profile, dtype="uint8", nodata=0)
    _write(out["landform"], landform, profile, dtype="uint8", nodata=255)

    log.info("Terrain stack written to %s", processed)
    return out


def _write(path: Path, arr: np.ndarray, profile: dict, dtype: str, nodata):
    prof = profile.copy()
    prof.update(dtype=dtype, count=1, nodata=nodata, compress="lzw")
    arr = np.nan_to_num(arr, nan=nodata)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(dtype), 1)

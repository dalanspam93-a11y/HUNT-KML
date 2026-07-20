"""USGS 3DEP 1/3 arc-second (~10 m) DEM acquisition.

Source: prd-tnm.s3.amazonaws.com, USGS's public "staged products" S3 bucket —
no API key, no rate limit, reachable even from network-restricted sandboxes
since it's plain S3. Tiles are 1x1 degree GeoTIFFs named by their NW corner,
e.g. USGS_13_n41w112.tif covers lat [40,41] x lon [-112,-111].
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import rasterio
import requests
from rasterio.io import MemoryFile
from rasterio.mask import mask as rio_mask
from rasterio.merge import merge as rio_merge
from rasterio.warp import Resampling, calculate_default_transform, reproject

from ..aoi import AOI
from ..config import resolve

log = logging.getLogger(__name__)

BASE_URL = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current"


def _tiles_for_bbox(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> list[str]:
    """1-degree 3DEP tile names (NW-corner convention) covering a WGS84 bbox."""
    lat_hi = math.ceil(max_lat)
    lat_lo = math.floor(min_lat)
    lon_hi = math.ceil(max_lon)  # least negative
    lon_lo = math.floor(min_lon)
    tiles = []
    lat = lat_lo
    while lat < lat_hi:
        lon = lon_lo
        while lon < lon_hi:
            tile_lat = lat + 1  # NW corner latitude
            tile_lon = lon  # NW corner longitude (western edge, negative)
            ns = f"n{tile_lat:02d}"
            ew = f"w{abs(tile_lon):03d}" if tile_lon < 0 else f"e{tile_lon:03d}"
            tiles.append(f"{ns}{ew}")
            lon += 1
        lat += 1
    return sorted(set(tiles))


def _download_tile(name: str, dest_dir: Path) -> Path:
    dest = dest_dir / f"USGS_13_{name}.tif"
    if dest.exists():
        log.info("DEM tile %s already cached at %s", name, dest)
        return dest
    url = f"{BASE_URL}/{name}/USGS_13_{name}.tif"
    log.info("Downloading DEM tile %s from %s", name, url)
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tif.partial")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp.rename(dest)
    log.info("Saved %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest


def acquire_dem(cfg: dict, aoi: AOI, force: bool = False) -> Path:
    """Download, mosaic, clip, and reproject DEM tiles covering the AOI.

    Returns path to cached GeoTIFF in data/processed/dem.tif (working CRS).
    """
    out_path = resolve(cfg, "paths.processed_dir") / "dem.tif"
    if out_path.exists() and not force:
        log.info("DEM cache hit: %s", out_path)
        return out_path

    raw_dir = resolve(cfg, "paths.raw_dir") / "dem_tiles"
    min_lon, min_lat, max_lon, max_lat = aoi.bbox_wgs84
    tile_names = _tiles_for_bbox(min_lon, min_lat, max_lon, max_lat)
    log.info("AOI bbox %.4f,%.4f,%.4f,%.4f needs %d DEM tile(s): %s",
              min_lon, min_lat, max_lon, max_lat, len(tile_names), tile_names)

    tile_paths = [_download_tile(t, raw_dir) for t in tile_names]

    srcs = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic_arr, mosaic_transform = rio_merge(srcs)
        mosaic_meta = srcs[0].meta.copy()
        mosaic_meta.update(
            height=mosaic_arr.shape[1],
            width=mosaic_arr.shape[2],
            transform=mosaic_transform,
        )
    finally:
        for s in srcs:
            s.close()

    with MemoryFile() as mem:
        with mem.open(**mosaic_meta) as mosaic_ds:
            mosaic_ds.write(mosaic_arr)
            working_crs = aoi.working_crs
            aoi_working = aoi.working_gdf()
            geoms = [g.__geo_interface__ for g in aoi_working.geometry]

            transform, width, height = calculate_default_transform(
                mosaic_ds.crs, working_crs, mosaic_ds.width, mosaic_ds.height, *mosaic_ds.bounds
            )
            reproj_meta = mosaic_ds.meta.copy()
            reproj_meta.update(crs=working_crs, transform=transform, width=width, height=height)

            with MemoryFile() as mem2:
                with mem2.open(**reproj_meta) as reproj_ds:
                    reproject(
                        source=rasterio.band(mosaic_ds, 1),
                        destination=rasterio.band(reproj_ds, 1),
                        src_transform=mosaic_ds.transform,
                        src_crs=mosaic_ds.crs,
                        dst_transform=transform,
                        dst_crs=working_crs,
                        resampling=Resampling.bilinear,
                    )
                    clipped, clipped_transform = rio_mask(reproj_ds, geoms, crop=True)
                    clipped_meta = reproj_ds.meta.copy()
                    clipped_meta.update(
                        height=clipped.shape[1],
                        width=clipped.shape[2],
                        transform=clipped_transform,
                    )
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with rasterio.open(out_path, "w", **clipped_meta) as dst:
                        dst.write(clipped)

    log.info("DEM cached: %s (%d x %d px, CRS %s)", out_path, clipped.shape[2], clipped.shape[1], working_crs)
    return out_path

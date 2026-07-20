"""Habitat score (0-1 per cell): weighted sum of aspect, cover/edge, landform,
and DWR crucial-range/migration overlap. Any missing input layer is dropped
from the sum with weights renormalized across whatever remains -- the score
is always computed from what's actually available, and the caller is told
exactly which terms were used so that's visible downstream (report + KML
honest-limitations text), never silently faked.
"""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

from .config import resolve

log = logging.getLogger(__name__)


def _aspect_score(aspect_deg: np.ndarray, center: float, halfwidth: float) -> np.ndarray:
    flat_mask = aspect_deg < 0
    diff = np.abs(((aspect_deg - center + 180) % 360) - 180)  # angular distance, 0-180
    score = np.clip(1.0 - diff / halfwidth, 0.0, 1.0)
    score = np.where(flat_mask, 0.5, score)  # undefined aspect (flat ground): neutral, not penalized
    return score


def _landform_score(landform: np.ndarray) -> np.ndarray:
    score = np.full(landform.shape, np.nan)
    score[landform == 3] = 1.0   # bench / saddle / ridge-adjacent high ground
    score[landform == 1] = 0.6   # open side slope
    score[landform == 0] = 0.4   # drainage bottom
    return score


def _cover_edge_score(landcover_path: Path | None, template_path: Path, forest_classes, shrub_classes,
                       edge_radius_m: float) -> np.ndarray | None:
    if landcover_path is None or not landcover_path.exists():
        return None
    from scipy.ndimage import distance_transform_edt, binary_dilation

    with rasterio.open(template_path) as t:
        template_shape = (t.height, t.width)
        template_transform = t.transform
        template_crs = t.crs
        px, py = t.res

    with rasterio.open(landcover_path) as lc:
        from rasterio.warp import Resampling, reproject
        lc_arr = np.zeros(template_shape, dtype="int32")
        reproject(
            source=rasterio.band(lc, 1), destination=lc_arr,
            src_transform=lc.transform, src_crs=lc.crs,
            dst_transform=template_transform, dst_crs=template_crs,
            resampling=Resampling.nearest,
        )

    forest_mask = np.isin(lc_arr, forest_classes)
    open_mask = ~forest_mask  # everything not forest, incl. shrub, counts as "open" for edge purposes

    # distance (m) from each forest cell to nearest open cell, and vice versa
    dist_to_open = distance_transform_edt(forest_mask, sampling=(py, px))
    dist_to_forest = distance_transform_edt(open_mask, sampling=(py, px))
    dist_to_edge = np.where(forest_mask, dist_to_open, dist_to_forest)

    score = np.clip(1.0 - dist_to_edge / edge_radius_m, 0.0, 1.0)
    return score


def _dwr_overlap_score(dwr_layers: dict, template_path: Path) -> np.ndarray | None:
    polys = [gdf for gdf in dwr_layers.values() if gdf is not None and len(gdf) > 0]
    if not polys:
        return None
    with rasterio.open(template_path) as t:
        shape = (t.height, t.width)
        transform = t.transform
        crs = t.crs

    combined = None
    for gdf in polys:
        g = gdf.to_crs(crs)
        r = rasterize(
            [(geom, 1) for geom in g.geometry if geom is not None and not geom.is_empty],
            out_shape=shape, transform=transform, fill=0, dtype="uint8",
        )
        combined = r if combined is None else np.maximum(combined, r)
    return combined.astype("float64")


def compute_habitat_score(cfg: dict, species: str, terrain_paths: dict, landcover_path: Path | None,
                           dwr_layers: dict, force: bool = False) -> Path:
    processed = resolve(cfg, "paths.processed_dir")
    out_path = processed / f"habitat_score_{species}.tif"
    if out_path.exists() and not force:
        log.info("Habitat score (%s) cache hit: %s", species, out_path)
        return out_path

    weights_cfg = cfg["habitat_weights"]
    aspect_cfg = cfg["aspect_preference_degrees"][species]

    with rasterio.open(terrain_paths["aspect"]) as ds:
        aspect = ds.read(1)
        profile = ds.profile.copy()
        aspect_nodata = ds.nodata
    aspect = np.where(aspect == aspect_nodata, np.nan, aspect)

    with rasterio.open(terrain_paths["landform"]) as ds:
        landform = ds.read(1)
        landform_nodata = ds.nodata
    valid_mask = landform != landform_nodata

    terms = {}
    terms["aspect"] = _aspect_score(aspect, aspect_cfg["center"], aspect_cfg["halfwidth"])
    terms["landform"] = _landform_score(landform)

    cover_edge_cfg = cfg["landcover"]
    cover_score = _cover_edge_score(
        landcover_path, terrain_paths["aspect"],
        cover_edge_cfg["forest_classes"], cover_edge_cfg["shrub_classes"], cover_edge_cfg["edge_search_radius_m"],
    )
    if cover_score is not None:
        terms["cover_edge"] = cover_score

    dwr_score = _dwr_overlap_score(dwr_layers, terrain_paths["aspect"])
    if dwr_score is not None:
        terms["dwr_overlap"] = dwr_score
        dwr_overlap_path = processed / "dwr_overlap.tif"
        if not dwr_overlap_path.exists() or force:
            with rasterio.open(terrain_paths["aspect"]) as t:
                dwr_profile = t.profile.copy()
            dwr_profile.update(dtype="uint8", nodata=255, compress="lzw")
            with rasterio.open(dwr_overlap_path, "w", **dwr_profile) as dst:
                dst.write(dwr_score.astype("uint8"), 1)

    missing = [k for k in weights_cfg if k not in terms]
    if missing:
        log.warning(
            "Habitat score (%s): missing input layer(s) %s -- dropping their weight and "
            "renormalizing across the remaining terms %s. Score is a PARTIAL habitat "
            "signal until these layers are supplied; note this in any output you rely on.",
            species, missing, list(terms.keys()),
        )
    active_weight_sum = sum(weights_cfg[k] for k in terms)
    if active_weight_sum == 0:
        raise ValueError("No habitat score terms available -- need at least aspect/landform from the DEM.")

    score = np.zeros(aspect.shape, dtype="float64")
    for k, arr in terms.items():
        w = weights_cfg[k] / active_weight_sum
        score += w * np.nan_to_num(arr, nan=0.0)

    score = np.where(valid_mask, score, np.nan)

    profile.update(dtype="float32", nodata=-1, compress="lzw")
    out_arr = np.where(np.isnan(score), -1, score).astype("float32")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out_arr, 1)

    log.info("Habitat score (%s) written using terms %s: %s", species, list(terms.keys()), out_path)
    return out_path

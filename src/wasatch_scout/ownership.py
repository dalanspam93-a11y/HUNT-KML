"""Ownership mask: zero out private/water, pass public land, flag SITLA separately.

If no ownership layer is available, cells are marked UNKNOWN rather than
silently treated as public -- the brief's validation bar is "zero priority
zones on private/CWMU land," which we cannot guarantee without real parcel
data. Downstream, any zone touching UNKNOWN ownership must be flagged as
unverified rather than presented as clean public land.
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

EXCLUDED = 0
PUBLIC = 1
SITLA = 2
UNKNOWN = 255


def compute_ownership_mask(cfg: dict, ownership_gdf: gpd.GeoDataFrame | None, template_path: Path,
                            force: bool = False) -> Path:
    out_path = resolve(cfg, "paths.processed_dir") / "ownership_mask.tif"
    if out_path.exists() and not force:
        log.info("Ownership mask cache hit: %s", out_path)
        return out_path

    with rasterio.open(template_path) as t:
        shape = (t.height, t.width)
        transform = t.transform
        crs = t.crs
        profile = t.profile.copy()

    if ownership_gdf is None or len(ownership_gdf) == 0:
        log.warning(
            "No ownership/parcel data available -- ownership mask is entirely UNKNOWN. "
            "Priority zones will be flagged 'ownership unverified' rather than assumed "
            "public. Supply ownership.boundary_ownership_file (or a service URL) before "
            "trusting the 'zero zones on private/CWMU land' guarantee."
        )
        mask = np.full(shape, UNKNOWN, dtype="uint8")
    else:
        own_cfg = cfg["ownership"]
        class_field = _guess_class_field(ownership_gdf)
        gdf = ownership_gdf.to_crs(crs)
        mask = np.full(shape, UNKNOWN, dtype="uint8")

        for cls in own_cfg["allowed_classes"]:
            sel = gdf[gdf[class_field].astype(str).str.lower().str.contains(cls, na=False)]
            if len(sel):
                code = SITLA if cls == "sitla" else PUBLIC
                r = rasterize([(g, 1) for g in sel.geometry if g and not g.is_empty], out_shape=shape,
                              transform=transform, fill=0, dtype="uint8")
                mask = np.where(r == 1, code, mask)

        for cls in own_cfg["exclude_classes"]:
            sel = gdf[gdf[class_field].astype(str).str.lower().str.contains(cls, na=False)]
            if len(sel):
                r = rasterize([(g, 1) for g in sel.geometry if g and not g.is_empty], out_shape=shape,
                              transform=transform, fill=0, dtype="uint8")
                mask = np.where(r == 1, EXCLUDED, mask)

    profile.update(dtype="uint8", count=1, nodata=UNKNOWN, compress="lzw")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask, 1)
    log.info("Ownership mask written: %s (public=%.1f%% excluded=%.1f%% sitla=%.1f%% unknown=%.1f%%)",
              out_path, *_pct(mask))
    return out_path


def _guess_class_field(gdf: gpd.GeoDataFrame) -> str:
    for candidate in ("own_type", "ownership", "own_class", "class", "landowner", "mgmt_agency", "OWNER"):
        if candidate in gdf.columns:
            return candidate
    raise ValueError(
        f"Ownership layer has no recognizable ownership-class field (tried own_type/ownership/own_class/"
        f"class/landowner/mgmt_agency/OWNER; got columns {list(gdf.columns)}). Update ownership.py's "
        f"_guess_class_field or rename the field in the source export."
    )


def _pct(mask: np.ndarray) -> tuple[float, float, float, float]:
    total = mask.size
    return (
        100 * np.sum(mask == PUBLIC) / total,
        100 * np.sum(mask == EXCLUDED) / total,
        100 * np.sum(mask == SITLA) / total,
        100 * np.sum(mask == UNKNOWN) / total,
    )

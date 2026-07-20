"""Quick-look PNGs for Phase 1/2 review: hillshade + effort bands, and
hillshade + habitat score, so weights/thresholds can be sanity-checked
before spending time on a full KML export.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.plot
from matplotlib.colors import ListedColormap, BoundaryNorm

log = logging.getLogger(__name__)


def _read(path: Path):
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype("float64")
        nodata = ds.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        extent = rasterio.plot.plotting_extent(ds)
    return arr, extent


def render_effort_preview(hillshade_path: Path, foot_minutes_path: Path, max_effort: int, out_path: Path):
    hs, extent = _read(hillshade_path)
    minutes, _ = _read(foot_minutes_path)
    minutes = np.where(minutes < 0, np.nan, minutes)

    fig, ax = plt.subplots(figsize=(10, 12), dpi=140)
    ax.imshow(hs, cmap="gray", extent=extent, vmin=0, vmax=255)

    bounds = [0, 30, 60, 90, max(90, max_effort) + 1, np.nanmax(minutes) if np.isfinite(np.nanmax(minutes)) else 500]
    bounds = sorted(set(bounds))
    cmap = ListedColormap(["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#7f7f7f"][: len(bounds) - 1])
    norm = BoundaryNorm(bounds, cmap.N)
    im = ax.imshow(minutes, cmap=cmap, norm=norm, alpha=0.55, extent=extent)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, label="One-way hiking minutes from foot access")
    ax.set_title("Effort surface preview (hillshade + hiking-time bands)")
    ax.set_xticks([])
    ax.set_yticks([])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Effort preview PNG: %s", out_path)


def render_habitat_preview(hillshade_path: Path, habitat_path: Path, out_path: Path):
    hs, extent = _read(hillshade_path)
    hab, _ = _read(habitat_path)
    hab = np.where(hab < 0, np.nan, hab)

    fig, ax = plt.subplots(figsize=(10, 12), dpi=140)
    ax.imshow(hs, cmap="gray", extent=extent, vmin=0, vmax=255)
    im = ax.imshow(hab, cmap="viridis", vmin=0, vmax=1, alpha=0.55, extent=extent)
    fig.colorbar(im, ax=ax, shrink=0.6, label="Habitat score (0-1)")
    ax.set_title("Habitat score preview (hillshade + score)")
    ax.set_xticks([])
    ax.set_yticks([])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Habitat preview PNG: %s", out_path)

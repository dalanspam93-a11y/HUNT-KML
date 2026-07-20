#!/usr/bin/env python3
"""Phase 2: build habitat score (per species + blended), ownership mask, and
priority zones. Prints a summary table before anything gets exported to KML.
Rerun after retuning config.yaml weights -- habitat/ownership/priority always
recompute fresh (cheap, seconds), Phase 1's DEM/terrain/effort-surface stay
cached (expensive, not touched here).

Usage: PYTHONPATH=src python3 scripts/phase2_score.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wasatch_scout.config import load_config, ensure_dirs, resolve  # noqa: E402
from wasatch_scout.aoi import load_aoi  # noqa: E402
from wasatch_scout.acquire.dwr import acquire_dwr_habitat, acquire_ownership  # noqa: E402
from wasatch_scout.acquire.access_seeds import load_access_seeds  # noqa: E402
from wasatch_scout.habitat import compute_habitat_score  # noqa: E402
from wasatch_scout.ownership import compute_ownership_mask  # noqa: E402
from wasatch_scout.priority import build_priority_zones  # noqa: E402
from wasatch_scout.preview import render_habitat_preview  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("phase2")


def main():
    cfg = load_config()
    ensure_dirs(cfg)
    processed = resolve(cfg, "paths.processed_dir")

    aoi = load_aoi(cfg)
    dwr_layers = acquire_dwr_habitat(cfg, aoi)

    terrain_paths = {
        "aspect": processed / "aspect.tif",
        "landform": processed / "landform.tif",
        "hillshade": processed / "hillshade.tif",
    }
    landcover_path = processed / "landcover_evt.tif"
    landcover_path = landcover_path if landcover_path.exists() else None
    dem_path = processed / "dem.tif"

    species_list = cfg["run"]["species"]
    habitat_paths = {}
    for sp in species_list:
        # habitat scoring is cheap (array math on cached terrain rasters) -- always
        # recompute so a weight retune in config.yaml takes effect without needing
        # --force, per the brief's "retuning takes seconds" requirement.
        habitat_paths[sp] = compute_habitat_score(cfg, sp, terrain_paths, landcover_path, dwr_layers, force=True)

    # blended score across species (mean), used for priority ranking
    import numpy as np
    import rasterio
    if len(species_list) > 1:
        arrs = []
        profile = None
        for sp in species_list:
            with rasterio.open(habitat_paths[sp]) as ds:
                a = ds.read(1).astype("float64")
                nodata = ds.nodata
                profile = ds.profile.copy()
            arrs.append(np.where(a == nodata, np.nan, a))
        blended = np.nanmean(np.stack(arrs), axis=0)
        blended_path = processed / "habitat_score_blended.tif"
        profile.update(dtype="float32", nodata=-1)
        with rasterio.open(blended_path, "w", **profile) as dst:
            dst.write(np.where(np.isnan(blended), -1, blended).astype("float32"), 1)
        log.info("Blended habitat score (mean of %s): %s", species_list, blended_path)
    else:
        blended_path = habitat_paths[species_list[0]]

    ownership_gdf = acquire_ownership(cfg, aoi)
    ownership_path = compute_ownership_mask(cfg, ownership_gdf, dem_path, force=True)  # cheap; always reflect current config

    seeds = load_access_seeds(cfg, aoi)
    foot_minutes_path = processed / "foot_minutes.tif"
    foot_nearest_path = processed / "foot_nearest_seed.tif"
    foot_seeds_path = processed / "foot_seeds.json"
    atv_minutes_path = processed / "atv_minutes.tif"
    atv_nearest_path = processed / "atv_nearest_seed.tif"
    atv_seeds_path = processed / "atv_seeds.json"
    has_atv = atv_minutes_path.exists()

    zones = build_priority_zones(
        cfg, blended_path, foot_minutes_path, foot_nearest_path, foot_seeds_path, ownership_path,
        terrain_paths["landform"], dem_path,
        atv_minutes_path=atv_minutes_path if has_atv else None,
        atv_nearest_seed_path=atv_nearest_path if has_atv else None,
        atv_seeds_path=atv_seeds_path if has_atv else None,
    )

    preview_path = Path(cfg["_repo_root"]) / "output" / "phase2_habitat_preview.png"
    render_habitat_preview(terrain_paths["hillshade"], blended_path, preview_path)

    log.info("\n\n===== PHASE 2 SUMMARY =====")
    log.info("%-5s %-8s %-8s %-10s %-22s %-8s %s", "Rank", "Acres", "HabScr", "FootMin", "Landform", "DWR%", "Flags")
    for z in zones:
        log.info("%-5d %-8.1f %-8.2f %-10.1f %-22s %-8.0f %s",
                  z.rank, z.acres, z.mean_habitat, z.mean_foot_minutes, z.dominant_landform, z.dwr_overlap_pct,
                  ",".join(z.ownership_flags) or "-")

    n_sitla = sum(1 for z in zones if "SITLA" in z.ownership_flags)
    n_unverified = sum(1 for z in zones if "OWNERSHIP UNVERIFIED" in z.ownership_flags)
    log.info("\nZones: %d total | SITLA-flagged: %d | ownership-unverified: %d", len(zones), n_sitla, n_unverified)
    log.info("Habitat preview PNG: %s", preview_path)

    import pickle
    zones_path = processed / "priority_zones.pkl"
    with open(zones_path, "wb") as f:
        pickle.dump(zones, f)
    log.info("Zones cached: %s (used by Phase 3)", zones_path)


if __name__ == "__main__":
    main()

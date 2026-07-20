#!/usr/bin/env python3
"""Phase 1: acquire DEM, land cover, DWR habitat layers, ownership, and access
seeds; build the terrain stack and effort surface; render a hillshade +
effort-band preview PNG. Reports what downloaded vs. what's missing.

Usage: PYTHONPATH=src python3 scripts/phase1_acquire.py [--force]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wasatch_scout.config import load_config, ensure_dirs  # noqa: E402
from wasatch_scout.aoi import load_aoi  # noqa: E402
from wasatch_scout.acquire.dem import acquire_dem  # noqa: E402
from wasatch_scout.acquire.landcover import acquire_landcover  # noqa: E402
from wasatch_scout.acquire.dwr import acquire_dwr_habitat, acquire_ownership  # noqa: E402
from wasatch_scout.acquire.access_seeds import load_access_seeds  # noqa: E402
from wasatch_scout.terrain import build_terrain_stack  # noqa: E402
from wasatch_scout.effort import compute_effort_surface  # noqa: E402
from wasatch_scout.preview import render_effort_preview  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("phase1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Ignore caches, recompute everything")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)

    report = {}

    aoi = load_aoi(cfg)
    report["aoi_authoritative"] = aoi.is_authoritative
    report["aoi_bbox_wgs84"] = aoi.bbox_wgs84

    log.info("=== DEM ===")
    dem_path = acquire_dem(cfg, aoi, force=args.force)
    report["dem"] = str(dem_path)

    log.info("=== Terrain stack (slope/aspect/hillshade/landform) ===")
    terrain_paths = build_terrain_stack(cfg, dem_path, force=args.force)
    report["terrain"] = {k: str(v) for k, v in terrain_paths.items()}

    log.info("=== Land cover (LANDFIRE EVT) ===")
    landcover_path = acquire_landcover(cfg, aoi, force=args.force)
    report["landcover"] = str(landcover_path) if landcover_path else "MISSING (see log)"

    log.info("=== DWR habitat layers ===")
    dwr_layers = acquire_dwr_habitat(cfg, aoi)
    report["dwr_crucial_range"] = "present" if dwr_layers["crucial_range"] is not None else "MISSING (see log)"
    report["dwr_migration_corridor"] = "present" if dwr_layers["migration_corridor"] is not None else "MISSING (see log)"

    log.info("=== Ownership / SITLA ===")
    ownership_gdf = acquire_ownership(cfg, aoi)
    report["ownership"] = "present" if ownership_gdf is not None else "MISSING (see log) -- zones will be flagged OWNERSHIP UNVERIFIED"

    log.info("=== Access seeds (trailheads / ATV routes) ===")
    seeds = load_access_seeds(cfg, aoi)
    report["foot_seeds"] = f"{len(seeds['foot'])} ({'authoritative' if seeds['foot_authoritative'] else 'DEMO placeholders'})"
    report["atv_seeds"] = f"{len(seeds['atv'])} ({'authoritative' if seeds['atv_authoritative'] else 'DEMO placeholders'})"

    log.info("=== Effort surface (Tobler anisotropic cost-distance) ===")
    effort_paths = compute_effort_surface(cfg, dem_path, seeds, force=args.force)
    report["effort"] = {k: str(v) for k, v in effort_paths.items()}

    log.info("=== Preview PNG ===")
    preview_path = Path(cfg["_repo_root"]) / "output" / "phase1_effort_preview.png"
    render_effort_preview(terrain_paths["hillshade"], effort_paths["foot_minutes"], cfg["run"]["max_effort_minutes"], preview_path)
    report["preview_png"] = str(preview_path)

    log.info("\n\n===== PHASE 1 REPORT =====")
    for k, v in report.items():
        log.info("%s: %s", k, v)

    import json
    report_path = Path(cfg["_repo_root"]) / "output" / "phase1_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    log.info("Report saved: %s", report_path)


if __name__ == "__main__":
    main()

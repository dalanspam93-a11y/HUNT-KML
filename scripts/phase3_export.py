#!/usr/bin/env python3
"""Phase 3: export wasatch_scout_overlay.kml from Phase 2's cached zones.

Usage: PYTHONPATH=src python3 scripts/phase3_export.py
"""
from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wasatch_scout.config import load_config, ensure_dirs, resolve  # noqa: E402
from wasatch_scout.aoi import load_aoi  # noqa: E402
from wasatch_scout.acquire.access_seeds import load_access_seeds  # noqa: E402
from wasatch_scout.kml_export import build_kml  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("phase3")


def main():
    cfg = load_config()
    ensure_dirs(cfg)
    processed = resolve(cfg, "paths.processed_dir")

    zones_path = processed / "priority_zones.pkl"
    if not zones_path.exists():
        raise SystemExit("No priority_zones.pkl found -- run scripts/phase2_score.py first.")
    with open(zones_path, "rb") as f:
        zones = pickle.load(f)

    aoi = load_aoi(cfg)
    seeds = load_access_seeds(cfg, aoi)

    out_path = resolve(cfg, "paths.output_dir") / "wasatch_scout_overlay.kml"
    build_kml(
        cfg, zones,
        foot_minutes_path=processed / "foot_minutes.tif",
        habitat_path=processed / "habitat_score_blended.tif" if (processed / "habitat_score_blended.tif").exists()
        else processed / f"habitat_score_{cfg['run']['species'][0]}.tif",
        seeds=seeds,
        out_path=out_path,
        aoi_authoritative=aoi.is_authoritative,
        seeds_authoritative=seeds["foot_authoritative"],
    )
    log.info("Done: %s", out_path)


if __name__ == "__main__":
    main()

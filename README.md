# Wasatch Scouting-Score Overlay

Terrain-aware scouting-score overlay for the **Wasatch Mountains, West** unit (Utah DWR
unit 17A), exported as a single KML to drop on top of OnX/GOHUNT. This does **not**
rebuild ownership, roads, or trails, and does **not** claim to show animal locations —
see the honest-limitations block baked into the KML's top-folder description.

Core idea: score the intersection of two rasters —

1. **Effort surface** — anisotropic, slope-aware hiking-TIME cost (Tobler's hiking
   function over the real DEM) from every motorized-legal access point / trailhead.
2. **Habitat quality** — aspect, forest cover/edge, bench/saddle landform, and DWR
   crucial-range/migration overlap, for an October elk/deer transition.

Priority zones = good habitat that's hard for the crowd to reach within *your* effort
budget.

## Run parameters (this run)

See `config.yaml` `run:` block — species (elk + deer, blended), general rifle + archery,
ATV available but treated as a secondary/alternate access note (foot trailheads drive
the primary effort score), 90-minute one-way foot budget, average pace.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running the phases

```bash
source .venv/bin/activate
PYTHONPATH=src python3 scripts/phase1_acquire.py   # DEM, land cover, DWR, access seeds, terrain, effort surface
PYTHONPATH=src python3 scripts/phase2_score.py      # habitat score, ownership mask, priority zones, summary table
PYTHONPATH=src python3 scripts/phase3_export.py     # wasatch_scout_overlay.kml
```

Every derived raster is cached as GeoTIFF under `data/processed/`. Retuning
`habitat_weights` (or `priority.*` thresholds) in `config.yaml` and rerunning
`phase2_score.py` takes seconds — it never re-downloads or re-runs the effort surface
(the expensive step, since it's a genuine anisotropic cost-distance over the whole DEM).
Pass `--force` to any phase script to ignore caches.

## IMPORTANT: network-restricted sandbox caveat

This pipeline was built and partly run inside a sandboxed Claude Code session whose
outbound network is locked to an allowlist (npm/PyPI/GitHub/**AWS S3** only) — direct
`.gov` and ArcGIS Online domains are blocked. That happens to be fine for the DEM (USGS
hosts 3DEP tiles as plain public GeoTIFFs on S3, `prd-tnm.s3.amazonaws.com`, which *is*
reachable), but it blocks:

- **Land cover** (LANDFIRE Product Service, `lfps.usgs.gov`)
- **DWR habitat layers** (crucial range, migration corridors) and **unit boundary**
  (`dwr-data-utahdnr.hub.arcgis.com`, ArcGIS Online)
- **Ownership/SITLA parcels** (`gis.utah.gov`, UGRC)

The corresponding modules (`src/wasatch_scout/acquire/landcover.py`, `dwr.py`) contain
real, correct implementations of the documented public APIs for these sources — they
should work unmodified once run somewhere with normal internet access (your machine, or
a less-restricted session). Until then:

- **Habitat scoring degrades gracefully**: missing terms (cover/edge, DWR overlap) are
  dropped and remaining weights renormalized — never faked. The log and any report say
  exactly which terms were used.
- **Ownership defaults to UNKNOWN, never to "assumed public."** Any zone touching
  unverified ownership is flagged `OWNERSHIP UNVERIFIED` in its balloon rather than
  silently presented as clean public land.
- **Unit boundary and access seeds fall back to explicitly-labeled placeholders** (an
  approximate bbox from public unit-description text, and a handful of real canyon-mouth
  trailhead coordinates) purely so the effort-surface and clustering algorithms can be
  validated end-to-end. These are loudly flagged everywhere (logs, KML top description,
  zone balloons) and **must not be used to plan a hunt.**

### To get real data in

Drop these files at the paths `config.yaml` expects (see comments there), no code
changes needed:

| Layer | Path | How to get it |
|---|---|---|
| Unit boundary | `data/raw/unit_boundary.geojson` | Export from OnX Hunt, or the DWR Hunt Planner (`dwrapps.utah.gov/huntboundary`) |
| Foot trailheads | `data/raw/access_foot.geojson` | Export points from OnX/GOHUNT |
| ATV-legal access | `data/raw/access_atv.geojson` | Export from OnX/GOHUNT, or UGRC roads + USFS MVUM |
| DWR crucial range | `data/raw/dwr_crucial_range.geojson` | DWR Open Data Hub, or fill in `dwr_habitat.crucial_range_service_url` in config to auto-fetch |
| DWR migration corridor | `data/raw/dwr_migration_corridor.geojson` | Same hub |
| Ownership/SITLA | `data/raw/ownership_parcels.geojson` | UGRC land ownership layer, or fill in `ownership.ownership_service_url` |
| Land cover | `data/processed/landcover_evt.tif` | LANDFIRE EVT download for the AOI, clipped/reprojected to match `dem.tif` |

Then delete the corresponding cached rasters under `data/processed/` (or pass
`--force`) and rerun Phase 1/2.

## Project layout

```
config.yaml                  run parameters + habitat weights (tune here)
src/wasatch_scout/
  aoi.py                     unit boundary + buffer resolution
  acquire/dem.py             USGS 3DEP DEM (live in this sandbox)
  acquire/landcover.py       LANDFIRE EVT (network-blocked here, real API)
  acquire/dwr.py             DWR habitat + ownership via ArcGIS REST (network-blocked here)
  acquire/access_seeds.py    OnX/GOHUNT trailhead + ATV route ingestion
  terrain.py                 slope, aspect, hillshade, landform classification
  effort.py                  Tobler anisotropic cost-distance (skimage.graph.MCP_Flexible)
  habitat.py                 weighted habitat score, degrades gracefully if inputs missing
  ownership.py                private/CWMU exclusion, SITLA flag, UNKNOWN-not-public default
  priority.py                zone clustering, ranking, nearest-access lookup
  kml_export.py              4-folder KML + honest-limitations text
  preview.py                 hillshade + effort/habitat preview PNGs
scripts/phase{1,2,3}_*.py    CLI entry points per the brief's phased workflow
data/{raw,interim,processed} gitignored; raw = downloads/manual exports, processed = cached derived rasters
output/                      preview PNGs, phase reports, final KML
```

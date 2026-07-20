"""Assemble wasatch_scout_overlay.kml: priority zones, effort bands, habitat
heat, and access seeds, per the brief's 4-folder structure. Kept visually
light (downsampled + simplified vector bands, thin outlines, translucent
fills) so it layers cleanly over OnX imagery without lag.
"""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import simplekml
from rasterio.features import shapes as rio_shapes
from scipy.ndimage import zoom, uniform_filter
from shapely.geometry import shape as shapely_shape, MultiPolygon
from shapely.ops import unary_union

log = logging.getLogger(__name__)

HONEST_LIMITATIONS = """Wasatch Scouting-Score Overlay -- read before using

WHAT THIS IS: a terrain-aware e-scouting aid layered on top of OnX/GOHUNT. It ranks \
public/legal-access terrain by (a) how hard it is for the average hunter on foot to \
reach, and (b) how good the terrain looks for October elk/deer transition habitat \
(aspect, cover edges, benches/saddles, DWR crucial range/migration overlap).

WHAT THIS IS NOT:
- No layer here shows where animals actually are. It shows where good October habitat \
is hard for the crowd to reach. Ground-truth with glassing and boots.
- DWR crucial range / migration corridor polygons are coarse, seasonal-average \
products, not October-specific or animal-location data.
- The effort surface assumes legal foot/ATV travel only and does not know about \
seasonal closures. Verify every route against OnX and the current MVUM before relying \
on it.
- Weather, rut timing, and hunting pressure move animals week to week. This is a \
starting point for e-scouting, not a plan.

DATA & METHOD NOTES (see this run's status report for specifics that applied to your run):
- DEM: USGS 3DEP ~10m (1/3 arc-second).
- Effort surface: anisotropic hiking-time cost (Tobler's hiking function) \
from motorized-legal access points/trailheads, computed directly from the DEM so uphill \
and downhill travel are costed differently.
- Habitat score: weighted blend of aspect, forest cover/edge proximity, bench/saddle \
landform, and DWR crucial-range/migration overlap -- any input layer that could not be \
acquired is dropped from the blend (weights renormalized) rather than faked; check the \
status report for which terms were actually used.
- Ownership: private and CWMU land is excluded from priority zones when ownership data \
is available; SITLA parcels are flagged, not excluded (legal but rules apply). Zones \
built without verified ownership data are flagged "OWNERSHIP UNVERIFIED" -- do not \
treat those as confirmed public land."""


def _reproject_arr_bounds(path: Path):
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        nodata = ds.nodata
        transform = ds.transform
        crs = ds.crs
    return arr, nodata, transform, crs


def _downsampled_bands(path: Path, bin_edges: list[float], nodata_override=None, downsample_factor: int = 10,
                        smooth_window: int = 5, min_part_acres: float = 2.0, max_parts_per_band: int = 300):
    """Reclassify a raster into bands, downsample + smooth + drop tiny fragments
    for a lighter KML. Raw per-pixel banding of a noisy/high-frequency raster
    (aspect-driven habitat score especially) fragments into tens of thousands of
    speckle polygons if vectorized directly -- smoothing before digitizing and
    dropping sub-threshold fragments keeps this to a small, GE-friendly polygon
    count while still reading as the same overall pattern.

    Returns list of (band_label, shapely_polygon_in_working_crs) merged per band.
    """
    arr, nodata, transform, crs = _reproject_arr_bounds(path)
    if nodata_override is not None:
        nodata = nodata_override
    valid = arr != nodata
    arr_small = zoom(np.where(valid, arr.astype("float64"), np.nan), 1 / downsample_factor, order=0)
    new_transform = transform * transform.scale(downsample_factor, downsample_factor)

    valid_small = ~np.isnan(arr_small)
    filled = np.where(valid_small, arr_small, 0.0)
    weight = uniform_filter(valid_small.astype("float64"), size=smooth_window)
    smoothed_sum = uniform_filter(filled, size=smooth_window)
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = np.where(weight > 0, smoothed_sum / np.where(weight > 0, weight, 1), np.nan)
    smoothed = np.where(valid_small, smoothed, np.nan)

    labels = np.digitize(smoothed, bin_edges, right=False)
    labels = np.where(np.isnan(smoothed), 255, labels).astype("uint8")

    pixel_area_m2 = abs(new_transform.a * new_transform.e)
    min_area_m2 = min_part_acres / 0.000247105

    results = []
    for band_idx in range(1, len(bin_edges)):
        band_mask = labels == band_idx
        if not band_mask.any():
            continue
        polys = [
            shapely_shape(geom)
            for geom, val in rio_shapes(band_mask.astype("uint8"), mask=band_mask, transform=new_transform)
            if val == 1
        ]
        if not polys:
            continue
        merged = unary_union(polys).simplify(30, preserve_topology=True)
        parts = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
        kept = [p for p in parts if not p.is_empty and p.area >= min_area_m2]
        dropped = len(parts) - len(kept)
        if dropped:
            log.info("Band %d: dropped %d fragment(s) under %.1f acres (kept %d)",
                      band_idx, dropped, min_part_acres, len(kept))
        if len(kept) > max_parts_per_band:
            kept.sort(key=lambda p: p.area, reverse=True)
            log.info("Band %d: capping %d fragments to largest %d", band_idx, len(kept), max_parts_per_band)
            kept = kept[:max_parts_per_band]
        if not kept:
            continue
        results.append((band_idx, MultiPolygon(kept) if len(kept) > 1 else kept[0]))
    return results, crs


def _add_multi(folder, geom, name, color_kml, fill_alpha_int):
    """One plain Polygon Placemark per fragment -- no MultiGeometry, no nested
    Folder. onX's KML importer is built around simple Waypoints/Routes/Lines/
    Shapes/Tracks and (per their own support docs) can reject files containing
    "extra data" that Google Earth tolerates; MultiGeometry and folder-within-
    folder are the two most likely offenders, so both are avoided here even
    though they're valid KML 2.2."""
    geoms = [g for g in (geom.geoms if hasattr(geom, "geoms") else [geom]) if not g.is_empty]
    for i, g in enumerate(geoms):
        label = name if len(geoms) == 1 else f"{name} ({i + 1}/{len(geoms)})"
        pol = folder.newpolygon(name=label, outerboundaryis=list(g.exterior.coords))
        pol.style.polystyle.color = simplekml.Color.changealphaint(fill_alpha_int, color_kml)
        pol.style.polystyle.fill = 1
        pol.style.polystyle.outline = 1
        pol.style.linestyle.color = simplekml.Color.changealphaint(180, color_kml)
        pol.style.linestyle.width = 1


def build_kml(cfg: dict, zones: list, foot_minutes_path: Path, habitat_path: Path, seeds: dict,
              out_path: Path, aoi_authoritative: bool, seeds_authoritative: bool):
    kml = simplekml.Kml()
    top_note = (
        ("\n\n*** AOI WARNING: unit boundary was NOT supplied -- an unofficial fallback bbox was used. "
         "Do not trust zone placement or ownership exclusion until you rerun with the real boundary. ***"
         if not aoi_authoritative else "")
        + ("\n\n*** ACCESS SEED WARNING: trailhead/route seeds are DEMO placeholders, not your real OnX/GOHUNT "
           "export. Effort bands and nearest-access times are for pipeline validation only. ***"
           if not seeds_authoritative else "")
    )
    kml.document.name = f"Wasatch Scouting-Score Overlay -- {cfg['run']['unit_name']}"
    kml.document.description = HONEST_LIMITATIONS + top_note

    # 1. Priority Scout Zones
    zone_folder = kml.newfolder(name="1. Priority Scout Zones")
    for z in zones:
        parts = [p for p in (z.geometry.geoms if hasattr(z.geometry, "geoms") else [z.geometry]) if not p.is_empty]
        base_name = f"#{z.rank} - {z.dominant_landform.title()} ({z.acres:.0f} ac)"

        why = f"Aspect/landform/cover blend score {z.mean_habitat:.2f}; dominant terrain: {z.dominant_landform}"
        if z.dwr_overlap_pct > 0:
            why += f"; {z.dwr_overlap_pct:.0f}% overlaps DWR crucial range/migration corridor"
        access_lines = []
        if z.nearest_foot:
            access_lines.append(
                f"Foot: {z.nearest_foot['name']} -- {z.nearest_foot['one_way_minutes']} min one-way, "
                f"{z.nearest_foot['vert_gain_m']:+.0f} m net vert (straight-line elev change, not total climb)"
            )
        if z.nearest_atv:
            access_lines.append(
                f"ATV-alt: {z.nearest_atv['name']} -- {z.nearest_atv['one_way_minutes']} min one-way, "
                f"{z.nearest_atv['vert_gain_m']:+.0f} m net vert (straight-line elev change, not total climb)"
            )
        flags = f"\nFlags: {', '.join(z.ownership_flags)}" if z.ownership_flags else ""
        description = (
            f"Rank {z.rank} of {len(zones)} | {z.acres:.0f} acres | priority score {z.priority_score:.2f}\n"
            f"Why it scored: {why}\n"
            f"Mean one-way hiking time from foot access: {z.mean_foot_minutes} min "
            f"(closest point {z.min_foot_minutes} min)\n"
            + "\n".join(access_lines) + flags
        )

        for i, part in enumerate(parts):
            name = base_name if len(parts) == 1 else f"{base_name} ({i + 1}/{len(parts)})"
            pol = zone_folder.newpolygon(name=name, outerboundaryis=list(part.exterior.coords), description=description)
            pol.style.polystyle.color = simplekml.Color.changealphaint(90, simplekml.Color.gold)
            pol.style.linestyle.color = simplekml.Color.changealphaint(220, simplekml.Color.orange)
            pol.style.linestyle.width = 2

    # 2. Effort Bands
    effort_folder = kml.newfolder(name="2. Effort Bands (one-way hiking minutes from foot access)")
    max_effort = cfg["run"]["max_effort_minutes"]
    edges = [0, 30, 60, 90, max(90, max_effort) + 1e6]
    band_names = {1: "< 30 min", 2: "30-60 min", 3: "60-90 min", 4: f"> 90 min"}
    band_colors = {1: simplekml.Color.green, 2: simplekml.Color.yellow, 3: simplekml.Color.orange, 4: simplekml.Color.red}
    bands, band_crs = _downsampled_bands(foot_minutes_path, edges, nodata_override=-1)
    band_gdf_crs = band_crs
    for band_idx, geom in bands:
        geom_wgs = gpd.GeoSeries([geom], crs=band_gdf_crs).to_crs("EPSG:4326").iloc[0]
        _add_multi(effort_folder, geom_wgs, band_names.get(band_idx, str(band_idx)),
                   band_colors.get(band_idx, simplekml.Color.white), 80)

    # 3. Habitat Score Heat
    heat_folder = kml.newfolder(name="3. Habitat Score Heat")
    hab_edges = [0, 0.4, 0.6, 0.75, 0.9, 1.01]
    hab_names = {1: "0.0-0.4 low", 2: "0.4-0.6 fair", 3: "0.6-0.75 good", 4: "0.75-0.9 very good", 5: "0.9-1.0 excellent"}
    hab_colors = {1: simplekml.Color.white, 2: simplekml.Color.lightblue, 3: simplekml.Color.blue,
                  4: simplekml.Color.purple, 5: simplekml.Color.magenta}
    hab_bands, hab_crs = _downsampled_bands(habitat_path, hab_edges, nodata_override=-1,
                                             downsample_factor=15, smooth_window=7, min_part_acres=3.0)
    for band_idx, geom in hab_bands:
        geom_wgs = gpd.GeoSeries([geom], crs=hab_crs).to_crs("EPSG:4326").iloc[0]
        _add_multi(heat_folder, geom_wgs, hab_names.get(band_idx, str(band_idx)),
                   hab_colors.get(band_idx, simplekml.Color.white), 65)

    # 4. Access Seeds Used
    seeds_folder = kml.newfolder(name="4. Access Seeds Used")
    foot_tag = "" if seeds_authoritative else " (DEMO placeholder)"
    for _, row in seeds["foot"].iterrows():
        pt = seeds_folder.newpoint(name=f"Foot: {row.get('name', 'trailhead')}{foot_tag}",
                                    coords=[(row.geometry.x, row.geometry.y)])
        pt.style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/shapes/hiker.png"
    if len(seeds.get("atv", [])):
        atv_tag = "" if seeds.get("atv_authoritative") else " (DEMO placeholder)"
        for _, row in seeds["atv"].iterrows():
            pt = seeds_folder.newpoint(name=f"ATV: {row.get('name', 'atv access')}{atv_tag}",
                                        coords=[(row.geometry.x, row.geometry.y)])
            pt.style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/shapes/cabin_baseball.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    kml.save(str(out_path))
    log.info("KML written: %s", out_path)
    return out_path

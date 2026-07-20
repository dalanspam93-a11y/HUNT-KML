"""Land cover acquisition: LANDFIRE Existing Vegetation Type (EVT) via the
LANDFIRE Product Service (LFPS), a REST geoprocessing API that clips rasters
to an AOI server-side so we don't have to pull the full CONUS mosaic.

NOT reachable from this sandbox (lfps.usgs.gov is blocked by the outbound
network policy here -- see README). The request/poll/download flow below
follows LANDFIRE's documented API and should work unmodified in an
environment with normal internet access; run scripts/phase1_acquire.py there,
or download the EVT GeoTIFF for the AOI manually from landfire.gov and place
it at the path this function returns.

Falls back cleanly: returns None (not an exception) if unreachable, so
habitat scoring can proceed without the cover/edge term and say so.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

from ..aoi import AOI
from ..config import resolve

log = logging.getLogger(__name__)

LFPS_BASE = "https://lfps.usgs.gov/arcgis/rest/services/LandfireProductService/GPServer/LandfireProductService"
EVT_LAYER = "200EVT"  # LANDFIRE product code for Existing Vegetation Type, most recent version


def acquire_landcover(cfg: dict, aoi: AOI, force: bool = False, timeout_s: int = 600) -> Path | None:
    out_path = resolve(cfg, "paths.processed_dir") / "landcover_evt.tif"
    if out_path.exists() and not force:
        log.info("Land cover cache hit: %s", out_path)
        return out_path

    min_lon, min_lat, max_lon, max_lat = aoi.bbox_wgs84
    aoi_wkt = f"POLYGON(({min_lon} {min_lat},{max_lon} {min_lat},{max_lon} {max_lat},{min_lon} {max_lat},{min_lon} {min_lat}))"

    submit_params = {
        "f": "json",
        "Layer_List": EVT_LAYER,
        "Area_of_Interest": aoi_wkt,
        "Output_Projection": "4326",
    }
    try:
        r = requests.post(f"{LFPS_BASE}/submitJob", data=submit_params, timeout=30)
        r.raise_for_status()
        job_id = r.json()["jobId"]
        log.info("LFPS job submitted: %s", job_id)

        status_url = f"{LFPS_BASE}/jobs/{job_id}"
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            s = requests.get(status_url, params={"f": "json"}, timeout=30).json()
            if s.get("jobStatus") == "esriJobSucceeded":
                break
            if s.get("jobStatus") in ("esriJobFailed", "esriJobCancelled"):
                raise RuntimeError(f"LFPS job {job_id} failed: {s}")
            time.sleep(10)
        else:
            raise TimeoutError(f"LFPS job {job_id} did not finish within {timeout_s}s")

        result_url = f"{status_url}/results/Output_File"
        result_meta = requests.get(result_url, params={"f": "json"}, timeout=30).json()
        download_url = result_meta["value"]["url"]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(download_url, stream=True, timeout=120) as dl:
            dl.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in dl.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        log.info("Land cover (LANDFIRE EVT) saved: %s", out_path)
        return out_path

    except Exception as e:
        log.warning(
            "Land cover acquisition failed (%s). This is expected if lfps.usgs.gov is "
            "unreachable from this environment. Habitat scoring will skip the cover/edge "
            "term until data/processed/landcover_evt.tif is provided manually.", e,
        )
        return None

"""Live PurpleAir sensor fetcher + same-day neighbor-feature computation.

The v3b model's single most important feature (nbr_pm25_50km, ~37% of total
importance) is the mean PM2.5 of OTHER sensors within 50km ON THE SAME DAY.
At training (pipeline/03_train_enhanced.py) this is computed per (sensor, date)
with a BallTree over that day's readings. To reproduce it at inference WITHOUT
train/serve skew, the backend must use LIVE same-day PurpleAir readings — not a
static historical mean.

This module:
  1. fetch_live_sensors(): one bounding-box call to PurpleAir's /v1/sensors
     live endpoint for all outdoor Texas sensors, QC'd to fresh + trustworthy.
  2. compute_neighbor_features(): replicates the training BallTree computation
     with census-tract centroids as query points and live sensors as the
     reference set, returning (nbr_pm25_50km, nbr_count_50km, nbr_std_50km).

CRITICAL correctness notes:
  - Use pm2.5_atm, the raw ATM-channel mass conc. Training used pm2.5_atm
    (pipeline/06 PA_HISTORY_FIELDS) renamed straight to the model target.
    Do NOT apply the EPA Barkjohn correction here — training data was raw.
  - 50km radius == 50.0/6371.0 radians on a haversine BallTree, identical to
    training (pipeline/03 line ~314).
  - Fallback for tracts with zero live neighbors == live statewide same-day
    mean, matching training's groupby(date) statewide-mean fallback. NO 100km
    widening tier (training never widened — adding one is train/serve skew).
"""
import os
import time
from datetime import datetime, timezone

import httpx
import numpy as np

# Texas + buffer bounding box (matches pipeline/06_pull_purpleair_full.py).
# PurpleAir bbox semantics: nw = NORTHWEST corner (max lat, min lon),
# se = SOUTHEAST corner (min lat, max lon).
TX_BBOX = {"nwlng": -106.65, "nwlat": 36.5, "selng": -93.5, "selat": 25.84}

PA_LIVE_URL = "https://api.purpleair.com/v1/sensors"

# QC thresholds (mirror training + live-only quality gates).
PM25_MIN_VALID = 0.0
PM25_MAX_VALID = 200.0       # same hard cap as the training pull
MAX_AGE_SEC = 3600           # drop sensors that haven't reported in 1 hour
MIN_CONFIDENCE = 50          # PurpleAir A/B channel agreement score
MIN_LIVE_SENSORS = 20        # below this we don't trust the live snapshot

EARTH_R_KM = 6371.0
NBR_RADIUS_KM = 50.0


# Fallback READ key so the live path works out-of-the-box on Render without a
# manual env-var step. This is the same key already committed (and therefore
# already public) in pipeline/06_pull_purpleair_full.py — using it here does not
# change the security posture. SET PURPLEAIR_API_KEY in the Render dashboard to
# override it, and rotate this key when convenient (see deploy notes).
_FALLBACK_READ_KEY = "8E76496A-3C3D-11F1-B596-4201AC1DC123"


def _get_api_key() -> str:
    """Read the PurpleAir READ key: env var first, committed fallback otherwise."""
    return os.environ.get("PURPLEAIR_API_KEY", "").strip() or _FALLBACK_READ_KEY


async def fetch_live_sensors() -> list[dict]:
    """Fetch current outdoor PurpleAir readings across Texas, QC'd.

    Returns a list of {lat, lon, pm25} dicts (pm25 = raw pm2.5_atm). Returns
    an empty list on any failure (no key, HTTP error, too few sensors) so the
    caller can cleanly fall back to the static climatology — never worse than
    the current behavior.
    """
    key = _get_api_key()
    if not key:
        return []

    params = {
        "fields": "latitude,longitude,pm2.5_atm,last_seen,confidence,location_type",
        "location_type": 0,        # outdoor only
        "max_age": MAX_AGE_SEC,    # server-side freshness filter
        **TX_BBOX,
    }
    try:
        async with httpx.AsyncClient(timeout=40.0) as client:
            resp = await client.get(PA_LIVE_URL, headers={"X-API-Key": key}, params=params)
            if resp.status_code != 200:
                print(f"[purpleair] HTTP {resp.status_code}: {resp.text[:200]}")
                return []
            payload = resp.json()
    except Exception as e:
        print(f"[purpleair] fetch error: {e}")
        return []

    cols = payload.get("fields")
    rows = payload.get("data")
    if not cols or not rows:
        return []

    idx = {name: i for i, name in enumerate(cols)}
    needed = ("latitude", "longitude", "pm2.5_atm", "last_seen", "confidence", "location_type")
    if not all(n in idx for n in needed):
        print(f"[purpleair] unexpected fields: {cols}")
        return []

    now = int(time.time())
    sensors = []
    for row in rows:
        try:
            lat = row[idx["latitude"]]
            lon = row[idx["longitude"]]
            pm = row[idx["pm2.5_atm"]]
            last_seen = row[idx["last_seen"]]
            conf = row[idx["confidence"]]
            loc = row[idx["location_type"]]
        except (IndexError, KeyError):
            continue
        # QC: outdoor, fresh, trustworthy, valid PM.
        if lat is None or lon is None or pm is None or last_seen is None:
            continue
        if loc != 0:
            continue
        if conf is None or conf < MIN_CONFIDENCE:
            continue
        if (now - int(last_seen)) > MAX_AGE_SEC:
            continue
        if not (PM25_MIN_VALID <= pm <= PM25_MAX_VALID):
            continue
        sensors.append({"lat": float(lat), "lon": float(lon), "pm25": float(pm)})

    return sensors


def compute_neighbor_features(tract_lats, tract_lons, sensors):
    """Replicate the training same-day neighbor computation for tract centroids.

    For each tract: mean / count / population-std of PM2.5 from live sensors
    within 50km. Tracts with zero neighbors get the live statewide same-day
    mean (matching the training fallback) with count=0, std=0.

    Returns (nbr_mean, nbr_count, nbr_std) numpy arrays aligned to the inputs,
    or (None, None, None) if there aren't enough live sensors to trust.
    """
    from sklearn.neighbors import BallTree

    if not sensors or len(sensors) < MIN_LIVE_SENSORS:
        return None, None, None

    s_lat = np.array([s["lat"] for s in sensors], dtype=np.float64)
    s_lon = np.array([s["lon"] for s in sensors], dtype=np.float64)
    s_pm = np.array([s["pm25"] for s in sensors], dtype=np.float64)

    s_coords = np.radians(np.column_stack([s_lat, s_lon]))
    t_coords = np.radians(np.column_stack([
        np.asarray(tract_lats, dtype=np.float64),
        np.asarray(tract_lons, dtype=np.float64),
    ]))
    tree = BallTree(s_coords, metric="haversine")
    radius = NBR_RADIUS_KM / EARTH_R_KM
    neighbors = tree.query_radius(t_coords, r=radius)

    n = len(t_coords)
    nbr_mean = np.full(n, np.nan)
    nbr_count = np.zeros(n, dtype=np.int32)
    nbr_std = np.zeros(n, dtype=np.float64)
    for i, nb in enumerate(neighbors):
        if len(nb) > 0:
            vals = s_pm[nb]
            nbr_mean[i] = float(vals.mean())
            nbr_count[i] = int(len(nb))
            nbr_std[i] = float(vals.std()) if len(nb) > 1 else 0.0

    # Fallback: tracts with no neighbors get the live statewide same-day mean.
    statewide_mean = float(s_pm.mean())
    nbr_mean = np.where(nbr_count > 0, nbr_mean, statewide_mean)

    return nbr_mean, nbr_count, nbr_std


async def fetch_live_snapshot() -> dict:
    """Fetch + package live sensors into a snapshot dict (cache/disk friendly)."""
    sensors = await fetch_live_sensors()
    now = datetime.now(timezone.utc)
    pm_vals = [s["pm25"] for s in sensors]
    return {
        "sensors": sensors,
        "count": len(sensors),
        "statewide_mean": float(np.mean(pm_vals)) if pm_vals else None,
        "fetched_at": now.isoformat(),
        "usable": len(sensors) >= MIN_LIVE_SENSORS,
    }

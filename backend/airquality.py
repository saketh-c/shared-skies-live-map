"""Live CAMS air-quality + met PBL-proxy fetcher for inference.

Provides the same 6 features the v4 model trains on:
  - aod, cams_pm25, dust   (Open-Meteo Air-Quality forecast, daily mean of today)
  - shortwave, et0, cloud_cover  (Open-Meteo forecast, today's daily values)

CONSISTENCY: training used the DAILY MEAN of hourly archive AOD/pm2_5/dust, so
inference averages today's hourly forecast to a daily mean (NOT a single current
hour) — otherwise the diurnal cycle would create train/serve skew. Met features
are daily aggregates in both phases.

To stay within rate limits and memory, tracts are binned to a coarse ~0.5deg
grid (~50-100 occupied cells over Texas); we make 2 batched multi-location calls
(air-quality + met) covering all cells, then map each tract to its cell value.
"""
import os
from datetime import datetime, timezone

import httpx
import numpy as np

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
MET_URL = "https://api.open-meteo.com/v1/forecast"
GRID_DEG = 0.5  # cell size for binning tracts


def _cell_key(lat: float, lon: float) -> tuple:
    return (round(lat / GRID_DEG) * GRID_DEG, round(lon / GRID_DEG) * GRID_DEG)


def build_cells(lats, lons):
    """Map tracts to coarse grid cells. Returns (cell_list, tract_cell_index)
    where cell_list is unique (lat, lon) cell centers and tract_cell_index[i] is
    the index into cell_list for tract i."""
    keys = [_cell_key(float(la), float(lo)) for la, lo in zip(lats, lons)]
    uniq = {}
    order = []
    idx = np.empty(len(keys), dtype=np.int32)
    for i, k in enumerate(keys):
        if k not in uniq:
            uniq[k] = len(order)
            order.append(k)
        idx[i] = uniq[k]
    return order, idx


def _daily_mean_today(times, values):
    """Mean of today's hourly values (skip NaN). times are ISO hour strings."""
    if not times or not values:
        return None
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    vals = [v for t, v in zip(times, values) if v is not None and str(t).startswith(today)]
    if not vals:
        vals = [v for v in values if v is not None]  # fallback: whole window
    return float(np.mean(vals)) if vals else None


async def fetch_cell_features(cell_centers) -> dict:
    """Fetch air-quality + met for each grid cell. Returns
    {cell_index: {aod, cams_pm25, dust, shortwave, et0, cloud_cover}} (None where
    unavailable). Batched multi-location calls to stay within rate limits."""
    if not cell_centers:
        return {}
    lats = ",".join(f"{c[0]:.4f}" for c in cell_centers)
    lons = ",".join(f"{c[1]:.4f}" for c in cell_centers)
    out = {i: {} for i in range(len(cell_centers))}

    # 1. Air-quality forecast (hourly today -> daily mean).
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.get(AQ_URL, params={
                "latitude": lats, "longitude": lons,
                "hourly": "aerosol_optical_depth,pm2_5,dust",
                "forecast_days": 1, "timezone": "UTC",
            })
            if r.status_code == 200:
                payload = r.json()
                if isinstance(payload, dict):
                    payload = [payload]
                for i, loc in enumerate(payload):
                    h = loc.get("hourly", {})
                    t = h.get("time", [])
                    out[i]["aod"] = _daily_mean_today(t, h.get("aerosol_optical_depth", []))
                    out[i]["cams_pm25"] = _daily_mean_today(t, h.get("pm2_5", []))
                    out[i]["dust"] = _daily_mean_today(t, h.get("dust", []))
    except Exception as e:
        print(f"[airquality] AQ fetch error: {e}")

    # 2. Met forecast (today's daily aggregates).
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.get(MET_URL, params={
                "latitude": lats, "longitude": lons,
                "daily": "shortwave_radiation_sum,et0_fao_evapotranspiration,cloud_cover_mean",
                "forecast_days": 1, "timezone": "UTC",
            })
            if r.status_code == 200:
                payload = r.json()
                if isinstance(payload, dict):
                    payload = [payload]
                for i, loc in enumerate(payload):
                    d = loc.get("daily", {})
                    sw = d.get("shortwave_radiation_sum", [None])
                    et = d.get("et0_fao_evapotranspiration", [None])
                    cc = d.get("cloud_cover_mean", [None])
                    out[i]["shortwave"] = sw[0] if sw else None
                    out[i]["et0"] = et[0] if et else None
                    out[i]["cloud_cover"] = cc[0] if cc else None
    except Exception as e:
        print(f"[airquality] met fetch error: {e}")

    return out


async def fetch_airquality_snapshot(lats, lons) -> dict:
    """Build a per-tract feature snapshot for all tracts. Returns
    {'features': {feat: [per-tract values]}, 'fetched_at', 'n_cells', 'usable'}.
    Values are None where the API didn't return them (backend fills with the
    training median)."""
    cells, idx = build_cells(lats, lons)
    cell_feats = await fetch_cell_features(cells)
    feat_names = ["aod", "cams_pm25", "dust", "shortwave", "et0", "cloud_cover"]
    per_tract = {f: [None] * len(lats) for f in feat_names}
    usable = False
    for i in range(len(lats)):
        cf = cell_feats.get(int(idx[i]), {})
        for f in feat_names:
            v = cf.get(f)
            per_tract[f][i] = v
            if v is not None:
                usable = True
    return {
        "features": per_tract,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "n_cells": len(cells),
        "usable": usable,
    }

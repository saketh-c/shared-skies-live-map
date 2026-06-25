"""
Shared Skies — FastAPI Backend
Serves PM2.5 predictions for multiple Texas counties (Dallas, Austin, Houston, San Antonio).

Start with:
    uvicorn backend.main:app --reload --port 8000
(run from the project root)
"""

import asyncio
import json
import math
import os
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import httpx
import joblib
import numpy as np
import pandas as pd
import sqlite3
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(ROOT, "backend", "static")
MODEL_PATH = os.path.join(ROOT, "models", "ensemble.joblib")
LOOKUP_PATH = os.path.join(STATIC_DIR, "tract_lookup.parquet")
TEXAS_GEOJSON_PATH = os.path.join(STATIC_DIR, "texas_all_tracts.geojson")
VISIT_COUNT_PATH = os.path.join(ROOT, "backend", "visit_count.json")
VISITS_DB = os.path.join(ROOT, "backend", "visits.sqlite")
SNAPSHOTS_DIR = os.path.join(STATIC_DIR, "snapshots")
TEXAS_SNAPSHOT_PATH = os.path.join(SNAPSHOTS_DIR, "texas_predictions_latest.json")
QUANTUM_SNAPSHOT_PATH = os.path.join(SNAPSHOTS_DIR, "quantum_latest.json")
HMS_SNAPSHOT_PATH = os.path.join(SNAPSHOTS_DIR, "hms_smoke_latest.json")
PURPLEAIR_SNAPSHOT_PATH = os.path.join(SNAPSHOTS_DIR, "purpleair_sensors_latest.json")
WEATHER_GRID_SNAPSHOT_PATH = os.path.join(SNAPSHOTS_DIR, "weather_grid_latest.json")
AQ_SNAPSHOT_PATH = os.path.join(SNAPSHOTS_DIR, "airquality_latest.json")
CONDITIONS_SNAPSHOT_PATH = os.path.join(SNAPSHOTS_DIR, "conditions_latest.json")
MODELS_DIR = os.path.join(ROOT, "models")

# Live PurpleAir: the model's dominant feature (nbr_pm25_*) must be computed
# from same-day live readings to match training. TTL 60 min: with the 24h-mean
# field a 1-hour cadence tracks events closely while staying cheap (24 small
# bbox pulls/day). Set PURPLEAIR_API_KEY in Render env to override the key.
PURPLEAIR_CACHE_TTL_MIN = int(os.environ.get("PURPLEAIR_CACHE_TTL_MIN", "15"))
# Clip dist_to_nearest_sensor to the training-network max so rural tracts (which
# can be 195km from any sensor vs the 164km training max) don't push the tree
# models out of distribution.
DIST_TO_SENSOR_MAX_KM = 164.4

# Upstash Redis REST — set these env vars in Render for persistent visit counts.
# If not set, falls back to SQLite (resets on every server restart).
UPSTASH_URL   = os.environ.get("UPSTASH_REDIS_REST_URL", "").rstrip("/")
UPSTASH_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN", "")

# How long the *publicly-shown* visit count is cached before refreshing.
# Real count still increments on every /api/visit — this only governs what
# users see, so the number doesn't obviously tick up when someone refreshes.
PUBLIC_VISITS_TTL_MIN = 10

# Module-level cache for the displayed visit count
_public_visits_cache: dict = {"count": None, "expires": datetime.min.replace(tzinfo=timezone.utc)}

# ── City Configuration (Generic, extensible design) ─────────────────────────
CITIES = {
    "dallas": {
        "fips": "48113",
        "center": (32.7767, -96.7970),
        "tz": "America/Chicago",
        "geojson": os.path.join(STATIC_DIR, "dallas_tracts.geojson"),
        "display_name": "Dallas County"
    },
    "austin": {
        "fips": "48453",
        "center": (30.2672, -97.7431),
        "tz": "America/Chicago",
        "geojson": os.path.join(STATIC_DIR, "austin_tracts.geojson"),
        "display_name": "Travis County"
    },
    "houston": {
        "fips": "48201",
        "center": (29.7604, -95.3698),
        "tz": "America/Chicago",
        "geojson": os.path.join(STATIC_DIR, "houston_tracts.geojson"),
        "display_name": "Harris County"
    },
    "san_antonio": {
        "fips": "48029",
        "center": (29.4241, -98.4936),
        "tz": "America/Chicago",
        "geojson": os.path.join(STATIC_DIR, "san_antonio_tracts.geojson"),
        "display_name": "Bexar County"
    }
}
DEFAULT_CITY = "dallas"

# ── Shared state ──────────────────────────────────────────────────────────────
state: dict = {}

# Flags so we never run more than one background recompute of the same thing
_revalidating: dict = {"texas": False, "quantum": False, "hms": False, "purpleair": False}

# NOAA HMS smoke polygon live cache (refreshed by background task every HMS_CACHE_TTL_MIN).
_hms_cache: dict = {"data": None, "expires": datetime.min.replace(tzinfo=timezone.utc)}
HMS_CACHE_TTL_MIN = 60  # HMS analysts publish daily and re-draw a few times/day

# Live PurpleAir sensor cache (list of {lat, lon, pm25} for current TX air).
_purpleair_cache: dict = {"data": None, "expires": datetime.min.replace(tzinfo=timezone.utc)}

# Open-Meteo grid caches (daily means — refreshing a few times a day is plenty).
# QUOTA: Open-Meteo's free tier is ~10,000 call-weights/day per API, and a
# multi-location call weighs ~1 PER LOCATION. Refetching a 254-cell grid every
# 30-min prediction cycle (48×/day ≈ 12k weights) exhausted the quota and 429'd
# (observed on Render). With a 6h TTL + disk snapshot + stale-on-failure, the
# steady-state cost is a few hundred weights/day per API.
OPENMETEO_GRID_TTL_MIN = int(os.environ.get("OPENMETEO_GRID_TTL_MIN", "360"))
_wgrid_cache: dict = {"data": None, "expires": datetime.min.replace(tzinfo=timezone.utc)}
_aq_cache: dict = {"data": None, "expires": datetime.min.replace(tzinfo=timezone.utc)}

# CURRENT-CONDITIONS grid (display-only): one batched 81-cell call per hour is
# the always-available baseline for every tract/city/overview panel.
CONDITIONS_TTL_MIN = int(os.environ.get("CONDITIONS_TTL_MIN", "60"))
_conditions_cache: dict = {"data": None, "expires": datetime.min.replace(tzinfo=timezone.utc)}

# EXACT-location current weather for a CLICKED tract (display only). The grid
# above is coarse (1.0° ≈ a cell center up to ~70km away → a few °F off vs a
# user's local weather); a clicked tract gets an exact single-location fetch
# instead, cached per ~0.1° cell (~11km) and SHARED across users so the unique-
# cell count is bounded by geography, not traffic. Hard guarantees against the
# past failures: (1) any fetch error/429 trips a cooldown and falls back to the
# conditions GRID (location-varying, never a frozen statewide constant);
# (2) the cache is size-capped.
CLICK_WEATHER_TTL_MIN = int(os.environ.get("CLICK_WEATHER_TTL_MIN", "60"))
_click_weather_cache = OrderedDict()       # "lat,lon" 0.1° key -> {weather, expires}
_click_weather_cooldown = {"until": datetime.min.replace(tzinfo=timezone.utc)}


# ── Spatial-context reference points (kept in sync with pipeline/03_train_enhanced.py) ──
TX_COAST_POINTS = [
    (25.97, -97.50), (27.80, -97.40), (28.93, -95.97),
    (29.30, -94.79), (29.70, -93.90),
]
TX_URBAN_POINTS = [
    (32.78, -96.80), (29.76, -95.37), (30.27, -97.74),
    (29.42, -98.49), (32.75, -97.33), (31.76, -106.49),
]


def _haversine_km_np(lat1, lon1, lat2, lon2):
    """Vectorized great-circle distance in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _min_dist_to_points(lats, lons, points):
    out = np.full(len(lats), np.inf)
    for (plat, plon) in points:
        out = np.minimum(out, _haversine_km_np(lats, lons, plat, plon))
    return out


def _enrich_tract_lookup_with_distances(df: pd.DataFrame) -> None:
    """Add dist_to_nearest_sensor, dist_to_coast, dist_to_urban to tract_lookup in place.
    The v3 model uses these features at inference. Reads sensor locations from
    p2_processed_v2.xls (falls back to p2_processed.xls). Idempotent — no-op if
    columns already exist."""
    if {"dist_to_nearest_sensor", "dist_to_coast", "dist_to_urban"} <= set(df.columns):
        return
    lats = df["lat"].values
    lons = df["lon"].values
    df["dist_to_coast"] = _min_dist_to_points(lats, lons, TX_COAST_POINTS)
    df["dist_to_urban"] = _min_dist_to_points(lats, lons, TX_URBAN_POINTS)

    # dist_to_nearest_sensor needs the training sensor list. Sensor locations
    # come from the same training file the model was trained on.
    sensor_lats, sensor_lons = None, None
    for cand in ("p2_processed_v2.xls", "p2_processed.xls"):
        path = os.path.join(ROOT, cand)
        if not os.path.exists(path):
            continue
        try:
            sdf = pd.read_csv(path, encoding="utf-8-sig", low_memory=False,
                              usecols=lambda c: c in ("sensor_id", "latitude", "longitude"))
            sdf = sdf.drop_duplicates("sensor_id")
            sensor_lats = sdf["latitude"].values
            sensor_lons = sdf["longitude"].values
            print(f"  loaded {len(sdf)} sensor locations from {cand} for dist features")
            break
        except Exception as e:
            print(f"  could not read sensor coords from {cand}: {e}")

    if sensor_lats is None:
        # Fall back to predicting at zero distance — degrades cleanly into a
        # constant feature the model will ignore at split time.
        print("  WARNING: no sensor coords found; dist_to_nearest_sensor = 0")
        df["dist_to_nearest_sensor"] = 0.0
        return

    # For each tract, distance to nearest sensor (vectorized in chunks).
    nearest = np.full(len(df), np.inf)
    chunk = 2048
    for start in range(0, len(sensor_lats), chunk):
        s_lats = sensor_lats[start:start + chunk]
        s_lons = sensor_lons[start:start + chunk]
        for sl, slo in zip(s_lats, s_lons):
            nearest = np.minimum(nearest, _haversine_km_np(lats, lons, sl, slo))
    # Clip to the training-network max. Training computed sensor->nearest-OTHER-
    # sensor (max ~164km); tracts can be ~195km from any sensor. Clipping keeps
    # rural tracts in the tree models' training distribution instead of
    # extrapolating into never-seen feature space.
    nearest = np.clip(nearest, 0.0, DIST_TO_SENSOR_MAX_KM)
    df["dist_to_nearest_sensor"] = nearest
    print(f"  enriched lookup: dist_to_nearest_sensor median={np.median(nearest):.1f} km "
          f"(clipped to {DIST_TO_SENSOR_MAX_KM}), "
          f"dist_to_coast median={np.median(df['dist_to_coast'].values):.1f} km, "
          f"dist_to_urban median={np.median(df['dist_to_urban'].values):.1f} km")

    # ── Neighbor-PM features: compute the SAME multi-radius (25/50/100km) neighbor
    # PM2.5 the model trains on, here from the per-sensor CLIMATOLOGICAL means.
    # This is the inference-time FALLBACK; the live path overrides these columns
    # each cycle with same-day PurpleAir. Both paths call the SHARED
    # compute_neighbor_features so they are byte-identical and match training.
    nbr_path = os.path.join(ROOT, "models", "sensor_recent_pm.json")
    if os.path.exists(nbr_path):
        try:
            from backend.purpleair import compute_neighbor_features
            with open(nbr_path) as f:
                sensors_meta = json.load(f)
            static_sensors = [
                {"lat": float(s["lat"]), "lon": float(s["lon"]),
                 "pm25": float(s["recent_mean_pm25"])}
                for s in sensors_meta
            ]
            nbr_feats = compute_neighbor_features(lats, lons, static_sensors)
            if nbr_feats is None:
                raise ValueError("too few sensors for neighbor features")
            for fname, arr in nbr_feats.items():
                df[fname] = arr
            print(f"  neighbor features (climatological fallback): nbr_pm25_50km "
                  f"median={np.median(nbr_feats['nbr_pm25_50km']):.2f} µg/m³, "
                  f"50km coverage={(nbr_feats['nbr_count_50km']>0).sum()/len(df)*100:.1f}%")
        except Exception as e:
            print(f"  WARNING: failed to build neighbor features from {nbr_path}: {e}; filling 0")
            for r_km in (25, 50, 100):
                df[f"nbr_pm25_{r_km}km"] = 0.0
                df[f"nbr_count_{r_km}km"] = 0
            df["nbr_std_50km"] = 0.0
    else:
        # Pre-v3 deployment — model doesn't use these features. Backend's
        # run_predictions_batch only picks features the bundle requests, so
        # leaving them off costs nothing.
        print(f"  no {os.path.basename(nbr_path)} (pre-v3 model); skipping neighbor features")


def _save_snapshot(path: str, data: dict) -> None:
    """Write a cached API response to disk so cold starts can serve it instantly."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, default=str)
        os.replace(tmp, path)
    except Exception as e:
        print(f"Snapshot save failed ({path}): {e}")


def _load_snapshot(path: str) -> dict | None:
    """Load a previously persisted API response (returns None if missing/corrupt)."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Snapshot load failed ({path}): {e}")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model in a thread so the event loop stays responsive
    if os.path.exists(MODEL_PATH):
        try:
            # Use memory-mapping when possible to speed load and reduce peak memory
            state['bundle'] = await asyncio.to_thread(joblib.load, MODEL_PATH, mmap_mode='r')
        except TypeError:
            # Older joblib versions may not accept mmap_mode; fall back to normal load
            state['bundle'] = await asyncio.to_thread(joblib.load, MODEL_PATH)
        print(f"Model loaded. Features: {state['bundle']['feature_names']}")
    else:
        state["bundle"] = None
        print("WARNING: models/ensemble.joblib not found. Run pipeline/02_train_model.py")

    # Load tract lookup in a thread
    if os.path.exists(LOOKUP_PATH):
        df = await asyncio.to_thread(pd.read_parquet, LOOKUP_PATH)
        df["GEOID"] = df["GEOID"].astype(str).str.zfill(11)
        # Enrich with spatial-context features the v3 model expects. These are
        # static once sensor locations are known, so we compute them once at
        # startup rather than per-prediction. Bound vars used below: TX coast +
        # urban reference points and the sensor location list.
        _enrich_tract_lookup_with_distances(df)
        state["tract_lookup"] = df
        print(f"Tract lookup loaded: {len(df)} total tracts")
        for city, config in CITIES.items():
            count = df["GEOID"].str.startswith(config["fips"]).sum()
            print(f"  {city}: {count} tracts")
    else:
        state["tract_lookup"] = None
        print("WARNING: backend/static/tract_lookup.parquet not found. Run pipeline/01_build_tract_lookup.py")

    # Load visit count from file (persists within a deploy)
    if os.path.exists(VISIT_COUNT_PATH):
        try:
            with open(VISIT_COUNT_PATH) as f:
                state["visits"] = json.load(f).get("visits", 0)
        except Exception:
            state["visits"] = 0
    else:
        state["visits"] = 0

    # Per-city caches
    for city in CITIES:
        state[f"cache_{city}"] = {"data": None, "expires": datetime.min.replace(tzinfo=timezone.utc)}

    # Statewide cache
    state["cache_texas"] = {"data": None, "expires": datetime.min.replace(tzinfo=timezone.utc)}

    # Hydrate caches from disk snapshots so cold-start users get an instant response.
    # We mark them as already-expired so the first request triggers a fresh recompute
    # in the background (stale-while-revalidate) — the user just doesn't have to wait.
    texas_snap = _load_snapshot(TEXAS_SNAPSHOT_PATH)
    if texas_snap is not None:
        state["cache_texas"] = {
            "data": texas_snap,
            "expires": datetime.min.replace(tzinfo=timezone.utc),
        }
        print(f"Loaded Texas predictions snapshot ({len(texas_snap.get('tracts', []))} tracts)")

    quantum_snap = _load_snapshot(QUANTUM_SNAPSHOT_PATH)
    if quantum_snap is not None:
        global _quantum_cache
        _quantum_cache = {
            "data": quantum_snap,
            "expires": datetime.min.replace(tzinfo=timezone.utc),
        }
        print("Loaded quantum sensor-placement snapshot")

    # NOAA HMS smoke polygon cache — hydrate from disk snapshot so cold-start
    # users get an instant red-polygon overlay (stale-while-revalidate).
    hms_snap = _load_snapshot(HMS_SNAPSHOT_PATH)
    if hms_snap is not None:
        global _hms_cache
        _hms_cache = {
            "data": hms_snap,
            "expires": datetime.min.replace(tzinfo=timezone.utc),
        }
        print(f"Loaded HMS snapshot ({hms_snap.get('count', 0)} polygons)")

    # Live PurpleAir cache — hydrate from disk snapshot (already-expired) so the
    # first prediction after cold start can use last-known-good live air rather
    # than blocking on the PurpleAir round-trip.
    pa_snap = _load_snapshot(PURPLEAIR_SNAPSHOT_PATH)
    if pa_snap is not None:
        global _purpleair_cache
        _purpleair_cache = {
            "data": pa_snap,
            "expires": datetime.min.replace(tzinfo=timezone.utc),
        }
        print(f"Loaded PurpleAir snapshot ({pa_snap.get('count', 0)} live sensors)")

    # Open-Meteo grid snapshots — hydrate as already-expired: the first cycle
    # tries a fresh fetch, and if the daily quota is exhausted it falls back to
    # this stale grid instead of training-median fills (much better than nothing
    # after a redeploy mid-outage).
    wg_snap = _load_snapshot(WEATHER_GRID_SNAPSHOT_PATH)
    if wg_snap is not None:
        global _wgrid_cache
        _wgrid_cache = {"data": wg_snap, "expires": datetime.min.replace(tzinfo=timezone.utc)}
        print(f"Loaded weather-grid snapshot ({wg_snap.get('n_cells', 0)} cells)")
    aq_snap = _load_snapshot(AQ_SNAPSHOT_PATH)
    if aq_snap is not None:
        global _aq_cache
        _aq_cache = {"data": aq_snap, "expires": datetime.min.replace(tzinfo=timezone.utc)}
        print(f"Loaded air-quality snapshot ({aq_snap.get('n_cells', 0)} cells)")
    cond_snap = _load_snapshot(CONDITIONS_SNAPSHOT_PATH)
    if cond_snap is not None:
        global _conditions_cache
        _conditions_cache = {"data": cond_snap, "expires": datetime.min.replace(tzinfo=timezone.utc)}
        print(f"Loaded conditions snapshot ({len(cond_snap.get('by_key', {}))} cells)")
    if os.environ.get("PURPLEAIR_API_KEY", "").strip():
        print("PurpleAir: using PURPLEAIR_API_KEY from environment.")
    else:
        print("PurpleAir: PURPLEAIR_API_KEY not set — using committed fallback key. "
              "Live same-day neighbor features are ACTIVE. Set/rotate the key in "
              "Render env to override.")

    # Kick off background precompute to warm the Texas predictions cache so
    # initial requests from the frontend don't time out when computing all tracts.
    async def _precompute_texas():
        try:
            print("Background: warming city caches (fast)...")
            # Warm per-city caches (faster than a full Texas pass)
            for city in CITIES:
                try:
                    await get_city_predictions(city)
                    print(f"  warmed cache for {city}")
                except Exception as e:
                    print(f"  city precompute {city} failed: {e}")

            # Defer full Texas precompute so startup stays responsive
            async def _deferred_full_texas():
                await asyncio.sleep(5)
                try:
                    print("Background: computing full Texas predictions (deferred)...")
                    await _compute_texas_predictions()
                    print("Background: Texas precompute complete.")
                except Exception as e:
                    print(f"Background Texas precompute failed: {e}")

                # Warm quantum cache too so the first Sensors-tab visitor gets it
                # in <1s instead of waiting ~95s for the annealing run.
                try:
                    print("Background: warming quantum sensor placement...")
                    await _revalidate_quantum_background()
                except Exception as e:
                    print(f"Background quantum precompute failed: {e}")

                # Warm NOAA HMS smoke layer (small fetch, daily-cadence data).
                try:
                    print("Background: warming NOAA HMS smoke layer...")
                    await _revalidate_hms_background()
                except Exception as e:
                    print(f"Background HMS precompute failed: {e}")

            asyncio.create_task(_deferred_full_texas())

            # Warm the live PurpleAir cache FIRST (before the deferred full-Texas
            # recompute below) so the neighbor features reflect current air.
            try:
                print("Background: warming live PurpleAir sensors...")
                await _revalidate_purpleair_background()
            except Exception as e:
                print(f"Background PurpleAir precompute failed: {e}")
        except Exception as e:
            print(f"Background Texas precompute failed: {e}")

    # Schedule background precompute (do not await)
    try:
        asyncio.create_task(_precompute_texas())
    except Exception:
        # In some environments create_task must be called differently; ignore failures
        pass

    # Keepalive loop: pings an external URL every ~14 min so Render's free tier
    # never spins us down. Internal health() calls do NOT count as external traffic
    # to Render — we must actually hit the public URL.
    async def _keepalive_loop():
        # Prefer explicit override, otherwise auto-detect Render's public URL.
        url = (
            os.environ.get("KEEPALIVE_URL")
            or os.environ.get("RENDER_EXTERNAL_URL")
        )
        if url:
            url = url.rstrip("/") + "/api/health"
        try:
            interval = int(os.environ.get("KEEPALIVE_INTERVAL", "840"))  # 14 min
        except Exception:
            interval = 840

        if not url:
            print("Keepalive: no KEEPALIVE_URL or RENDER_EXTERNAL_URL set — disabled. "
                  "Cold starts will happen after 15 min of inactivity.")
            return

        print(f"Keepalive: will self-ping {url} every {interval}s")
        async with httpx.AsyncClient(timeout=10.0) as client:
            while True:
                try:
                    try:
                        resp = await client.get(url)
                        print(f"Keepalive: pinged {url} status={resp.status_code}")
                    except Exception as e:
                        print(f"Keepalive HTTP error pinging {url}: {e}")
                    await asyncio.sleep(interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"Keepalive loop error: {e}")
                    await asyncio.sleep(60)

    try:
        asyncio.create_task(_keepalive_loop())
    except Exception:
        pass

    # Initialize a simple SQLite visits DB for a persistent visit counter.
    try:
        def _init_visits_db():
            conn = sqlite3.connect(VISITS_DB, timeout=5)
            cur = conn.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS visits (id INTEGER PRIMARY KEY CHECK (id = 1), count INTEGER)")
            cur.execute("INSERT OR IGNORE INTO visits (id, count) VALUES (1, 0)")
            conn.commit()
            conn.close()
        await asyncio.to_thread(_init_visits_db)
        print(f"Visits DB initialized at {VISITS_DB}")
    except Exception as e:
        print(f"Failed to initialize visits DB: {e}")

    yield


app = FastAPI(title="Shared Skies API", lifespan=lifespan)

# CORS. This is a PUBLIC, read-only data API that the map frontend (Vercel) and
# any embedder fetch cross-origin, so the default stays permissive ("*"). To lock
# it down WITHOUT a code change, set ALLOWED_ORIGINS in the environment to a
# comma-separated allowlist, e.g.
#   ALLOWED_ORIGINS=https://sharedskiesinitiative.org,https://shared-skies-initiative.vercel.app
# NOTE: CORS is a browser-enforced READ guard — it does NOT stop a script/curl
# from POSTing /api/visit, so it is not what protects the visit counter (the
# 10-minute public display cache does that). Methods are limited to the verbs the
# API actually serves.
_origins_env = os.environ.get("ALLOWED_ORIGINS", "").strip()
ALLOWED_ORIGINS = [o.strip() for o in _origins_env.split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS", "HEAD"],
    allow_headers=["*"],
)

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── Utilities ─────────────────────────────────────────────────────────────────

# Weather fallback = exact TRAINING MEDIANS (and training units: °C, %, surface
# hPa, m/s, mm). The old fallback {72, 55, 1013, 8} was °F/MSL-shaped — values
# the model never saw in training (temperature median is 20.2 °C; pressure is
# SURFACE pressure ~991 hPa, not sea-level 1013).
WEATHER_FALLBACK = {
    "temperature": 20.2, "humidity": 65.0, "pressure": 991.3,
    "wind_speed": 5.5, "precipitation": 0.0,
}


DISPLAY_WEATHER_FALLBACK = {
    "temperature": round(WEATHER_FALLBACK["temperature"] * 9 / 5 + 32, 1),  # 68.4 °F
    "humidity":    WEATHER_FALLBACK["humidity"],
    "pressure":    1013.0,                                                  # MSL-style
    "wind_speed":  round(WEATHER_FALLBACK["wind_speed"] * 2.23694, 1),      # 12.3 mph
}


def _weather_fallback() -> dict:
    """Model-unit fallback weather (training medians) WITH a converted display
    dict. Display fallbacks must NEVER expose bare model units — that once
    rendered the 20.2 °C training median as '20 °F' on every tract."""
    w = dict(WEATHER_FALLBACK)
    w["display"] = dict(DISPLAY_WEATHER_FALLBACK)
    return w


async def _get_conditions_grid() -> dict | None:
    """Cached CURRENT-CONDITIONS cell grid (display units, TTL 60 min + disk
    snapshot, stale-on-failure with 15-min backoff). The ONLY source of
    'Current Conditions' shown in the UI — endpoints never call the weather API
    per request, so an API outage degrades to slightly-stale conditions instead
    of a frozen statewide constant."""
    global _conditions_cache
    now = datetime.now(timezone.utc)
    cached = _conditions_cache.get("data")
    if cached and cached.get("usable") and now < _conditions_cache.get("expires", datetime.min.replace(tzinfo=timezone.utc)):
        return cached
    lookup = state.get("tract_lookup")
    if lookup is None:
        return cached if cached and cached.get("usable") else None
    try:
        from backend.airquality import fetch_conditions_grid
        snap = await fetch_conditions_grid(lookup["lat"].values, lookup["lon"].values)
    except Exception as e:
        print(f"[conditions] fetch raised: {e}")
        snap = None
    if snap and snap.get("usable"):
        _conditions_cache = {"data": snap, "expires": now + timedelta(minutes=CONDITIONS_TTL_MIN)}
        _save_snapshot(CONDITIONS_SNAPSHOT_PATH, snap)
        return snap
    if cached and cached.get("usable"):
        print("[conditions] refetch failed — serving stale conditions (backoff 15 min)")
        _conditions_cache["expires"] = now + timedelta(minutes=15)
        return cached
    return None


async def get_display_weather(lat: float, lon: float) -> dict:
    """Current conditions for a location (display units), from the cached cell
    grid. Falls back: exact cell -> nearest cell -> static converted medians.
    Never calls the weather API directly for a single location."""
    grid = await _get_conditions_grid()
    if grid and grid.get("by_key"):
        from backend.airquality import conditions_cell_key
        by_key = grid["by_key"]
        v = by_key.get(conditions_cell_key(lat, lon))
        if v is None:
            # Nearest occupied cell (e.g. coastal/border rounding) — ~81 keys.
            try:
                best, best_d = None, float("inf")
                for k, vals in by_key.items():
                    kla, klo = (float(x) for x in k.split(","))
                    d = (kla - lat) ** 2 + (klo - lon) ** 2
                    if d < best_d:
                        best, best_d = vals, d
                v = best
            except Exception:
                v = None
        if v is not None:
            return dict(v)
    return dict(DISPLAY_WEATHER_FALLBACK)


async def get_click_weather(lat: float, lon: float) -> dict:
    """EXACT-location current weather (display units) for a clicked tract, with
    a per-0.1°-cell shared cache. On cooldown or any failure, falls back to the
    coarse conditions grid (location-varying) — so a quota outage degrades to
    'slightly coarse' instead of the old 'frozen statewide constant'."""
    key = f"{round(float(lat), 1):.1f},{round(float(lon), 1):.1f}"
    now = datetime.now(timezone.utc)
    hit = _click_weather_cache.get(key)
    if hit:
        _click_weather_cache.move_to_end(key)
        if now < hit["expires"]:
            return hit["weather"]
    if now < _click_weather_cooldown["until"]:
        return await get_display_weather(lat, lon)  # API recently failed
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get("https://api.open-meteo.com/v1/forecast", params={
                "latitude": lat, "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m",
                "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "timezone": "UTC",
            })
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")
        cur = r.json().get("current", {})
        if cur.get("temperature_2m") is None:
            raise RuntimeError("no current data")
        weather = {
            "temperature": round(float(cur["temperature_2m"]), 1),
            "humidity":    round(float(cur.get("relative_humidity_2m") or 0), 1),
            "pressure":    round(float(cur.get("pressure_msl") or 1013.0), 1),
            "wind_speed":  round(float(cur.get("wind_speed_10m") or 0), 1),
        }
        _click_weather_cache[key] = {"weather": weather, "expires": now + timedelta(minutes=CLICK_WEATHER_TTL_MIN)}
        if len(_click_weather_cache) > 600:
            _click_weather_cache.popitem(last=False)
        return weather
    except Exception as e:
        # Trip a 30-min cooldown so we don't hammer a quota-exhausted API on
        # every click; serve the grid (still varies by location) meanwhile.
        _click_weather_cooldown["until"] = now + timedelta(minutes=30)
        print(f"[click-weather] exact fetch failed ({e}); cooldown 30m, using conditions grid")
        return await get_display_weather(lat, lon)


def get_temporal(tz: str = "America/Chicago") -> dict:
    """Return current temporal features in specified timezone."""
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo(tz))
    return {
        "month":       now.month,
        "hour":        now.hour,
        "dow":         now.weekday(),
        "day_of_year": now.timetuple().tm_yday,
    }


_ZERO_WEIGHT_EPS = 1e-12


def _active_models(weights: dict, models) -> list:
    """Model names with non-negligible ensemble weight. Skips zero-weight models
    (e.g. CatBoost at weight 0.0 in the v6 simplex blend) so we don't pay for a
    .predict() call + its RAM that contributes nothing on the 512MB free tier.
    The weighted sum is numerically identical with the zero-weight terms removed."""
    active = [n for n in models if abs(weights.get(n, 0.0)) > _ZERO_WEIGHT_EPS]
    return active or list(models)  # defensive: never return an empty model set


def run_prediction(tract_row: pd.Series, weather: dict, temporal: dict) -> float:
    bundle = state["bundle"]
    features = bundle["feature_names"]
    weights  = bundle["weights"]
    models   = bundle["models"]

    row = _compute_v3_shared(weather, temporal)

    # Add tract-level features
    for feat in features:
        if feat not in row:
            row[feat] = float(tract_row.get(feat, 0.0) or 0.0)

    # v3 spatial features for single tract
    if "elevation" in features and "elevation" not in row:
        elev_path = os.path.join(ROOT, "pipeline", "elevations.json")
        if os.path.exists(elev_path):
            with open(elev_path) as f:
                elev_data = json.load(f)
            geoid = str(tract_row.get("GEOID", ""))
            row["elevation"] = elev_data.get("tracts", {}).get(geoid, 0.0)
    if "dist_to_coast" in features:
        row.setdefault("dist_to_coast", abs(float(tract_row.get("lon", -97)) - (-94.0)))
    if "dist_to_border" in features:
        row.setdefault("dist_to_border", abs(float(tract_row.get("lat", 31)) - 26.0))

    X = np.array([[row.get(f, 0.0) for f in features]])
    pred = sum(weights[n] * models[n].predict(X)[0] for n in _active_models(weights, models))
    return max(0.0, float(pred))


def _compute_v3_shared(weather: dict, temporal: dict) -> dict:
    """Compute all derived features that are shared across tracts (weather + temporal).
    Defaults are the training medians in TRAINING UNITS (°C, %, surface hPa, m/s)."""
    month = temporal.get("month", 1)
    dow = temporal.get("dow", 0)
    doy = temporal.get("day_of_year", 1)
    temp = weather.get("temperature", WEATHER_FALLBACK["temperature"])
    hum = weather.get("humidity", WEATHER_FALLBACK["humidity"])
    ws = weather.get("wind_speed", WEATHER_FALLBACK["wind_speed"])
    precip = weather.get("precipitation", 0)

    shared = {**weather, **temporal}
    shared.pop("display", None)  # UI-only sub-dict, never a model feature
    shared["precipitation"] = precip
    # Cyclical
    shared["month_sin"] = np.sin(2 * np.pi * month / 12)
    shared["month_cos"] = np.cos(2 * np.pi * month / 12)
    shared["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    shared["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    shared["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    shared["doy_cos"] = np.cos(2 * np.pi * doy / 365)
    # Interactions: ONLY the two the deployed model actually uses (features 36-37
    # in feature_names.json). Formulas mirror pipeline/03_train_enhanced.py exactly.
    # run_prediction() (single-tract cold-start path) reads these from `shared`;
    # without them the model would silently receive 0.0 → train/serve skew. The
    # three previously-computed extras (wind_x_season / humidity_x_season /
    # precip_x_temp) were never model features — dead compute — so they stay gone.
    shared["temp_x_humidity"] = temp * hum / 100.0
    shared["wind_x_temp"] = ws * temp / 100.0
    # Rolling weather (at inference we only have current → use current as proxy)
    shared["temperature_3d"] = temp
    shared["humidity_3d"] = hum
    shared["wind_speed_3d"] = ws
    return shared


def _add_v3_spatial(df: pd.DataFrame) -> pd.DataFrame:
    """Add v3 spatial features to tract lookup DataFrame if not present."""
    bundle = state.get("bundle", {})

    # Elevation
    if "elevation" not in df.columns:
        elev_path = os.path.join(ROOT, "pipeline", "elevations.json")
        if os.path.exists(elev_path):
            with open(elev_path) as f:
                elev_data = json.load(f)
            tract_elev = elev_data.get("tracts", {})
            df["elevation"] = df["GEOID"].map(tract_elev).fillna(0.0).astype(float)

    # Distance to coast / border
    if "dist_to_coast" not in df.columns:
        df["dist_to_coast"] = np.abs(pd.to_numeric(df["lon"], errors="coerce").fillna(-97) - (-94.0))
    if "dist_to_border" not in df.columns:
        df["dist_to_border"] = np.abs(pd.to_numeric(df["lat"], errors="coerce").fillna(31) - 26.0)

    # Spatial cluster distance
    if "dist_to_cluster_center" not in df.columns:
        cluster_path = os.path.join(MODELS_DIR, "spatial_clusters.joblib")
        if os.path.exists(cluster_path):
            kmeans = joblib.load(cluster_path)
            lats = pd.to_numeric(df["lat"], errors="coerce").fillna(31).values
            lons = pd.to_numeric(df["lon"], errors="coerce").fillna(-97).values
            coords = np.column_stack([lats, lons])
            clusters = kmeans.predict(coords)
            centers = kmeans.cluster_centers_
            df["dist_to_cluster_center"] = np.sqrt(
                (lats - centers[clusters, 0])**2 + (lons - centers[clusters, 1])**2
            )
        else:
            df["dist_to_cluster_center"] = 0.0

    # Urban index
    if "urban_index" not in df.columns:
        df["urban_index"] = (
            pd.to_numeric(df.get("traffic_proximity", 0), errors="coerce").fillna(0) * 0.4 +
            pd.to_numeric(df.get("diesel_pm_proximity", 0), errors="coerce").fillna(0) * 0.3 +
            pd.to_numeric(df.get("ejf_score", 0), errors="coerce").fillna(0) * 0.3
        ) / 100.0

    return df


def run_predictions_batch(df: pd.DataFrame, weather: dict, temporal: dict) -> np.ndarray:
    """
    Vectorized batch prediction. Handles v1/v2/v3 feature sets automatically
    based on what the loaded model expects.
    """
    bundle   = state["bundle"]
    features = bundle["feature_names"]
    weights  = bundle["weights"]
    models   = bundle["models"]

    shared = _compute_v3_shared(weather, temporal)

    # Add v3 spatial features only if the model needs them AND they're not
    # already present in the enriched tract_lookup. The lifespan call to
    # _enrich_tract_lookup_with_distances populates dist_to_coast etc. on
    # startup, so this legacy path is a no-op for v3b. The `f not in df.columns`
    # guard also keeps us safe from latent bugs inside _add_v3_spatial (e.g.
    # the now-fixed MODELS_DIR NameError) when the columns are already there.
    needs_v3_spatial = any(
        f in features and f not in df.columns
        for f in ["elevation", "dist_to_coast", "urban_index", "dist_to_border", "dist_to_cluster_center"]
    )
    if needs_v3_spatial:
        df = _add_v3_spatial(df)

    # Lookup uses lat/lon; v2 model was trained with latitude/longitude column names
    feat_set = set(features)
    if "latitude" in feat_set and "lat" in df.columns and "latitude" not in df.columns:
        df = df.copy()
        df["latitude"] = df["lat"]
    if "longitude" in feat_set and "lon" in df.columns and "longitude" not in df.columns:
        df = df.copy() if "latitude" not in df.columns else df
        df["longitude"] = df["lon"]

    n = len(df)
    X = np.zeros((n, len(features)), dtype=np.float64)

    # Use the EXACT training-time fill values for any feature missing/NaN at
    # inference, so the model never sees a value distribution it wasn't trained
    # on. bundle['feature_fill'] holds the training medians (and hms_smoke=0).
    feature_fill = bundle.get("feature_fill", {})

    # Precedence: per-tract df column FIRST (live gridded weather / neighbors /
    # HMS / AOD vary by tract), then the shared scalar (single-point weather +
    # temporal), then the training-median fill. df-first matters: the texas path
    # now writes per-tract weather columns that must not be shadowed by the
    # single shared scalar.
    # Vectorized DataFrame cast
    df_cols = [f for f in features if f in df.columns]
    if df_cols:
        df[df_cols] = df[df_cols].apply(pd.to_numeric, errors="coerce")

    for i, feat in enumerate(features):
        fill = float(feature_fill.get(feat, 0.0))
        if feat in df_cols:
            col = df[feat]
            if feat in shared:
                col = col.fillna(float(shared[feat]))
            X[:, i] = col.fillna(fill).values
        elif feat in shared:
            X[:, i] = shared[feat]
        else:
            X[:, i] = fill  # feature entirely absent -> training fill, not 0

    preds = sum(weights[name] * models[name].predict(X) for name in _active_models(weights, models))
    # v3 models fit on log1p(pm25); invert before returning. Pre-v3 bundles
    # don't have this key so this is a no-op for them.
    if bundle.get("target_transform") == "log1p":
        preds = np.expm1(preds)
    return np.maximum(0.0, preds)


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb):
    """Convert RGB tuple to hex color. Uses round-half-up (int(c+0.5)) — matching
    JS Math.round for non-negative values — so the backend's per-tract `color`
    is BYTE-IDENTICAL to the frontend legend's pm25Color (avoids Python's
    banker's rounding diverging by 1 LSB at exact .5 midpoints)."""
    return '#{:02x}{:02x}{:02x}'.format(
        max(0, min(255, int(rgb[0] + 0.5))),
        max(0, min(255, int(rgb[1] + 0.5))),
        max(0, min(255, int(rgb[2] + 0.5))),
    )


def interpolate_color(color1, color2, factor):
    """Interpolate between two colors based on factor (0-1)."""
    factor = max(0, min(1, factor))
    rgb1 = hex_to_rgb(color1)
    rgb2 = hex_to_rgb(color2)

    rgb = tuple(rgb1[i] + (rgb2[i] - rgb1[i]) * factor for i in range(3))
    return rgb_to_hex(rgb)


def pm25_color_gradient(pm25: float) -> str:
    """Get gradient color based on PM2.5 value.

    Bands: Good 0-9 (green), Moderate 9-13 (yellow), Elevated 13-17 (orange),
    High 17+ (red→dark red). Each band darkens toward its upper boundary, and
    the High band keeps darkening with concentration so wildfire-smoke days read
    dramatically dark. 9 = U.S. EPA annual NAAQS (2024).
    """
    if pm25 < 0:
        pm25 = 0

    # Green: 0.0-9.0  (within the EPA annual standard)
    if pm25 <= 9.0:
        factor = pm25 / 9.0
        return interpolate_color("#90EE90", "#00b894", factor)

    # Yellow: 9.0-13.0
    elif pm25 <= 13.0:
        factor = (pm25 - 9.0) / 4.0
        return interpolate_color("#FFFF99", "#FFD700", factor)

    # Orange: 13.0-17.0
    elif pm25 <= 17.0:
        factor = (pm25 - 13.0) / 4.0
        return interpolate_color("#FFB347", "#E8590C", factor)

    # Red → dark red: 17.0+  (darkens with concentration; saturates ~55, the EPA
    # 24-hr Unhealthy threshold, so smoke/dust days read dramatically dark).
    else:
        factor = min(1.0, (pm25 - 17.0) / 38.0)
        return interpolate_color("#FF6B6B", "#800000", factor)


# U.S. EPA PM2.5 AQI breakpoints (May 2024 revision). Used to show an
# AQI-equivalent next to our µg/m³ so users can compare directly with apps that
# display AQI (PurpleAir map, AirNow). (C_lo, C_hi, AQI_lo, AQI_hi)
_EPA_AQI_BREAKPOINTS = [
    (0.0,   9.0,   0,   50),
    (9.1,   35.4,  51,  100),
    (35.5,  55.4,  101, 150),
    (55.5,  125.4, 151, 200),
    (125.5, 225.4, 201, 300),
    (225.5, 325.4, 301, 500),
]


def pm25_to_epa_aqi(pm25: float) -> int:
    """Convert PM2.5 (µg/m³) to the U.S. EPA AQI (2024 breakpoints)."""
    c = max(0.0, math.floor(float(pm25) * 10) / 10)  # EPA truncates to 0.1
    for c_lo, c_hi, a_lo, a_hi in _EPA_AQI_BREAKPOINTS:
        if c <= c_hi:
            c_lo_eff = c_lo if c >= c_lo else 0.0
            return int(round((a_hi - a_lo) / (c_hi - c_lo_eff) * (c - c_lo_eff) + a_lo))
    return 500


def pm25_info(pm25: float) -> dict:
    """Get category info. Bands: Good 0-9 / Moderate 9-13 / Elevated 13-17 /
    High 17+. 9 µg/m³ = U.S. EPA annual NAAQS; 15 µg/m³ = WHO 24-hr guideline."""
    color = pm25_color_gradient(pm25)

    if pm25 <= 9.0:
        return {
            "category": "Good",
            "color": color,
            "aqi_range": "0–9",
            "health_msg": "Air quality is good — within the U.S. EPA annual PM2.5 standard (9 µg/m³).",
        }
    elif pm25 <= 13.0:
        return {
            "category": "Moderate",
            "color": color,
            "aqi_range": "9–13",
            "health_msg": "Moderate — above the EPA annual standard. Unusually sensitive people may want to limit prolonged outdoor exertion.",
        }
    elif pm25 <= 17.0:
        return {
            "category": "Elevated",
            "color": color,
            "aqi_range": "13–17",
            "health_msg": "Elevated — above the WHO 24-hour guideline (15 µg/m³). Sensitive groups should limit prolonged outdoor activity.",
        }
    else:
        return {
            "category": "High",
            "color": color,
            "aqi_range": "17+",
            "health_msg": "⚠️ High — everyone may begin to feel effects; sensitive groups are at greater risk. Often driven by wildfire smoke or dust.",
        }


def _safe_float(val) -> float:
    """Convert to float, handle NaN and None."""
    if val is None:
        return 0.0
    try:
        f = float(val)
        return 0.0 if np.isnan(f) else round(f, 1)
    except (TypeError, ValueError):
        return 0.0


async def get_city_predictions(city: str):
    """
    Compute PM2.5 predictions for all tracts in a city.
    Results are cached per-city for 30 minutes.
    """
    if city not in CITIES:
        raise HTTPException(404, f"Unknown city: {city}. Available: {', '.join(CITIES.keys())}")

    city_config = CITIES[city]
    cache_key = f"cache_{city}"
    cache = state.get(cache_key, {})
    now = datetime.now(timezone.utc)

    # Check cache
    if cache.get("data") and now < cache.get("expires", datetime.min.replace(tzinfo=timezone.utc)):
        return cache["data"]

    if state.get("bundle") is None:
        raise HTTPException(503, "Model not loaded. Run pipeline/02_train_model.py first.")
    if state.get("tract_lookup") is None:
        raise HTTPException(503, "Tract lookup not loaded. Run pipeline/01_build_tract_lookup.py first.")

    # CONSISTENCY: slice the texas-wide batch (live neighbors + HMS + AOD +
    # gridded weather) instead of recomputing here. The old path called
    # run_predictions_batch with NO live features, so city views predicted from
    # training medians — blind to same-day events and inconsistent with the map.
    texas = state.get("cache_texas", {}).get("data")
    if texas is None:
        texas = await _compute_texas_predictions()

    fips = city_config["fips"]
    tracts = [t for t in texas["tracts"] if str(t["geoid"]).startswith(fips)]
    if not tracts:
        raise HTTPException(404, f"No tracts found for city: {city}")

    # City-center CURRENT conditions (display only, from the cached cell grid —
    # no direct API call).
    weather_display = await get_display_weather(*city_config["center"])

    avg_pm25 = round(float(np.mean([t["pm25"] for t in tracts])), 2)

    result = {
        "city":         city,
        "display_name": city_config["display_name"],
        "generated_at": texas.get("generated_at", now.isoformat()),
        "expires_at":   texas.get("expires_at", (now + timedelta(minutes=30)).isoformat()),
        "weather":      weather_display,
        "avg_pm25":     avg_pm25,
        "avg_epa_aqi":  pm25_to_epa_aqi(avg_pm25),
        "avg_info":     pm25_info(avg_pm25),
        "tract_count":  len(tracts),
        "data_sources": texas.get("data_sources", {}),
        "tracts":       tracts,
    }

    state[cache_key] = {"data": result, "expires": now + timedelta(minutes=30)}
    return result


# ── Persistent visits counter helpers ────────────────────────────────────────
# Priority: Upstash Redis (truly persistent across restarts) > SQLite (ephemeral)

def _get_visit_count_sync():
    conn = sqlite3.connect(VISITS_DB, timeout=5)
    cur = conn.cursor()
    cur.execute("SELECT count FROM visits WHERE id=1")
    row = cur.fetchone()
    conn.close()
    return int(row[0]) if row else 0


def _inc_visit_sync():
    conn = sqlite3.connect(VISITS_DB, timeout=5)
    cur = conn.cursor()
    cur.execute("UPDATE visits SET count = count + 1 WHERE id = 1")
    conn.commit()
    cur.execute("SELECT count FROM visits WHERE id=1")
    row = cur.fetchone()
    conn.close()
    return int(row[0]) if row else 0


async def _upstash_incr() -> int:
    """Atomically increment the visit counter in Upstash Redis and return the new value."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.post(
            f"{UPSTASH_URL}/incr/shared_skies_visits",
            headers={"Authorization": f"Bearer {UPSTASH_TOKEN}"},
        )
        resp.raise_for_status()
        return int(resp.json()["result"])


async def _upstash_get() -> int:
    """Get the current visit count from Upstash Redis."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.get(
            f"{UPSTASH_URL}/get/shared_skies_visits",
            headers={"Authorization": f"Bearer {UPSTASH_TOKEN}"},
        )
        resp.raise_for_status()
        val = resp.json().get("result")
        return int(val) if val is not None else 0


async def _get_public_visit_count() -> int:
    """Return the 10-min-cached visit count shown to users.
    Backend keeps incrementing the real count on every /api/visit; this layer
    only controls what's displayed so the number can't be watched ticking up.
    All clients in the same 10-min window see the same value."""
    global _public_visits_cache
    now = datetime.now(timezone.utc)
    cached_count = _public_visits_cache.get("count")
    expires_at = _public_visits_cache.get("expires")

    if cached_count is not None and expires_at is not None and now < expires_at:
        return cached_count

    # Cache miss or expired — read the true count and refresh the cache.
    try:
        if UPSTASH_URL and UPSTASH_TOKEN:
            count = await _upstash_get()
        else:
            count = await asyncio.to_thread(_get_visit_count_sync)
    except Exception as e:
        print(f"Public visit cache refresh failed: {e}")
        # If reading fails, keep showing the last good value rather than zero.
        return cached_count if cached_count is not None else 0

    _public_visits_cache = {
        "count": count,
        "expires": now + timedelta(minutes=PUBLIC_VISITS_TTL_MIN),
    }
    return count


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    """Health + live-data-source status. The data_sources block makes silent
    degradation visible: if PurpleAir/HMS/weather quietly fail, predictions
    fall back to climatology/medians — operators need to SEE that."""
    pa = _purpleair_cache.get("data") or {}
    texas = state.get("cache_texas", {}).get("data") or {}
    return {
        "status": "ok",
        "model_loaded": state.get("bundle") is not None,
        "model_version": (state.get("bundle") or {}).get("version"),
        "lookup_loaded": state.get("tract_lookup") is not None,
        "available_cities": list(CITIES.keys()),
        "live_purpleair_sensors": len(pa.get("sensors", [])) if pa.get("usable") else 0,
        "purpleair_fetched_at": pa.get("fetched_at"),
        "predictions_generated_at": texas.get("generated_at"),
        "conditions_fetched_at": (_conditions_cache.get("data") or {}).get("fetched_at"),
        "data_sources": texas.get("data_sources", {}),
    }


@app.get("/api/cities")
async def list_cities():
    """List available cities with metadata."""
    return {
        "cities": [
            {
                "id": city_id,
                "name": config["display_name"],
                "center": config["center"],
                "fips": config["fips"]
            }
            for city_id, config in CITIES.items()
        ]
    }


@app.post("/api/visit")
async def record_visit():
    """Increment the real visit counter (Upstash if configured, else SQLite),
    but return the *publicly-cached* total so users can't watch the number
    tick up by refreshing. Real count stays accurate in storage."""
    try:
        if UPSTASH_URL and UPSTASH_TOKEN:
            await _upstash_incr()
        else:
            await asyncio.to_thread(_inc_visit_sync)
    except Exception as e:
        print(f"Visit increment error (primary): {e}")
        # Last-resort fallback — try SQLite so we at least record something
        try:
            await asyncio.to_thread(_inc_visit_sync)
        except Exception as e2:
            print(f"Visit increment error (fallback): {e2}")
    # Return the display-cached count (refreshes every PUBLIC_VISITS_TTL_MIN)
    try:
        return {"visits": await _get_public_visit_count()}
    except Exception:
        return {"visits": 0}


@app.get("/api/metrics")
async def get_metrics():
    """Return the publicly-cached visit count (refreshes every ~10 min).
    For the raw real-time count, read Upstash/SQLite directly."""
    try:
        return {"visits": await _get_public_visit_count()}
    except Exception as e:
        print(f"Metrics read error: {e}")
        return {"visits": 0}


def _grid_matches(snap: dict | None, n: int) -> bool:
    """A cached per-tract grid is reusable only if it was built for the same
    tract list length (the tract lookup is static, so length is a sufficient key)."""
    if not snap or not snap.get("usable"):
        return False
    feats = snap.get("features") or {}
    arr = next(iter(feats.values()), None)
    return arr is not None and len(arr) == n


async def _get_weather_grid(lats, lons) -> dict | None:
    """Cached per-tract daily-mean weather grid (TTL 6h + disk snapshot).
    On refetch failure, serves the stale grid (daily means age gracefully) and
    backs off 30 min — never hammers a quota-exhausted API every cycle."""
    global _wgrid_cache
    now = datetime.now(timezone.utc)
    cached = _wgrid_cache.get("data")
    if _grid_matches(cached, len(lats)) and now < _wgrid_cache.get("expires", datetime.min.replace(tzinfo=timezone.utc)):
        return cached
    try:
        from backend.airquality import fetch_weather_grid
        snap = await fetch_weather_grid(lats, lons)
    except Exception as e:
        print(f"[weather-grid] fetch raised: {e}")
        snap = None
    if snap and snap.get("usable"):
        _wgrid_cache = {"data": snap, "expires": now + timedelta(minutes=OPENMETEO_GRID_TTL_MIN)}
        _save_snapshot(WEATHER_GRID_SNAPSHOT_PATH, snap)
        return snap
    if _grid_matches(cached, len(lats)):
        print("[weather-grid] refetch failed — serving stale cached grid (backoff 30 min)")
        _wgrid_cache["expires"] = now + timedelta(minutes=30)
        return cached
    return None


async def _get_aq_snapshot(lats, lons, include_met: bool) -> dict | None:
    """Cached per-tract CAMS air-quality snapshot (TTL 6h + disk snapshot,
    stale-on-failure). Was refetched EVERY 30-min cycle before — the main
    Open-Meteo quota burner."""
    global _aq_cache
    now = datetime.now(timezone.utc)
    cached = _aq_cache.get("data")
    if _grid_matches(cached, len(lats)) and now < _aq_cache.get("expires", datetime.min.replace(tzinfo=timezone.utc)):
        return cached
    try:
        from backend.airquality import fetch_airquality_snapshot
        snap = await fetch_airquality_snapshot(lats, lons, include_met=include_met)
    except Exception as e:
        print(f"[airquality] fetch raised: {e}")
        snap = None
    if snap and snap.get("usable"):
        _aq_cache = {"data": snap, "expires": now + timedelta(minutes=OPENMETEO_GRID_TTL_MIN)}
        _save_snapshot(AQ_SNAPSHOT_PATH, snap)
        return snap
    if _grid_matches(cached, len(lats)):
        print("[airquality] refetch failed — serving stale cached snapshot (backoff 30 min)")
        _aq_cache["expires"] = now + timedelta(minutes=30)
        return cached
    return None


async def _compute_texas_predictions() -> dict:
    """Heavy path: actually compute predictions for every TX tract, update cache + snapshot."""
    if state.get("bundle") is None:
        raise HTTPException(503, "Model not loaded. Run pipeline/02_train_model.py first.")
    if state.get("tract_lookup") is None:
        raise HTTPException(503, "Tract lookup not loaded. Run pipeline/01_build_tract_lookup.py first.")

    lookup = state["tract_lookup"]
    tracts = []
    now = datetime.now(timezone.utc)

    print(f"Generating predictions for all {len(lookup)} Texas tracts (vectorized)...")
    temporal = get_temporal("America/Chicago")

    # Work on a COPY so concurrent per-city requests never see half-updated columns.
    lookup_reset = lookup.reset_index(drop=True)

    # ── Per-tract DAILY-MEAN weather (training-consistent, CACHED 6h) ───────
    # Training weather is per-sensor-location daily aggregates (°C, %, surface
    # hPa, m/s, mm). One batched Open-Meteo call over a coarse grid, cached with
    # a 6h TTL + disk snapshot (daily means don't change every 30 min, and
    # refetching per cycle exhausted the free API quota).
    weather_grid_cells = 0
    weather = _weather_fallback()  # model-unit statewide backfill scalars
    wsnap = await _get_weather_grid(
        lookup_reset["lat"].values, lookup_reset["lon"].values)
    if wsnap is not None:
        lookup_reset = lookup_reset.copy()
        for f in ("temperature", "humidity", "pressure", "wind_speed", "precipitation"):
            vals = wsnap["features"].get(f)
            if vals is not None:
                # None -> NaN; run_predictions_batch backfills with the
                # statewide scalar, then the training median.
                lookup_reset[f] = pd.Series(vals, dtype="float64").values
                # Statewide backfill scalar = today's median across the grid
                # (model units) — better than the all-time training median.
                med = float(np.nanmedian(lookup_reset[f].values))
                if np.isfinite(med):
                    weather[f] = med
        # Per-tract interactions (must mirror training formulas exactly).
        lookup_reset["temp_x_humidity"] = (
            lookup_reset["temperature"] * lookup_reset["humidity"] / 100.0)
        lookup_reset["wind_x_temp"] = (
            lookup_reset["wind_speed"] * lookup_reset["temperature"] / 100.0)
        weather_grid_cells = int(wsnap.get("n_cells", 0))
        print(f"Gridded daily-mean weather applied ({weather_grid_cells} cells): "
              f"T median={lookup_reset['temperature'].median():.1f}°C, "
              f"wind median={lookup_reset['wind_speed'].median():.1f} m/s")
    else:
        print("Weather grid unavailable — training-median weather backfill in effect.")

    # ── Live neighbor-feature recompute (the fix for scrambled predictions) ──
    # The multi-radius neighbor PM features (~dominant model signal) MUST reflect
    # same-day air to match training. Recompute them from LIVE PurpleAir sensors
    # here, every prediction cycle. If live data is unavailable (no key / API
    # down), we leave the static climatological columns from
    # _enrich_tract_lookup_with_distances in place — stable and non-scrambled.
    live_sensor_count = 0
    try:
        live_sensors = get_live_purpleair_sensors()
        if live_sensors:
            from backend.purpleair import compute_neighbor_features
            nbr_feats = await asyncio.to_thread(
                compute_neighbor_features,
                lookup_reset["lat"].values, lookup_reset["lon"].values, live_sensors,
            )
            if nbr_feats is not None:
                lookup_reset = lookup_reset.copy()
                for fname, arr in nbr_feats.items():
                    lookup_reset[fname] = arr
                live_sensor_count = len(live_sensors)
                nbr_mean = nbr_feats["nbr_pm25_50km"]
                nbr_count = nbr_feats["nbr_count_50km"]
                print(f"Live multi-radius neighbor features applied: {live_sensor_count} sensors, "
                      f"nbr_pm25_50km mean={float(np.mean(nbr_mean)):.2f}, "
                      f"50km coverage={(nbr_count>0).sum()/len(nbr_count)*100:.1f}%")
        else:
            print("No live PurpleAir data — using static climatological neighbor features.")
    except Exception as e:
        print(f"Live neighbor recompute failed ({e}); using static neighbor features.")

    # ── Live HMS smoke feature (hms_smoke) ──────────────────────────────────
    # Compute the SAME point-in-polygon smoke density the model trained on, for
    # every tract centroid, using today's cached HMS polygons. Uses the shared
    # backend.hms helpers so train/inference are byte-identical. Tracts under no
    # smoke polygon get 0 (the training fill). Only runs if the model uses it.
    features_needed = set(state.get("bundle", {}).get("feature_names", []))
    if "hms_smoke" in features_needed:
        try:
            hms_data = _hms_cache.get("data")
            polys = []
            if hms_data and hms_data.get("features"):
                from backend.hms import build_density_polygons
                polys = await asyncio.to_thread(build_density_polygons, hms_data["features"])
            if polys:
                from backend.hms import smoke_density_at
                def _smoke_all(lats, lons):
                    return np.array([smoke_density_at(polys, float(lo), float(la))
                                     for la, lo in zip(lats, lons)], dtype=np.int16)
                hms_vals = await asyncio.to_thread(
                    _smoke_all, lookup_reset["lat"].values, lookup_reset["lon"].values)
                lookup_reset = lookup_reset.copy()
                lookup_reset["hms_smoke"] = hms_vals
                print(f"HMS smoke feature applied: {(hms_vals>0).sum()} tracts under smoke "
                      f"(max tier {int(hms_vals.max()) if len(hms_vals) else 0})")
            else:
                lookup_reset = lookup_reset.copy()
                lookup_reset["hms_smoke"] = 0  # no polygons today = no smoke
        except Exception as e:
            print(f"HMS smoke feature failed ({e}); defaulting to 0.")
            lookup_reset = lookup_reset.copy()
            lookup_reset["hms_smoke"] = 0

    # ── Live CAMS air-quality (+ met PBL proxies only if the model uses them) ──
    # aod/cams_pm25 per tract, computed the SAME way as training (daily-mean of
    # today's hourly AOD), CACHED 6h + disk snapshot. The met call (shortwave/
    # et0/cloud_cover) is skipped for v6 — those features were dropped, and the
    # 254-location met call every cycle was the main api.open-meteo.com quota
    # burner. Missing values fall back to the training median fill.
    aq_feats = {"aod", "cams_pm25", "dust", "shortwave", "et0", "cloud_cover"}
    if aq_feats & features_needed:
        met_needed = bool({"shortwave", "et0", "cloud_cover"} & features_needed)
        snap = await _get_aq_snapshot(
            lookup_reset["lat"].values, lookup_reset["lon"].values,
            include_met=met_needed)
        if snap is not None:
            lookup_reset = lookup_reset.copy()
            applied = []
            for f in aq_feats & features_needed:
                vals = snap["features"].get(f)
                if vals is not None:
                    # leave None -> NaN so run_predictions_batch fills with training median
                    lookup_reset[f] = pd.Series(vals, dtype="float64").values
                    applied.append(f)
            print(f"Air-quality features applied ({snap['n_cells']} cells): {applied}")
        else:
            print("No usable live air-quality data — features will use training-median fill.")

    # Vectorized batch prediction — single model.predict() call for all 6,900+ tracts
    pm25_array = await asyncio.to_thread(run_predictions_batch, lookup_reset, weather, temporal)

    for i, (_, row) in enumerate(lookup.iterrows()):
        pm25 = round(float(pm25_array[i]), 2)
        info = pm25_info(pm25)
        tracts.append({
            "geoid":               row["GEOID"],
            "lat":                 round(float(row["lat"]), 6),
            "lon":                 round(float(row["lon"]), 6),
            "pm25":                pm25,
            "epa_aqi":             pm25_to_epa_aqi(pm25),
            "category":            info["category"],
            "color":               info["color"],
            "aqi_range":           info["aqi_range"],
            "health_msg":          info["health_msg"],
            "ejf_score":           _safe_float(row.get("ejf_score")),
            "pct_people_of_color": _safe_float(row.get("pct_people_of_color")),
            "pct_low_income":      _safe_float(row.get("pct_low_income")),
            "traffic_proximity":   _safe_float(row.get("traffic_proximity")),
            "superfund_proximity": _safe_float(row.get("superfund_proximity")),
            "diesel_pm_proximity": _safe_float(row.get("diesel_pm_proximity")),
            "pct_ling_isolated":   _safe_float(row.get("pct_ling_isolated")),
            "county":              str(row.get("CNTY_NAME", "Texas")),
        })

    avg_pm25 = round(float(np.mean([t["pm25"] for t in tracts])), 2)

    hms_data = _hms_cache.get("data") or {}
    result = {
        "region":       "texas",
        "display_name": "All of Texas",
        "generated_at": now.isoformat(),
        "expires_at":   (now + timedelta(minutes=30)).isoformat(),
        # Display weather (current conditions at Austin, °F/mph/MSL) for the
        # statewide overview panel; the model consumed per-tract gridded daily
        # means in training units.
        "weather":      await get_display_weather(30.2672, -97.7431),
        "avg_pm25":     avg_pm25,
        "avg_epa_aqi":  pm25_to_epa_aqi(avg_pm25),
        "avg_info":     pm25_info(avg_pm25),
        "tract_count":  len(tracts),
        # Data-source transparency: lets the UI (and operators) see whether the
        # dominant live signals were actually live for THIS prediction cycle.
        "data_sources": {
            "live_purpleair_sensors": live_sensor_count,
            "using_live_neighbors":   live_sensor_count > 0,
            "hms_polygons":           int(hms_data.get("count") or 0),
            "hms_data_date":          hms_data.get("data_date"),
            "weather_grid_cells":     weather_grid_cells,
        },
        "value_semantics": "Model-predicted 24-hour-average PM2.5 (µg/m³) per census tract",
        "tracts":       tracts,
    }

    state["cache_texas"] = {"data": result, "expires": now + timedelta(minutes=30)}
    state["cache_texas_by_geoid"] = {t["geoid"]: t for t in tracts}
    _save_snapshot(TEXAS_SNAPSHOT_PATH, result)
    print(f"✓ Texas predictions complete: {len(tracts)} tracts "
          f"(live sensors={live_sensor_count}, hms_polys={hms_data.get('count', 0)}, "
          f"weather_cells={weather_grid_cells})")
    return result


async def _revalidate_texas_background():
    """Recompute Texas predictions in the background without blocking any request.
    Refreshes the live PurpleAir cache first (if expired) so the neighbor
    features reflect current air."""
    if _revalidating.get("texas"):
        return
    _revalidating["texas"] = True
    try:
        # Refresh live air first if its cache has expired (cheap, ~1s; PURPLEAIR_CACHE_TTL_MIN, default 15m).
        pa_exp = _purpleair_cache.get("expires", datetime.min.replace(tzinfo=timezone.utc))
        if datetime.now(timezone.utc) >= pa_exp:
            await _revalidate_purpleair_background()
        await _compute_texas_predictions()
    except Exception as e:
        print(f"Texas revalidation failed: {e}")
    finally:
        _revalidating["texas"] = False


@app.get("/api/texas/predictions")
async def texas_predictions():
    """
    Returns PM2.5 predictions for all Texas census tracts.
    Stale-while-revalidate: cached data (incl. disk snapshot) is served instantly,
    and a background refresh kicks off when it's older than 30 min. Only the very
    first request after a clean deploy (no snapshot yet) waits for the compute.
    """
    cache = state.get("cache_texas", {})
    now = datetime.now(timezone.utc)
    cached_data = cache.get("data")
    expires_at = cache.get("expires", datetime.min.replace(tzinfo=timezone.utc))

    if cached_data is not None:
        if now >= expires_at:
            asyncio.create_task(_revalidate_texas_background())
        return cached_data

    # No cache, no snapshot — must compute synchronously (first request after clean deploy).
    return await _compute_texas_predictions()


@app.get("/api/{city}/predictions")
async def city_predictions(city: str):
    """
    Returns PM2.5 predictions for all census tracts in the specified city.
    Results are cached for 30 minutes.
    """
    return await get_city_predictions(city)


@app.get("/api/dallas/predictions")
async def dallas_predictions_legacy():
    """
    Legacy endpoint for backward compatibility.
    Redirects to /api/dallas/predictions via new multi-city system.
    """
    return await get_city_predictions("dallas")


@app.get("/api/tract/{geoid}")
async def get_tract(geoid: str):
    """Detailed view for a single census tract."""
    geoid = geoid.zfill(11)

    if state.get("tract_lookup") is None:
        raise HTTPException(503, "Tract lookup not loaded.")

    lookup = state["tract_lookup"]
    matches = lookup[lookup["GEOID"] == geoid]
    if matches.empty:
        raise HTTPException(404, f"Tract {geoid} not found.")

    row = matches.iloc[0]

    # Determine timezone based on tract's FIPS code
    tract_fips = geoid[:5]
    tz = "America/Chicago"
    for city, config in CITIES.items():
        if geoid.startswith(config["fips"]):
            tz = config["tz"]
            break

    # Local CURRENT conditions — display only (°F/mph). A clicked tract gets an
    # EXACT-location fetch (cached per ~11km, shared) so it matches the user's
    # local weather closely; on any failure it falls back to the coarse hourly
    # conditions grid (location-varying — never the old frozen statewide value).
    display_weather = await get_click_weather(float(row["lat"]), float(row["lon"]))

    # CONSISTENCY: the tract's PM2.5 comes from the SAME texas-wide batch the
    # map shows (live neighbors + HMS + AOD + gridded weather). The old path
    # recomputed it here through run_prediction() with climatological neighbors
    # and no live features — so clicking a tract showed a different number than
    # the map. Recompute is now only the cold-start fallback.
    cached = (state.get("cache_texas_by_geoid") or {}).get(geoid)
    if cached is not None:
        pm25 = float(cached["pm25"])
    elif state.get("bundle") is None:
        # Cold-start before the model finished loading — don't NoneType-crash.
        raise HTTPException(503, "Model still loading. Please retry in a moment.")
    else:
        # Cold-start only (no texas batch yet): model-unit fallback weather.
        temporal = get_temporal(tz)
        pm25 = run_prediction(row, _weather_fallback(), temporal)
    info = pm25_info(pm25)

    return {
        "geoid":               geoid,
        "lat":                 round(float(row["lat"]), 6),
        "lon":                 round(float(row["lon"]), 6),
        "pm25":                round(pm25, 2),
        "epa_aqi":             pm25_to_epa_aqi(pm25),
        "category":            info["category"],
        "color":               info["color"],
        "aqi_range":           info["aqi_range"],
        "health_msg":          info["health_msg"],
        "weather":             display_weather,
        "from_map_batch":      cached is not None,
        "ejf_score":           _safe_float(row.get("ejf_score")),
        "pct_people_of_color": _safe_float(row.get("pct_people_of_color")),
        "pct_low_income":      _safe_float(row.get("pct_low_income")),
        "traffic_proximity":   _safe_float(row.get("traffic_proximity")),
        "superfund_proximity": _safe_float(row.get("superfund_proximity")),
        "diesel_pm_proximity": _safe_float(row.get("diesel_pm_proximity")),
        "pct_ling_isolated":   _safe_float(row.get("pct_ling_isolated")),
        "county":              str(row.get("CNTY_NAME", "Unknown")),
    }


@app.get("/api/texas/tracts/geojson")
async def texas_tracts_geojson():
    """Serve all Texas census tract GeoJSON. MUST be before /{city}/ route."""
    if not os.path.exists(TEXAS_GEOJSON_PATH):
        raise HTTPException(503, "Statewide GeoJSON not found. Run pipeline/01_build_tract_lookup.py first.")
    return FileResponse(TEXAS_GEOJSON_PATH, media_type="application/geo+json")


@app.get("/api/tracts/geojson")
async def tracts_geojson_legacy():
    """Legacy endpoint - returns Dallas GeoJSON for backward compatibility."""
    geojson_path = CITIES["dallas"]["geojson"]
    if not os.path.exists(geojson_path):
        raise HTTPException(503, "GeoJSON not found. Run pipeline/01_build_tract_lookup.py first.")
    return FileResponse(geojson_path, media_type="application/geo+json")


@app.get("/api/{city}/tracts/geojson")
async def city_tracts_geojson(city: str):
    """Serve census tract GeoJSON for the specified city."""
    if city not in CITIES:
        raise HTTPException(404, f"Unknown city: {city}")

    geojson_path = CITIES[city]["geojson"]
    if not os.path.exists(geojson_path):
        raise HTTPException(503, f"GeoJSON not found for {city}. Run pipeline/01_build_tract_lookup.py first.")
    return FileResponse(geojson_path, media_type="application/geo+json")


# ── Quantum Sensor Placement ────────────────────────────────────────────────
_quantum_cache: dict = {"data": None, "expires": datetime.min.replace(tzinfo=timezone.utc)}
QUANTUM_CACHE_TTL_MIN = 60

# Path to PurpleAir sensor training data (contains real sensor locations)
SENSOR_DATA_PATH = os.path.join(ROOT, "p2_processed.xls")
LOSO_RESIDUALS_PATH = os.path.join(ROOT, "models", "loso_residuals.json")


def _load_existing_sensors():
    """Load 240 real PurpleAir sensor locations from training data."""
    if not os.path.exists(SENSOR_DATA_PATH):
        print("WARNING: p2_processed.xls not found — quantum solver won't have existing sensor locations")
        return []
    try:
        try:
            df = pd.read_csv(SENSOR_DATA_PATH, encoding="utf-8-sig", low_memory=False)
        except Exception:
            try:
                df = pd.read_excel(SENSOR_DATA_PATH, engine="xlrd")
            except Exception:
                df = pd.read_excel(SENSOR_DATA_PATH, engine="openpyxl")

        # De-duplicate to unique sensor locations
        sensors_df = df.drop_duplicates(subset=["sensor_id"])[["sensor_id", "latitude", "longitude", "city"]].copy()
        sensors_df = sensors_df.dropna(subset=["latitude", "longitude"])
        sensors = [
            {"lat": float(row["latitude"]), "lon": float(row["longitude"]),
             "sensor_id": str(row["sensor_id"]), "city": str(row.get("city", ""))}
            for _, row in sensors_df.iterrows()
        ]
        print(f"Loaded {len(sensors)} existing PurpleAir sensor locations for quantum solver")
        return sensors
    except Exception as e:
        print(f"Failed to load sensor data: {e}")
        return []


def _compute_model_disagreement(df: pd.DataFrame, weather: dict, temporal: dict) -> np.ndarray:
    """
    Compute per-tract prediction uncertainty across weather perturbations.
    Since the ensemble was retrained on full data, individual models may agree
    closely. Instead, we measure sensitivity: how much does the prediction
    change when weather varies? Tracts that are highly sensitive to weather
    inputs are harder to predict accurately without a local sensor.
    """
    bundle = state.get("bundle")
    if bundle is None:
        raise ValueError("Model bundle not loaded")
    features = bundle["feature_names"]
    weights = bundle["weights"]
    models_dict = bundle["models"]

    n = len(df)

    def _predict_with_weather(w):
        shared = {**w, **temporal}
        X = np.zeros((n, len(features)), dtype=np.float64)
        for i, feat in enumerate(features):
            if feat in shared:
                X[:, i] = shared[feat]
            elif feat in df.columns:
                X[:, i] = pd.to_numeric(df[feat], errors="coerce").fillna(0.0).values
        raw = sum(weights[name] * models_dict[name].predict(X) for name in _active_models(weights, models_dict))
        if bundle.get("target_transform") == "log1p":
            raw = np.expm1(raw)
        return np.maximum(0.0, raw)

    # Predict under several weather perturbations
    base_pred = _predict_with_weather(weather)
    perturbations = [
        {**weather, "temperature": weather["temperature"] + 15},
        {**weather, "temperature": weather["temperature"] - 15},
        {**weather, "humidity": min(100, weather["humidity"] + 25)},
        {**weather, "humidity": max(0, weather["humidity"] - 25)},
        {**weather, "wind_speed": weather["wind_speed"] + 10},
        {**weather, "pressure": weather["pressure"] + 15},
    ]
    all_preds = [base_pred] + [_predict_with_weather(w) for w in perturbations]
    pred_stack = np.array(all_preds)
    disagreement = np.std(pred_stack, axis=0)

    return disagreement


def _load_loso_residuals():
    """Load per-GEOID LOSO-CV residuals for quantum solver."""
    if not os.path.exists(LOSO_RESIDUALS_PATH):
        return None
    try:
        with open(LOSO_RESIDUALS_PATH) as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load LOSO residuals: {e}")
        return None


async def _run_quantum_placement():
    """Run quantum sensor placement with real data + LOSO residuals."""
    from backend.quantum.qubo_solver import solve_quantum, compute_coverage

    texas_data = await texas_predictions()
    tracts = texas_data.get("tracts", [])

    if not tracts:
        raise HTTPException(503, "No tract predictions available. Wait for predictions to load.")

    # Load real PurpleAir sensor locations
    existing_sensors = await asyncio.to_thread(_load_existing_sensors)

    # Load LOSO-CV residuals (true spatial prediction errors)
    loso_residuals = await asyncio.to_thread(_load_loso_residuals)

    # Build per-tract model disagreement from LOSO residuals
    model_disagreement = None
    if loso_residuals:
        lookup = state.get("tract_lookup")
        if lookup is not None:
            geoids = lookup["GEOID"].astype(str).str.zfill(11).values
            model_disagreement = np.array([
                loso_residuals.get(g, 0.0) for g in geoids
            ], dtype=np.float64)
            print(f"LOSO residuals loaded: mean={model_disagreement.mean():.3f}, "
                  f"max={model_disagreement.max():.3f}")

    # Fallback to weather sensitivity if no LOSO residuals
    if model_disagreement is None:
        if state.get("bundle") and state.get("tract_lookup") is not None:
            try:
                # Model-unit fallback (the payload's `weather` is the DISPLAY
                # dict in °F/mph — never feed that to the model).
                weather = dict(WEATHER_FALLBACK)
                temporal = get_temporal("America/Chicago")
                model_disagreement = await asyncio.to_thread(
                    _compute_model_disagreement, state["tract_lookup"],
                    weather, temporal)
            except Exception as e:
                print(f"Disagreement computation failed: {e}")

    # Run quantum solver only
    quantum_result = await asyncio.to_thread(
        solve_quantum,
        tracts,
        k=25,
        num_reads=500,
        top_candidates=120,
        proximity_threshold_miles=8.0,
        existing_sensors=existing_sensors,
        model_disagreement=model_disagreement,
    )

    # Compute coverage
    coverage = compute_coverage(
        tracts, quantum_result["selected_tracts"],
        radius_miles=10.0,
        existing_sensors=existing_sensors,
    )

    avg_ej = float(np.mean([
        t.get("ejf_score", 0.0) or 0.0
        for t in quantum_result["selected_tracts"]
    ])) if quantum_result["selected_tracts"] else 0.0

    avg_composite = float(np.mean([
        t["composite_score"]
        for t in quantum_result["selected_tracts"]
    ])) if quantum_result["selected_tracts"] else 0.0

    now = datetime.now(timezone.utc)
    result = {
        "generated_at": now.isoformat(),
        "expires_at": (now + timedelta(minutes=QUANTUM_CACHE_TTL_MIN)).isoformat(),
        "num_total_tracts": len(tracts),
        "num_sensors": 25,
        "num_existing_sensors": len(existing_sensors),
        "proximity_threshold_miles": 8.0,
        "coverage_radius_miles": 10.0,
        "existing_sensors": existing_sensors,
        "methods": {
            "quantum_annealing": {
                "method_display": quantum_result["method_display"],
                "num_sensors": quantum_result["num_sensors"],
                "coverage": coverage,
                "avg_ej_score": round(avg_ej, 1),
                "avg_composite_score": round(avg_composite, 4),
                "timing": quantum_result["timing"],
                "selected_tracts": quantum_result["selected_tracts"],
                "best_energy": quantum_result.get("best_energy"),
                "num_reads": quantum_result.get("num_reads"),
                "num_candidates": quantum_result.get("num_candidates"),
            },
        },
    }

    return result


async def _revalidate_quantum_background():
    """Recompute quantum placement off the request path; updates cache + disk snapshot."""
    global _quantum_cache
    if _revalidating.get("quantum"):
        return
    _revalidating["quantum"] = True
    try:
        result = await _run_quantum_placement()
        _quantum_cache = {
            "data": result,
            "expires": datetime.now(timezone.utc) + timedelta(minutes=QUANTUM_CACHE_TTL_MIN),
        }
        _save_snapshot(QUANTUM_SNAPSHOT_PATH, result)
        print("✓ Quantum placement revalidated")
    except Exception as e:
        print(f"Quantum revalidation failed: {e}")
    finally:
        _revalidating["quantum"] = False


@app.get("/api/quantum/sensor-placement")
async def quantum_sensor_placement():
    """
    Returns quantum-optimized sensor placement recommendations.
    Stale-while-revalidate: cached/snapshot data is served instantly; a background
    refresh kicks off when older than 60 min. The 94s annealing run never blocks
    a user-facing request once the snapshot exists.
    """
    global _quantum_cache
    now = datetime.now(timezone.utc)
    cached_data = _quantum_cache.get("data")
    expires_at = _quantum_cache.get("expires", datetime.min.replace(tzinfo=timezone.utc))

    if cached_data is not None:
        if now >= expires_at:
            asyncio.create_task(_revalidate_quantum_background())
        return cached_data

    # No cache, no snapshot — must compute synchronously (first request ever).
    try:
        result = await _run_quantum_placement()
        _quantum_cache = {
            "data": result,
            "expires": now + timedelta(minutes=QUANTUM_CACHE_TTL_MIN),
        }
        _save_snapshot(QUANTUM_SNAPSHOT_PATH, result)
        return result
    except HTTPException:
        raise
    except Exception as e:
        print(f"Quantum placement error: {e}")
        raise HTTPException(500, f"Quantum solver failed: {str(e)}")


# ── NOAA HMS smoke polygon live layer ─────────────────────────────────────────

async def _revalidate_hms_background():
    """Fetch the most recent HMS smoke polygons from NOAA, clip to TX bbox,
    cache + snapshot. Runs in background so a request never waits on the
    NOAA round-trip."""
    global _hms_cache
    if _revalidating.get("hms"):
        return
    _revalidating["hms"] = True
    try:
        from backend.hms import fetch_latest_hms

        result = await fetch_latest_hms()
        _hms_cache = {
            "data": result,
            "expires": datetime.now(timezone.utc) + timedelta(minutes=HMS_CACHE_TTL_MIN),
        }
        _save_snapshot(HMS_SNAPSHOT_PATH, result)
        print(
            f"✓ HMS revalidated: {result.get('count', 0)} polygons "
            f"(data_date={result.get('data_date')})"
        )
    except Exception as e:
        print(f"HMS revalidation failed: {e}")
    finally:
        _revalidating["hms"] = False


@app.get("/api/live/hms-smoke")
async def hms_smoke():
    """Returns the most recent NOAA HMS smoke polygons clipped to Texas, as a
    GeoJSON FeatureCollection. Stale-while-revalidate: cached/snapshot data is
    served instantly; a background refresh kicks off when older than 60 min."""
    global _hms_cache
    now = datetime.now(timezone.utc)
    cached_data = _hms_cache.get("data")
    expires_at = _hms_cache.get("expires", datetime.min.replace(tzinfo=timezone.utc))

    if cached_data is not None:
        if now >= expires_at:
            asyncio.create_task(_revalidate_hms_background())
        return cached_data

    # No cache, no snapshot — fetch synchronously (first request ever).
    try:
        from backend.hms import fetch_latest_hms

        result = await fetch_latest_hms()
        _hms_cache = {
            "data": result,
            "expires": now + timedelta(minutes=HMS_CACHE_TTL_MIN),
        }
        _save_snapshot(HMS_SNAPSHOT_PATH, result)
        return result
    except Exception as e:
        print(f"HMS endpoint error: {e}")
        # Don't 500 the frontend just because NOAA is down — return empty FC.
        return {
            "type": "FeatureCollection",
            "features": [],
            "fetched_at": now.isoformat(),
            "data_date": None,
            "count": 0,
            "density_counts": {},
            "error": str(e),
        }


# ── Live PurpleAir layer (feeds the model's nbr_pm25_50km feature) ─────────────

async def _revalidate_purpleair_background():
    """Fetch current TX PurpleAir readings, cache + snapshot. Background so no
    request waits on the PurpleAir round-trip."""
    global _purpleair_cache
    if _revalidating.get("purpleair"):
        return
    _revalidating["purpleair"] = True
    try:
        from backend.purpleair import fetch_live_snapshot

        snap = await fetch_live_snapshot()
        # Only overwrite the cache with a usable snapshot. A transient empty
        # fetch (API hiccup) must not wipe a good prior snapshot.
        if snap.get("usable"):
            _purpleair_cache = {
                "data": snap,
                "expires": datetime.now(timezone.utc) + timedelta(minutes=PURPLEAIR_CACHE_TTL_MIN),
            }
            _save_snapshot(PURPLEAIR_SNAPSHOT_PATH, snap)
            print(f"✓ PurpleAir revalidated: {snap['count']} live sensors, "
                  f"statewide mean={snap.get('statewide_mean')}")
        else:
            print(f"PurpleAir fetch returned {snap.get('count', 0)} usable sensors "
                  f"(<{20}); keeping prior cache")
    except Exception as e:
        print(f"PurpleAir revalidation failed: {e}")
    finally:
        _revalidating["purpleair"] = False


def get_live_purpleair_sensors() -> list:
    """Return the cached live sensor list (stale-while-revalidate). Kicks a
    background refresh when the cache is expired. Returns [] if no usable data
    (no key / API down) so the caller falls back to the static climatology."""
    now = datetime.now(timezone.utc)
    cached = _purpleair_cache.get("data")
    expires_at = _purpleair_cache.get("expires", datetime.min.replace(tzinfo=timezone.utc))
    if cached is not None:
        if now >= expires_at:
            try:
                asyncio.create_task(_revalidate_purpleair_background())
            except RuntimeError:
                pass  # no running loop (called outside async context)
        return cached.get("sensors", []) if cached.get("usable") else []
    return []

@app.get("/api/geocode")
async def proxy_geocode(q: str, accept_language: str = Header(default="en")):
    """Proxy Nominatim geocoding. We proxy server-side so we can send the
    descriptive User-Agent Nominatim's usage policy requires (browsers forbid
    setting User-Agent from fetch). The caller's Accept-Language (es/en) is
    forwarded so suggestions are localized to the UI language; limit=5 matches
    the autocomplete dropdown the frontend renders."""
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": q, "format": "json", "countrycodes": "us", "limit": 5},
                headers={
                    "User-Agent": "SharedSkiesInitiative/1.0 (contact@example.com)",
                    "Accept-Language": accept_language,
                },
                timeout=10.0
            )
            r.raise_for_status()
            return r.json()
        except Exception:
            raise HTTPException(status_code=502, detail="Failed to reach geocoding service")

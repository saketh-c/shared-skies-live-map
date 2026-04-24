"""
Shared Skies — FastAPI Backend
Serves PM2.5 predictions for multiple Texas counties (Dallas, Austin, Houston, San Antonio).

Start with:
    uvicorn backend.main:app --reload --port 8000
(run from the project root)
"""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import httpx
import joblib
import numpy as np
import pandas as pd
import sqlite3
from fastapi import FastAPI, HTTPException
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
                await asyncio.sleep(10)
                try:
                    print("Background: computing full Texas predictions (deferred)...")
                    await texas_predictions()
                    print("Background: Texas precompute complete.")
                except Exception as e:
                    print(f"Background Texas precompute failed: {e}")

            asyncio.create_task(_deferred_full_texas())
        except Exception as e:
            print(f"Background Texas precompute failed: {e}")

    # Schedule background precompute (do not await)
    try:
        asyncio.create_task(_precompute_texas())
    except Exception:
        # In some environments create_task must be called differently; ignore failures
        pass

    # Keepalive loop: periodically ping an external URL (KEEPALIVE_URL)
    # or call the internal health() endpoint to keep the process warm.
    async def _keepalive_loop():
        url = os.environ.get("KEEPALIVE_URL")
        try:
            interval = int(os.environ.get("KEEPALIVE_INTERVAL", "840"))  # seconds (14 minutes)
        except Exception:
            interval = 840

        async with httpx.AsyncClient(timeout=10.0) as client:
            while True:
                try:
                    if url:
                        try:
                            resp = await client.get(url)
                            print(f"Keepalive: pinged {url} status={resp.status_code}")
                        except Exception as e:
                            print(f"Keepalive HTTP error pinging {url}: {e}")
                    else:
                        # Fallback: call internal health() so FastAPI handlers run briefly
                        try:
                            await health()
                            print("Keepalive: called internal health()")
                        except Exception as e:
                            print(f"Keepalive internal health() error: {e}")

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── Utilities ─────────────────────────────────────────────────────────────────

async def fetch_weather(lat: float, lon: float) -> dict:
    """Fetch current weather from Open-Meteo (free, no API key)."""
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m"
        "&temperature_unit=fahrenheit"
        "&wind_speed_unit=mph"
        "&timezone=America%2FChicago"
    )
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        cur = resp.json()["current"]
    return {
        "temperature": round(cur["temperature_2m"], 1),
        "humidity":    round(cur["relative_humidity_2m"], 1),
        "pressure":    round(cur["pressure_msl"], 1),
        "wind_speed":  round(cur["wind_speed_10m"], 1),
    }


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
    pred = sum(weights[n] * models[n].predict(X)[0] for n in models)
    return max(0.0, float(pred))


def _compute_v3_shared(weather: dict, temporal: dict) -> dict:
    """Compute all derived features that are shared across tracts (weather + temporal)."""
    month = temporal.get("month", 1)
    dow = temporal.get("dow", 0)
    doy = temporal.get("day_of_year", 1)
    temp = weather.get("temperature", 72)
    hum = weather.get("humidity", 55)
    ws = weather.get("wind_speed", 8)
    precip = weather.get("precipitation", 0)

    shared = {**weather, **temporal}
    shared["precipitation"] = precip
    # Cyclical
    shared["month_sin"] = np.sin(2 * np.pi * month / 12)
    shared["month_cos"] = np.cos(2 * np.pi * month / 12)
    shared["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    shared["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    shared["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    shared["doy_cos"] = np.cos(2 * np.pi * doy / 365)
    # Interactions
    shared["temp_x_humidity"] = temp * hum / 100.0
    shared["wind_x_temp"] = ws * temp / 100.0
    shared["wind_x_season"] = ws * shared["month_sin"]
    shared["humidity_x_season"] = hum * shared["doy_cos"] / 100.0
    shared["precip_x_temp"] = precip * temp / 100.0
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

    # Add spatial features to df if v3 model needs them
    if any(f in features for f in ["elevation", "dist_to_coast", "urban_index"]):
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

    for i, feat in enumerate(features):
        if feat in shared:
            X[:, i] = shared[feat]
        elif feat in df.columns:
            X[:, i] = pd.to_numeric(df[feat], errors="coerce").fillna(0.0).values

    preds = sum(weights[name] * models[name].predict(X) for name in models)
    return np.maximum(0.0, preds)


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb):
    """Convert RGB tuple to hex color."""
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def interpolate_color(color1, color2, factor):
    """Interpolate between two colors based on factor (0-1)."""
    factor = max(0, min(1, factor))
    rgb1 = hex_to_rgb(color1)
    rgb2 = hex_to_rgb(color2)

    rgb = tuple(rgb1[i] + (rgb2[i] - rgb1[i]) * factor for i in range(3))
    return rgb_to_hex(rgb)


def pm25_color_gradient(pm25: float) -> str:
    """Get gradient color based on PM2.5 value."""
    if pm25 < 0:
        pm25 = 0

    # Green range: 0.0-8.9
    if pm25 <= 8.9:
        factor = pm25 / 8.9
        return interpolate_color("#90EE90", "#00b894", factor)

    # Yellow range: 9.0-12.9
    elif pm25 <= 12.9:
        factor = (pm25 - 9.0) / 3.9
        return interpolate_color("#FFFF99", "#FFD700", factor)

    # Red range: 13.0-17.9
    elif pm25 <= 17.9:
        factor = (pm25 - 13.0) / 4.9
        return interpolate_color("#FF6B6B", "#d63031", factor)

    # Dark red range: 18.0+ (gets darker as pollution rises, fully saturates at ~30)
    else:
        factor = min(1.0, (pm25 - 18.0) / 12.0)
        return interpolate_color("#8b0000", "#1a0000", factor)


def pm25_info(pm25: float) -> dict:
    """Get AQI info with custom gradient scale."""
    color = pm25_color_gradient(pm25)

    if pm25 <= 8.9:
        return {
            "category": "Good",
            "color": color,
            "aqi_range": "0–8.9",
            "health_msg": "Air quality is good. Enjoy outdoor activities.",
        }
    elif pm25 <= 12.9:
        return {
            "category": "Moderate",
            "color": color,
            "aqi_range": "9–12.9",
            "health_msg": "Air quality is acceptable. Sensitive individuals should take precautions.",
        }
    elif pm25 <= 17.9:
        return {
            "category": "Unhealthy",
            "color": color,
            "aqi_range": "13–17.9",
            "health_msg": "Air quality is unhealthy. Everyone should limit outdoor exposure.",
        }
    else:
        return {
            "category": "Hazardous",
            "color": color,
            "aqi_range": "18+",
            "health_msg": "⚠️ Air quality is hazardous. Avoid all outdoor activities.",
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

    lookup = state["tract_lookup"]
    city_tracts = lookup[lookup["GEOID"].str.startswith(city_config["fips"])].copy()

    if city_tracts.empty:
        raise HTTPException(404, f"No tracts found for city: {city}")

    # Fetch weather for the city center
    try:
        weather = await fetch_weather(*city_config["center"])
    except Exception as e:
        print(f"Weather API error for {city}: {e}. Using fallback values.")
        weather = {"temperature": 72.0, "humidity": 55.0, "pressure": 1013.0, "wind_speed": 8.0}

    temporal = get_temporal(city_config["tz"])
    tracts = []

    # Vectorized batch prediction — single model.predict() call for all tracts
    city_tracts_reset = city_tracts.reset_index(drop=True)
    pm25_array = await asyncio.to_thread(run_predictions_batch, city_tracts_reset, weather, temporal)

    for i, (_, row) in enumerate(city_tracts.iterrows()):
        pm25 = round(float(pm25_array[i]), 2)
        info = pm25_info(pm25)
        tracts.append({
            "geoid":               row["GEOID"],
            "lat":                 round(float(row["lat"]), 6),
            "lon":                 round(float(row["lon"]), 6),
            "pm25":                pm25,
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
            "county":              str(row.get("CNTY_NAME", city)),
        })

    avg_pm25 = round(float(np.mean([t["pm25"] for t in tracts])), 2)

    result = {
        "city":         city,
        "display_name": city_config["display_name"],
        "generated_at": now.isoformat(),
        "expires_at":   (now + timedelta(minutes=30)).isoformat(),
        "weather":      weather,
        "avg_pm25":     avg_pm25,
        "avg_info":     pm25_info(avg_pm25),
        "tract_count":  len(tracts),
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
    return {
        "status": "ok",
        "model_loaded": state.get("bundle") is not None,
        "lookup_loaded": state.get("tract_lookup") is not None,
        "available_cities": list(CITIES.keys()),
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


@app.get("/api/texas/predictions")
async def texas_predictions():
    """
    Returns PM2.5 predictions for all Texas census tracts.
    Results are cached for 30 minutes.
    MUST be before /api/{city}/predictions to match correctly.
    """
    cache_key = "cache_texas"
    cache = state.get(cache_key, {})
    now = datetime.now(timezone.utc)

    # Check cache
    if cache.get("data") and now < cache.get("expires", datetime.min.replace(tzinfo=timezone.utc)):
        return cache["data"]

    if state.get("bundle") is None:
        raise HTTPException(503, "Model not loaded. Run pipeline/02_train_model.py first.")
    if state.get("tract_lookup") is None:
        raise HTTPException(503, "Tract lookup not loaded. Run pipeline/01_build_tract_lookup.py first.")

    lookup = state["tract_lookup"]
    tracts = []

    print(f"Generating predictions for all {len(lookup)} Texas tracts (vectorized)...")
    # Use Austin as default weather location (central Texas)
    try:
        weather = await fetch_weather(30.2672, -97.7431)
    except Exception as e:
        print(f"Weather API error: {e}. Using fallback values.")
        weather = {"temperature": 72.0, "humidity": 55.0, "pressure": 1013.0, "wind_speed": 8.0}

    temporal = get_temporal("America/Chicago")

    # Vectorized batch prediction — single model.predict() call for all 6,900+ tracts
    lookup_reset = lookup.reset_index(drop=True)
    pm25_array = await asyncio.to_thread(run_predictions_batch, lookup_reset, weather, temporal)

    for i, (_, row) in enumerate(lookup.iterrows()):
        pm25 = round(float(pm25_array[i]), 2)
        info = pm25_info(pm25)
        tracts.append({
            "geoid":               row["GEOID"],
            "lat":                 round(float(row["lat"]), 6),
            "lon":                 round(float(row["lon"]), 6),
            "pm25":                pm25,
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

    result = {
        "region":       "texas",
        "display_name": "All of Texas",
        "generated_at": now.isoformat(),
        "expires_at":   (now + timedelta(minutes=30)).isoformat(),
        "weather":      weather,
        "avg_pm25":     avg_pm25,
        "avg_info":     pm25_info(avg_pm25),
        "tract_count":  len(tracts),
        "tracts":       tracts,
    }

    state[cache_key] = {"data": result, "expires": now + timedelta(minutes=30)}
    print(f"✓ Texas predictions complete: {len(tracts)} tracts")
    return result


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

    try:
        weather = await fetch_weather(float(row["lat"]), float(row["lon"]))
    except Exception:
        weather = {"temperature": 72.0, "humidity": 55.0, "pressure": 1013.0, "wind_speed": 8.0}

    temporal = get_temporal(tz)
    pm25     = run_prediction(row, weather, temporal)
    info     = pm25_info(pm25)

    return {
        "geoid":               geoid,
        "lat":                 round(float(row["lat"]), 6),
        "lon":                 round(float(row["lon"]), 6),
        "pm25":                round(pm25, 2),
        "category":            info["category"],
        "color":               info["color"],
        "aqi_range":           info["aqi_range"],
        "health_msg":          info["health_msg"],
        "weather":             weather,
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
    bundle = state["bundle"]
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
        return np.maximum(0.0, sum(weights[name] * models_dict[name].predict(X)
                                   for name in models_dict))

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
                weather = texas_data.get("weather", {
                    "temperature": 72.0, "humidity": 55.0,
                    "pressure": 1013.0, "wind_speed": 8.0,
                })
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


@app.get("/api/quantum/sensor-placement")
async def quantum_sensor_placement():
    """
    Returns quantum-optimized sensor placement recommendations.
    Compares three methods: Quantum Annealing, Greedy, Classical SA.
    Uses real PurpleAir sensor locations + ensemble model disagreement.
    Results are cached for 1 hour.
    """
    global _quantum_cache
    now = datetime.now(timezone.utc)

    if (_quantum_cache.get("data") and
        now < _quantum_cache.get("expires", datetime.min.replace(tzinfo=timezone.utc))):
        return _quantum_cache["data"]

    try:
        result = await _run_quantum_placement()
        _quantum_cache = {
            "data": result,
            "expires": now + timedelta(minutes=QUANTUM_CACHE_TTL_MIN),
        }
        return result
    except HTTPException:
        raise
    except Exception as e:
        print(f"Quantum placement error: {e}")
        raise HTTPException(500, f"Quantum solver failed: {str(e)}")

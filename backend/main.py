"""
Shared Skies — FastAPI Backend
Serves PM2.5 predictions for multiple Texas counties (Dallas, Austin, Houston, San Antonio).

Start with:
    uvicorn backend.main:app --reload --port 8000
(run from the project root)
"""

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import httpx
import joblib
import numpy as np
import pandas as pd
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
    # Load model
    if os.path.exists(MODEL_PATH):
        state["bundle"] = joblib.load(MODEL_PATH)
        print(f"Model loaded. Features: {state['bundle']['feature_names']}")
    else:
        state["bundle"] = None
        print("WARNING: models/ensemble.joblib not found. Run pipeline/02_train_model.py")

    # Load tract lookup
    if os.path.exists(LOOKUP_PATH):
        df = pd.read_parquet(LOOKUP_PATH)
        df["GEOID"] = df["GEOID"].astype(str).str.zfill(11)
        state["tract_lookup"] = df
        print(f"Tract lookup loaded: {len(df)} total tracts")
        for city, config in CITIES.items():
            count = df["GEOID"].str.startswith(config["fips"]).sum()
            print(f"  {city}: {count} tracts")
    else:
        state["tract_lookup"] = None
        print("WARNING: backend/static/tract_lookup.parquet not found. Run pipeline/01_build_tract_lookup.py")

    # Per-city caches
    for city in CITIES:
        state[f"cache_{city}"] = {"data": None, "expires": datetime.min.replace(tzinfo=timezone.utc)}

    # Statewide cache
    state["cache_texas"] = {"data": None, "expires": datetime.min.replace(tzinfo=timezone.utc)}

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

    row = {**weather, **temporal}
    for feat in features:
        if feat not in row:
            row[feat] = float(tract_row.get(feat, 0.0) or 0.0)

    X = np.array([[row.get(f, 0.0) for f in features]])
    pred = sum(weights[n] * models[n].predict(X)[0] for n in models)
    return max(0.0, float(pred))


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

    # Green range: 0.0-3.9
    if pm25 <= 3.9:
        factor = pm25 / 3.9
        return interpolate_color("#90EE90", "#00b894", factor)

    # Yellow range: 4.0-8.9
    elif pm25 <= 8.9:
        factor = (pm25 - 4.0) / 4.9
        return interpolate_color("#FFFF99", "#FFD700", factor)

    # Red range: 9.0-12.9
    elif pm25 <= 12.9:
        factor = (pm25 - 9.0) / 3.9
        return interpolate_color("#FF6B6B", "#d63031", factor)

    # Purple range: 13.0+
    else:
        # Gradient from bright purple to dark purple (13.0-25+)
        factor = min(1.0, (pm25 - 13.0) / 12.0)
        return interpolate_color("#9d4edd", "#3c096c", factor)


def pm25_info(pm25: float) -> dict:
    """Get AQI info with custom gradient scale."""
    color = pm25_color_gradient(pm25)

    if pm25 <= 3.9:
        return {
            "category": "Good",
            "color": color,
            "aqi_range": "0–3.9",
            "health_msg": "Air quality is good. Enjoy outdoor activities.",
        }
    elif pm25 <= 8.9:
        return {
            "category": "Moderate",
            "color": color,
            "aqi_range": "4–8.9",
            "health_msg": "Air quality is acceptable. Sensitive individuals should take precautions.",
        }
    elif pm25 <= 12.9:
        return {
            "category": "Unhealthy",
            "color": color,
            "aqi_range": "9–12.9",
            "health_msg": "Air quality is unhealthy. Everyone should limit outdoor exposure.",
        }
    else:
        return {
            "category": "Hazardous",
            "color": color,
            "aqi_range": "13+",
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

    for _, row in city_tracts.iterrows():
        pm25 = run_prediction(row, weather, temporal)
        info = pm25_info(pm25)
        tracts.append({
            "geoid":               row["GEOID"],
            "lat":                 round(float(row["lat"]), 6),
            "lon":                 round(float(row["lon"]), 6),
            "pm25":                round(pm25, 2),
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

    print("Generating predictions for all Texas tracts...")
    # Use Austin as default weather location (central Texas)
    try:
        weather = await fetch_weather(30.2672, -97.7431)
    except Exception as e:
        print(f"Weather API error: {e}. Using fallback values.")
        weather = {"temperature": 72.0, "humidity": 55.0, "pressure": 1013.0, "wind_speed": 8.0}

    temporal = get_temporal("America/Chicago")

    for idx, (_, row) in enumerate(lookup.iterrows()):
        if idx % 1000 == 0:
            print(f"  Processing tract {idx}/{len(lookup)}...")

        pm25 = run_prediction(row, weather, temporal)
        info = pm25_info(pm25)
        tracts.append({
            "geoid":               row["GEOID"],
            "lat":                 round(float(row["lat"]), 6),
            "lon":                 round(float(row["lon"]), 6),
            "pm25":                round(pm25, 2),
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

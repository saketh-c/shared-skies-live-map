"""National Weather Service (api.weather.gov) CURRENT conditions for DISPLAY.

Replaces the Open-Meteo current-conditions grid, whose calls fail from
Render's shared egress IP (the display conditions went ~18 h stale). This is
UI weather only: model features are untouched (train/serve consistency).

api.weather.gov is free and keyless but asks every client to send a
User-Agent with contact details, and it serves station observations, not a
gridded analysis. So:

  * fetch_stations(): the observing-station catalog for TX and the bordering
    states, kept to reliable surface METAR stations (rule below), cached 24 h
    in memory plus a small JSON snapshot so a restart skips the ~6.5 MB
    catalog download.
  * fetch_conditions_grid(lats, lons): the same shape as
    airquality.fetch_conditions_grid (1.0 deg cells keyed by
    airquality.conditions_cell_key), each cell filled from the nearest station
    with a fresh observation, per field from the next-nearest (up to 3).
  * fetch_point_current(lat, lon): the same for one clicked location.

STATION RULE (measured 2026-11-02 over the 5,367 catalog entries in the
bbox): keep an id matching ^K[A-Z0-9]{3}$ (a CONUS ICAO airport id) whose
provider is one of ASOS, ASOS-HFM, OTHER-MTR or NonFedAWOS (FAA/NWS ASOS
and state/municipal AWOS, all reporting METARs): 307 stations, 304 with an
observation <= 3 h old. Dropped: the 9 K-ids with no provider (8 returned no
observations at all), and every non-ICAO network (LCRA, RAWS, MesoWest,
COOP, ...), which carries no pressure and often no humidity or wind.

UNITS (display, same as the Open-Meteo path): temperature degF (1 dp),
humidity %, pressure hPa at sea level, wind mph. Sea-level pressure comes
from seaLevelPressure when the observation has it (routine hourly METARs);
otherwise from barometricPressure, which api.weather.gov fills with the
METAR altimeter setting (A2984 -> 101049.84 Pa). The altimeter setting reads
2-8 hPa above MSL pressure at West Texas elevations, so it is converted back
to station pressure with the observation's elevation and reduced to sea level
with its temperature (_altimeter_to_msl; 0.3 hPa MAE vs reported SLP).

POLITENESS: one /observations/latest call per cell's nearest station (~81
per hourly grid refresh, a few more when a nearest station is stale),
concurrency 6, one retry on 5xx. A refresh in which no station answers stops
after its first round. Observations are cached per station for 20 minutes
and shared by the grid and clicks, so clicks cost at most one call per
station per 20 minutes. The grid is bounded to GRID_TIMEOUT_S and a click to
POINT_TIMEOUT_S. Nothing here raises: failures return an unusable grid /
None and the caller falls back.
"""
import asyncio
import json
import math
import os
import re
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone

import httpx
import numpy as np

from backend.airquality import WEATHER_GRID_DEG, build_cells, conditions_cell_key

API_BASE = "https://api.weather.gov"
USER_AGENT = "SharedSkiesInitiative/1.0 (+https://sharedskiesinitiative.org; chebrolusaketh@gmail.com)"
HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/geo+json"}

STATES = ("TX", "NM", "OK", "AR", "LA")
BBOX = {"lat_min": 25.5, "lat_max": 36.8, "lon_min": -107.0, "lon_max": -93.2}
METAR_PROVIDERS = frozenset({"ASOS", "ASOS-HFM", "OTHER-MTR", "NonFedAWOS"})
_ICAO_RE = re.compile(r"^K[A-Z0-9]{3}$")

STATIONS_TTL_H = 24
STATIONS_RETRY_MIN = 30          # backoff after a failed or partial catalog fetch
STATIONS_PAGE_LIMIT = 500
STATIONS_MAX_PAGES = 30          # per state; TX is ~9 pages of 500
MAX_OBS_AGE_H = 3.0
MAX_STATIONS_PER_POINT = 3
MAX_STATION_KM = 150.0           # never describe a place with a station farther than this
BORROW_K = 12                    # fields still missing after 3 stations may come from
BORROW_MAX_KM = 100.0            # an ALREADY-fetched observation this near (no extra calls)
CONCURRENCY = 6
GRID_TIMEOUT_S = 150.0           # whole grid refresh (normally ~2 s; catalog ~1 s more)
POINT_TIMEOUT_S = 8.0            # one click (normally < 1 s)
OBS_TTL_MIN = 20
OBS_FAIL_TTL_MIN = 5             # negative cache: a failing station is not retried per click
OBS_CACHE_MAX = 600              # above the ~307 catalog stations, so it never thrashes
RETRY_DELAY_S = 1.0
OBS_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
CATALOG_TIMEOUT = httpx.Timeout(30.0, connect=5.0)

# MADIS quality-control codes: X = failed (rejected), Q = questioned,
# B = subjectively bad. Z (not yet QC'd), C, S, V, G pass.
_QC_REJECT = frozenset({"X", "Q", "B"})

# Plausible display ranges; anything outside is a sensor/decoder fault.
_BOUNDS = {
    "temperature": (-40.0, 130.0),   # degF
    "humidity":    (0.0, 100.0),     # %
    "pressure":    (850.0, 1090.0),  # hPa
    "wind_speed":  (0.0, 150.0),     # mph
}
FIELDS = ("temperature", "humidity", "pressure", "wind_speed")

_snap_dir = os.environ.get("SSI_SNAPSHOTS_DIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "static", "snapshots")
# Set to None to disable the on-disk catalog copy (tests do).
STATIONS_SNAPSHOT_PATH = os.path.join(_snap_dir, "nws_stations.json")

_stations = {"list": None, "lat": None, "lon": None, "fetched_at": None, "retry_after": 0.0}
_obs_cache: OrderedDict = OrderedDict()   # station id -> {"obs": dict | None, "expires": monotonic}
_locks: dict = {}                          # (event loop id, name) -> asyncio.Lock
# Point-lookup circuit breaker: a timed-out lookup is cancelled before it can
# negative-cache its stations, so without this every click during an NWS
# brownout would wait the full POINT_TIMEOUT_S.
_point_backoff = {"until": 0.0}


def _reset_state() -> None:
    """Forget every cached catalog and observation (tests)."""
    _stations.update({"list": None, "lat": None, "lon": None, "fetched_at": None, "retry_after": 0.0})
    _obs_cache.clear()
    _locks.clear()
    _point_backoff["until"] = 0.0


def _lock(name: str) -> asyncio.Lock:
    """A per-event-loop lock, so the module works under repeated asyncio.run()."""
    key = (id(asyncio.get_running_loop()), name)
    lk = _locks.get(key)
    if lk is None:
        if len(_locks) > 16:
            _locks.clear()
        lk = _locks[key] = asyncio.Lock()
    return lk


# ── HTTP ──────────────────────────────────────────────────────────────────

async def _get_json(client: httpx.AsyncClient, url: str, params=None, timeout=OBS_TIMEOUT):
    """GET -> (status, json | None). One retry on 5xx or a transport error;
    never raises. status is 0 when no response arrived."""
    status = 0
    for attempt in range(2):
        try:
            r = await client.get(url, params=params, timeout=timeout)
            status = r.status_code
            if status == 200:
                return status, r.json()
            if status < 500:
                return status, None
        except (httpx.HTTPError, ValueError) as e:
            status = 0
            if attempt == 1:
                print(f"[nws] {url}: {type(e).__name__}: {e}")
        except Exception as e:  # never let a decoder surprise escape
            print(f"[nws] {url}: unexpected {type(e).__name__}: {e}")
            return 0, None
        if attempt == 0:
            await asyncio.sleep(RETRY_DELAY_S)
    return status, None


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(headers=HEADERS, timeout=OBS_TIMEOUT, follow_redirects=True)


# ── Station catalog ──────────────────────────────────────────────────────

def is_reliable_station(feature: dict) -> bool:
    """The station rule from the module docstring: ICAO K-id, METAR provider,
    inside the Texas+border bbox."""
    try:
        p = feature.get("properties") or {}
        sid = str(p.get("stationIdentifier") or "")
        lon, lat = (float(x) for x in feature["geometry"]["coordinates"][:2])
    except Exception:
        return False
    return (bool(_ICAO_RE.match(sid))
            and (p.get("provider") or "") in METAR_PROVIDERS
            and BBOX["lat_min"] <= lat <= BBOX["lat_max"]
            and BBOX["lon_min"] <= lon <= BBOX["lon_max"])


def _station_record(feature: dict) -> dict:
    p = feature["properties"]
    lon, lat = (float(x) for x in feature["geometry"]["coordinates"][:2])
    return {"id": p["stationIdentifier"], "name": p.get("name") or p["stationIdentifier"],
            "lat": round(lat, 5), "lon": round(lon, 5), "provider": p.get("provider") or ""}


async def _fetch_state(client: httpx.AsyncClient, state: str):
    """All reliable stations listed for one state, following the cursor
    pagination (the last page is empty but still carries a 'next' link).
    Returns None if any page fails."""
    out = []
    url, params = f"{API_BASE}/stations", {"state": state, "limit": STATIONS_PAGE_LIMIT}
    for _ in range(STATIONS_MAX_PAGES):
        status, body = await _get_json(client, url, params=params, timeout=CATALOG_TIMEOUT)
        if status != 200 or not isinstance(body, dict):
            print(f"[nws] station list {state}: HTTP {status}")
            return None
        feats = body.get("features") or []
        out.extend(_station_record(f) for f in feats if is_reliable_station(f))
        nxt = (body.get("pagination") or {}).get("next")
        if not feats or not nxt or not str(nxt).startswith(API_BASE + "/"):
            return out
        url, params = nxt, None
    return out


def _install_stations(records: list, fetched_at: str) -> None:
    uniq = {}
    for s in records:
        uniq.setdefault(s["id"], s)
    lst = sorted(uniq.values(), key=lambda s: s["id"])
    _stations["list"] = lst
    _stations["lat"] = np.array([s["lat"] for s in lst], dtype=float)
    _stations["lon"] = np.array([s["lon"] for s in lst], dtype=float)
    _stations["fetched_at"] = fetched_at


def _age_h(iso: str | None) -> float:
    try:
        t = datetime.fromisoformat(iso)
        return (datetime.now(timezone.utc) - t).total_seconds() / 3600.0
    except Exception:
        return float("inf")


def _load_stations_snapshot() -> None:
    path = STATIONS_SNAPSHOT_PATH
    if not path or _stations["list"] is not None or not os.path.exists(path):
        return
    try:
        with open(path) as f:
            snap = json.load(f)
        if snap.get("stations"):
            _install_stations(snap["stations"], snap.get("fetched_at"))
    except Exception as e:
        print(f"[nws] station snapshot unreadable: {e}")


def _save_stations_snapshot() -> None:
    path = STATIONS_SNAPSHOT_PATH
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"fetched_at": _stations["fetched_at"], "stations": _stations["list"]}, f)
        os.replace(tmp, path)
    except Exception as e:
        print(f"[nws] station snapshot not saved: {e}")


async def fetch_stations(client: httpx.AsyncClient | None = None) -> list:
    """Reliable surface stations for TX + NM/OK/AR/LA inside the bbox:
    [{id, name, lat, lon, provider}]. Cached 24 h (memory + snapshot file).
    On failure keeps serving the last catalog, however old, and retries after
    30 min. Returns [] only when no catalog has ever loaded. Never raises."""
    try:
        _load_stations_snapshot()
        if _stations["list"] and _age_h(_stations["fetched_at"]) < STATIONS_TTL_H:
            return _stations["list"]
        async with _lock("stations"):
            if _stations["list"] and _age_h(_stations["fetched_at"]) < STATIONS_TTL_H:
                return _stations["list"]
            if time.monotonic() < _stations["retry_after"]:
                return _stations["list"] or []
            own = client is None
            c = client or _client()
            try:
                parts = await asyncio.gather(*(_fetch_state(c, s) for s in STATES))
            finally:
                if own:
                    await c.aclose()
            ok = [p for p in parts if p is not None]
            now = datetime.now(timezone.utc).isoformat()
            if len(ok) == len(STATES):
                _install_stations([s for p in ok for s in p], now)
                _stations["retry_after"] = 0.0
                await asyncio.to_thread(_save_stations_snapshot)
                print(f"[nws] station catalog: {len(_stations['list'])} stations")
            else:
                failed = [s for s, p in zip(STATES, parts) if p is None]
                _stations["retry_after"] = time.monotonic() + STATIONS_RETRY_MIN * 60
                if not _stations["list"]:
                    partial = [s for p in ok for s in p]
                    if partial:
                        # Better a partial catalog than none; stamp it old so
                        # the next attempt (after the backoff) refetches it.
                        stale = (datetime.now(timezone.utc) - timedelta(hours=STATIONS_TTL_H)).isoformat()
                        _install_stations(partial, stale)
                print(f"[nws] station catalog fetch failed for {failed}; "
                      f"serving {len(_stations['list'] or [])} cached stations")
            return _stations["list"] or []
    except Exception as e:
        print(f"[nws] fetch_stations error: {e}")
        return _stations["list"] or []


def _haversine_km(lat, lon, lats, lons):
    p1, p2 = math.radians(lat), np.radians(lats)
    dphi = p2 - p1
    dlmb = np.radians(lons) - math.radians(lon)
    a = np.sin(dphi / 2) ** 2 + math.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def nearest_stations(lat: float, lon: float, k: int = MAX_STATIONS_PER_POINT) -> list:
    """Up to k cached-catalog stations nearest (lat, lon) within
    MAX_STATION_KM, as [(station, km)] nearest first."""
    lst = _stations["list"]
    if not lst:
        return []
    d = _haversine_km(float(lat), float(lon), _stations["lat"], _stations["lon"])
    order = np.argsort(d, kind="stable")[:k]
    return [(lst[i], float(d[i])) for i in order if d[i] <= MAX_STATION_KM]


# ── Observations ─────────────────────────────────────────────────────────

def _qv(prop) -> tuple:
    """(value, unit code) of an observation quantity, or (None, None) when
    missing, null, or rejected by quality control."""
    if not isinstance(prop, dict):
        return None, None
    v = prop.get("value")
    if v is None or prop.get("qualityControl") in _QC_REJECT:
        return None, None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None, None
    if not math.isfinite(v):
        return None, None
    return v, str(prop.get("unitCode") or "").split(":")[-1]


def _to_f(v, unit):
    if unit == "degC":
        return v * 9.0 / 5.0 + 32.0
    if unit == "degF":
        return v
    if unit == "K":
        return (v - 273.15) * 9.0 / 5.0 + 32.0
    return None


def _to_hpa(v, unit):
    if unit == "Pa":
        return v / 100.0
    if unit == "hPa":
        return v
    return None


def _altimeter_to_msl(alt_hpa, elev_m, temp_f):
    """Altimeter setting -> sea-level pressure (hPa). The altimeter setting is
    a standard-atmosphere reduction, which reads high at elevation when the
    air is warmer than standard (Amarillo 2026-11-03: A2978 = 1008.5 vs METAR
    SLP 1004.7). Undo it to station pressure (NWS/ASOS altimeter formula),
    then reduce with the observed temperature (hypsometric equation). Over 6
    hourly METARs this matched the reported SLP to 0.3 hPa MAE (altimeter
    alone: 3.4). Without elevation or temperature, the altimeter as is."""
    if alt_hpa is None or elev_m is None or temp_f is None or not (-100.0 <= elev_m <= 4000.0):
        return alt_hpa
    try:
        n = 0.190284
        p_stn = (alt_hpa ** n - 8.4228807e-5 * elev_m) ** (1.0 / n) + 0.3
        t_mean_k = (temp_f - 32.0) * 5.0 / 9.0 + 273.15 + 0.0065 * elev_m / 2.0
        p = p_stn * math.exp(9.80665 * elev_m / (287.05 * t_mean_k))
        return p if math.isfinite(p) else alt_hpa
    except (ValueError, ZeroDivisionError, OverflowError):
        return alt_hpa


def _to_mph(v, unit):
    if unit == "km_h-1":
        return v / 1.609344
    if unit == "m_s-1":
        return v * 2.2369363
    if unit in ("kn", "kt"):
        return v * 1.1507794
    if unit in ("mph", "mi_h-1"):
        return v
    return None


def _to_pct(v, unit):
    return v if unit == "percent" else None


def _bounded(field: str, v):
    if v is None:
        return None
    lo, hi = _BOUNDS[field]
    if field == "humidity" and hi < v <= hi + 3.0:
        v = hi  # RH computed from T/Td can round a hair over 100
    if not (lo <= v <= hi):
        return None
    return round(v, 1)


def parse_observation(body) -> dict | None:
    """One /observations/latest body -> {station_id, station_name, timestamp
    (aware datetime), fields: {temperature, humidity, pressure, wind_speed}}
    in display units (None per field when missing/rejected), or None."""
    try:
        p = (body or {}).get("properties")
        if not isinstance(p, dict) or not p.get("timestamp"):
            return None
        ts = datetime.fromisoformat(str(p["timestamp"]).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        temp_f = _bounded("temperature", _to_f(*_qv(p.get("temperature"))))
        elev, elev_unit = _qv(p.get("elevation"))
        slp = _to_hpa(*_qv(p.get("seaLevelPressure")))
        alt = _altimeter_to_msl(_to_hpa(*_qv(p.get("barometricPressure"))),
                                elev if elev_unit == "m" else None, temp_f)
        fields = {
            "temperature": temp_f,
            "humidity":    _bounded("humidity", _to_pct(*_qv(p.get("relativeHumidity")))),
            "pressure":    _bounded("pressure", slp) or _bounded("pressure", alt),
            "wind_speed":  _bounded("wind_speed", _to_mph(*_qv(p.get("windSpeed")))),
        }
        return {"station_id": p.get("stationId"), "station_name": p.get("stationName"),
                "timestamp": ts.astimezone(timezone.utc), "fields": fields}
    except Exception:
        return None


def _is_fresh(obs: dict | None, now: datetime) -> bool:
    if not obs:
        return False
    age_h = (now - obs["timestamp"]).total_seconds() / 3600.0
    return -0.25 <= age_h <= MAX_OBS_AGE_H


async def _latest_obs(client: httpx.AsyncClient, sid: str, sem: asyncio.Semaphore,
                      calls: dict | None = None) -> dict | None:
    """Latest parsed observation for a station, through the shared 20-min
    cache (failures are negatively cached for 5 min). calls["n"] counts the
    requests actually sent."""
    hit = _obs_cache.get(sid)
    if hit and time.monotonic() < hit["expires"]:
        _obs_cache.move_to_end(sid)
        return hit["obs"]
    if calls is not None:
        calls["n"] += 1
    async with sem:
        status, body = await _get_json(client, f"{API_BASE}/stations/{sid}/observations/latest")
    obs = parse_observation(body) if status == 200 else None
    ttl = OBS_TTL_MIN if obs else OBS_FAIL_TTL_MIN
    _obs_cache[sid] = {"obs": obs, "expires": time.monotonic() + ttl * 60}
    _obs_cache.move_to_end(sid)
    while len(_obs_cache) > OBS_CACHE_MAX:
        _obs_cache.popitem(last=False)
    return obs


def _assemble(cands: list, obs_by_id: dict, now: datetime):
    """Combine the candidates' observations (nearest first) into one display
    dict. Each field comes from the nearest station with a fresh, valid
    value. Returns (dict or None, set of missing fields). The station/
    observed_at labels name the station that supplied the temperature."""
    vals, primary = {}, None
    for st, km in cands:
        obs = obs_by_id.get(st["id"])
        if not _is_fresh(obs, now):
            continue
        for f in FIELDS:
            if f not in vals and obs["fields"].get(f) is not None:
                vals[f] = obs["fields"][f]
                if f == "temperature":
                    primary = (st, km, obs)
    missing = set(FIELDS) - set(vals)
    if missing or primary is None:
        return None, missing
    st, km, obs = primary
    out = {f: vals[f] for f in FIELDS}
    out.update({"source": "nws", "station": st["id"],
                "station_name": obs.get("station_name") or st["name"],
                "observed_at": obs["timestamp"].isoformat()})
    return out, missing


def _with_borrowed(lat: float, lon: float, cands: list, obs_by_id: dict) -> list:
    """cands extended with farther stations (up to BORROW_K, <= BORROW_MAX_KM)
    whose observation is already in hand. Fills a field that the 3 nearest
    all lack (often pressure: many AWOS 'latest' reports carry no altimeter)
    without sending another request; nearer stations still win per field."""
    have = {st["id"] for st, _ in cands}
    extra = [(st, km) for st, km in nearest_stations(lat, lon, k=BORROW_K)
             if st["id"] not in have and km <= BORROW_MAX_KM and st["id"] in obs_by_id]
    return list(cands) + extra


def _cached_obs() -> dict:
    """Unexpired observations in the shared cache, by station id."""
    t = time.monotonic()
    return {sid: h["obs"] for sid, h in _obs_cache.items() if t < h["expires"] and h["obs"]}


# ── Public API ───────────────────────────────────────────────────────────

async def fetch_conditions_grid(lats, lons) -> dict:
    """CURRENT conditions per 1.0 deg cell, in display units, from NWS
    stations. Same shape as airquality.fetch_conditions_grid (keys from
    airquality.conditions_cell_key) plus 'source': 'nws' and per-cell
    station / station_name / observed_at / source. A cell is included only
    when all four fields were found (3 nearest stations, then observations
    already in hand within 100 km); the caller's nearest-cell lookup covers
    the rest. Bounded by GRID_TIMEOUT_S so a hanging API cannot stall the
    refresh loop (a timeout returns an unusable grid). Never raises."""
    out = {"by_key": {}, "n_cells": 0, "usable": False, "source": "nws",
           "fetched_at": datetime.now(timezone.utc).isoformat(),
           "n_stations": 0, "n_station_calls": 0, "station_km": None}
    try:
        await asyncio.wait_for(_fill_grid(out, lats, lons), GRID_TIMEOUT_S)
    except asyncio.TimeoutError:
        print(f"[nws] conditions grid timed out after {GRID_TIMEOUT_S:.0f} s")
        out.update({"by_key": {}, "usable": False, "station_km": None})
    except Exception as e:
        print(f"[nws] conditions grid error: {e}")
        out.update({"by_key": {}, "usable": False})
    return out


async def _fill_grid(out: dict, lats, lons) -> None:
    """The body of fetch_conditions_grid; fills out in place."""
    cells, _ = build_cells(lats, lons, grid_deg=WEATHER_GRID_DEG)
    out["n_cells"] = len(cells)
    if not cells:
        return
    async with _client() as client:
        stations = await fetch_stations(client)
        out["n_stations"] = len(stations)
        if not stations:
            return
        cands = [nearest_stations(la, lo) for la, lo in cells]
        sem = asyncio.Semaphore(CONCURRENCY)
        obs_by_id, done, calls = {}, [None] * len(cells), {"n": 0}
        # Round r asks only the r-th nearest station of every cell still
        # missing something, so a healthy refresh is one call per cell.
        for rank in range(MAX_STATIONS_PER_POINT):
            need = set()
            for i, cs in enumerate(cands):
                if done[i] is None and rank < len(cs):
                    sid = cs[rank][0]["id"]
                    if sid not in obs_by_id:
                        need.add(sid)
            ids = sorted(need)
            got = await asyncio.gather(*(_latest_obs(client, s, sem, calls) for s in ids))
            obs_by_id.update(zip(ids, got))
            now = datetime.now(timezone.utc)
            for i, cs in enumerate(cands):
                if done[i] is None:
                    done[i] = _assemble(cs[:rank + 1], obs_by_id, now)[0]
            if ids and not any(obs_by_id.values()):
                # Not one observation parsed: the API is down or blocking us.
                # Asking the next ~80 stations would only repeat the failure.
                print(f"[nws] no station answered in round {rank + 1}; giving up this refresh")
                break
        out["n_station_calls"] = calls["n"]
        pool = {**_cached_obs(), **obs_by_id}
        for i, (la, lo) in enumerate(cells):
            if done[i] is None and cands[i]:
                done[i] = _assemble(_with_borrowed(la, lo, cands[i], pool), pool, now)[0]
    by_key, kms = {}, []
    for (la, lo), v in zip(cells, done):
        if v is None:
            continue
        by_key[conditions_cell_key(la, lo)] = v
        kms.extend(km for st, km in nearest_stations(la, lo, k=BORROW_K) if st["id"] == v["station"])
    if kms:
        out["station_km"] = {"median": round(float(np.median(kms)), 1),
                             "p90": round(float(np.percentile(kms, 90)), 1),
                             "max": round(float(max(kms)), 1)}
    out["by_key"] = by_key
    out["usable"] = len(by_key) > 0
    print(f"[nws] conditions grid: {len(by_key)}/{len(cells)} cells, "
          f"{out['n_station_calls']} station calls")


async def fetch_point_current(lat: float, lon: float) -> dict | None:
    """Current conditions at an exact location (a clicked tract) from the
    nearest station with a fresh observation, per field from the next-nearest
    (up to 3, then cached observations within 100 km), through the shared
    per-station cache. Returns
    {temperature, humidity, pressure, wind_speed, source, station,
    station_name, observed_at} or None (the caller falls back to the grid).
    Bounded by POINT_TIMEOUT_S, since a click waits on it (3 stations with a
    retry each could otherwise take about a minute). After a timeout, returns
    None at once for OBS_FAIL_TTL_MIN, so one slow click pays the wait during
    an outage, not every click. Never raises."""
    if time.monotonic() < _point_backoff["until"]:
        return None
    try:
        return await asyncio.wait_for(_point_current(lat, lon), POINT_TIMEOUT_S)
    except asyncio.TimeoutError:
        _point_backoff["until"] = time.monotonic() + OBS_FAIL_TTL_MIN * 60
        print(f"[nws] point conditions timed out after {POINT_TIMEOUT_S:.0f} s; "
              f"skipping point lookups for {OBS_FAIL_TTL_MIN} min")
        return None
    except Exception as e:
        print(f"[nws] point conditions error: {e}")
        return None


async def _point_current(lat: float, lon: float) -> dict | None:
    if not await fetch_stations():
        return None
    cands = nearest_stations(lat, lon)
    if not cands:
        return None
    sem = asyncio.Semaphore(1)
    obs_by_id = {}
    async with _client() as client:
        for rank, (st, _km) in enumerate(cands):
            obs_by_id[st["id"]] = await _latest_obs(client, st["id"], sem)
            v = _assemble(cands[:rank + 1], obs_by_id, datetime.now(timezone.utc))[0]
            if v is not None:
                return v
    pool = {**_cached_obs(), **obs_by_id}
    return _assemble(_with_borrowed(lat, lon, cands, pool), pool, datetime.now(timezone.utc))[0]

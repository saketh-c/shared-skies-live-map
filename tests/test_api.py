"""HTTP contract of the public API, through the real app and lifespan, with
no PurpleAir key and no network (the startup compute uses climatology)."""
import asyncio
import gzip
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import backend.main as main
from backend.ratelimit import RecentKeys

TOP_LEVEL_KEYS = ["region", "display_name", "generated_at", "expires_at", "weather",
                  "avg_pm25", "avg_epa_aqi", "avg_info", "tract_count", "calibration",
                  "data_sources", "value_semantics", "tracts"]
TRACT_KEYS = ["geoid", "lat", "lon", "pm25", "pm25_model", "epa_aqi", "category", "color",
              "aqi_range", "health_msg", "ejf_score", "pct_people_of_color", "pct_low_income",
              "traffic_proximity", "superfund_proximity", "diesel_pm_proximity",
              "pct_ling_isolated", "county"]
DATA_SOURCE_KEYS = {"live_purpleair_sensors", "using_live_neighbors", "hms_polygons",
                    "hms_data_date", "weather_grid_cells", "purpleair_fetched_at",
                    "purpleair_key_configured", "hms_is_proxy", "refresh_interval_min",
                    "conditions_source", "airnow_monitors"}
HEALTH_NEW_KEYS = {"status", "purpleair_key_configured", "live_neighbors",
                   "predictions_age_min", "purpleair_age_min", "refresh_interval_min",
                   "memory_rss_mb", "conditions_source", "weather_grid_fetched_at",
                   "airquality_fetched_at", "airnow_monitors", "airnow_window_end",
                   "calibration_applied", "calibration_mae_loo"}


# ── /api/health ──────────────────────────────────────────────────────────────

def test_health_without_key_is_degraded(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    h = r.json()
    assert HEALTH_NEW_KEYS | {"model_loaded", "model_features", "lookup_loaded",
                              "live_purpleair_sensors", "data_sources"} <= set(h)
    assert h["status"] == "degraded"
    assert h["purpleair_key_configured"] is False
    assert h["live_neighbors"] is False
    assert h["purpleair_age_min"] is None
    assert h["refresh_interval_min"] == 15
    assert h["model_loaded"] is True and h["model_features"] == 30
    assert isinstance(h["predictions_age_min"], (int, float)) and h["predictions_age_min"] >= 0
    assert h["memory_rss_mb"] is None or h["memory_rss_mb"] > 0


def _fake_texas(age_min, live=True):
    generated = datetime.now(timezone.utc) - timedelta(minutes=age_min)
    return {"data": {"generated_at": generated.isoformat(),
                     "data_sources": {"using_live_neighbors": live}},
            "expires": generated + timedelta(minutes=15)}


@pytest.mark.parametrize("age_min, live, model, status", [
    (2, True, True, "ok"),
    (44, True, True, "ok"),
    (46, True, True, "degraded"),     # older than 3 x refresh_interval_min
    (2, False, True, "degraded"),     # climatology neighbors
    (2, True, False, "degraded"),     # model not loaded
])
def test_health_status_rules(client, monkeypatch, age_min, live, model, status):
    monkeypatch.setitem(main.state, "cache_texas", _fake_texas(age_min, live))
    if not model:
        monkeypatch.setitem(main.state, "bundle", None)
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == status


def test_health_is_cheap(client, net):
    # Loose bound for slow CI runners; the real check is no upstream call.
    t0 = time.monotonic()
    client.get("/api/health")
    assert time.monotonic() - t0 < 2.0
    assert net.requests == []


# ── /api/texas/predictions ───────────────────────────────────────────────────

def test_predictions_shape(predictions):
    d = predictions
    assert list(d) == TOP_LEVEL_KEYS
    assert d["region"] == "texas"
    assert d["tract_count"] == len(d["tracts"]) == 6896
    assert list(d["tracts"][0]) == TRACT_KEYS
    assert set(d["data_sources"]) == DATA_SOURCE_KEYS
    assert {"category", "color", "aqi_range", "health_msg"} <= set(d["avg_info"])


def test_predictions_without_key(predictions):
    ds = predictions["data_sources"]
    assert ds["using_live_neighbors"] is False
    assert ds["live_purpleair_sensors"] == 0
    assert ds["purpleair_fetched_at"] is None
    assert ds["purpleair_key_configured"] is False
    assert ds["refresh_interval_min"] == 15
    # NOAA is unreachable offline, so there is no current smoke.
    assert ds["hms_polygons"] == 0 and ds["hms_data_date"] is None
    assert ds["hms_is_proxy"] is False


def test_predictions_values_are_sane(predictions):
    pm = [t["pm25"] for t in predictions["tracts"]]
    assert all(0.0 <= v < 60.0 for v in pm)
    assert len({t["geoid"] for t in predictions["tracts"]}) == 6896


def test_expires_at_is_next_refresh(predictions):
    g = datetime.fromisoformat(predictions["generated_at"])
    e = datetime.fromisoformat(predictions["expires_at"])
    assert (e - g).total_seconds() == pytest.approx(15 * 60, abs=1)


def test_predictions_gzip_etag_and_304(client, predictions):
    r = client.get("/api/texas/predictions", headers={"Accept-Encoding": "gzip"})
    assert r.status_code == 200
    assert r.headers["content-encoding"] == "gzip"
    assert r.headers["cache-control"] == "no-cache"
    assert "accept-encoding" in r.headers["vary"].lower()
    etag = r.headers["etag"]
    assert etag.startswith('"') and etag.endswith('"')
    assert r.json() == predictions

    r304 = client.get("/api/texas/predictions",
                      headers={"Accept-Encoding": "gzip", "If-None-Match": etag})
    assert r304.status_code == 304
    assert r304.content == b""
    assert r304.headers["etag"] == etag

    # Weak and cross-encoding tags match too (proxies re-encode or weaken).
    r_weak = client.get("/api/texas/predictions",
                        headers={"Accept-Encoding": "identity", "If-None-Match": "W/" + etag})
    assert r_weak.status_code == 304

    r_other = client.get("/api/texas/predictions", headers={"If-None-Match": '"stale"'})
    assert r_other.status_code == 200


def test_predictions_identity_encoding(client, predictions):
    r = client.get("/api/texas/predictions", headers={"Accept-Encoding": "identity"})
    assert r.status_code == 200
    assert "content-encoding" not in r.headers
    assert json.loads(r.content) == predictions
    # The gzip body is the same bytes, compressed.
    payload = main.state["texas_payload"]
    assert gzip.decompress(payload["gz"]) == payload["raw"] == r.content
    assert r.headers["etag"] == payload["etag"]


def test_city_and_tract_match_the_map(client, predictions):
    hou = [t for t in predictions["tracts"] if t["geoid"].startswith("48201")]
    city = client.get("/api/houston/predictions").json()
    assert city["generated_at"] == predictions["generated_at"]
    assert [t["pm25"] for t in city["tracts"]] == [t["pm25"] for t in hou]

    t0 = hou[0]
    t = client.get(f"/api/tract/{t0['geoid']}").json()
    assert t["pm25"] == t0["pm25"] and t["from_map_batch"] is True
    assert client.get("/api/tract/99999999999").status_code == 404
    assert client.get("/api/nowhere/predictions").status_code == 404


# ── /api/visit ───────────────────────────────────────────────────────────────

@pytest.fixture
def fresh_visits(monkeypatch):
    monkeypatch.setattr(main, "_visit_dedupe", RecentKeys(main.VISIT_DEDUPE_MIN * 60.0))
    monkeypatch.setattr(main, "_public_visits_cache",
                        {"count": None, "expires": main._EPOCH})


def test_visit_counts_once_per_ip(client, fresh_visits):
    start = main._get_visit_count_sync()

    def visit(xff):
        r = client.post("/api/visit", headers={"X-Forwarded-For": xff})
        assert r.status_code == 200
        body = r.json()
        assert set(body) == {"visits"} and isinstance(body["visits"], int)
        return main._get_visit_count_sync()

    assert visit("203.0.113.10") == start + 1
    assert visit("203.0.113.10") == start + 1
    assert visit("203.0.113.10, 10.0.0.1") == start + 1   # same left-most client
    assert visit("203.0.113.11") == start + 2


def test_visit_window_is_30_min():
    assert main.VISIT_DEDUPE_MIN == 30
    assert main._visit_dedupe.ttl_s == 30 * 60


def test_metrics_shape(client):
    r = client.get("/api/metrics")
    assert r.status_code == 200 and set(r.json()) == {"visits"}


# ── /api/quantum/sensor-placement ────────────────────────────────────────────

def test_sensor_placement_contract(client):
    r = client.get("/api/quantum/sensor-placement")
    assert r.status_code == 200
    q = r.json()
    qa = q["methods"]["quantum_annealing"]
    assert set(qa["weights"]) == {"coverage", "error", "ej", "pm25"}
    assert sum(qa["weights"].values()) == pytest.approx(1.0)
    assert qa["feasible"] is True
    assert "simulated annealing" in qa["method_display"].lower()
    assert qa["num_sensors"] == q["num_sensors"] == len(qa["selected_tracts"]) == 25
    # No key, so the network is the training sensor set.
    assert q["num_existing_sensors"] == len(q["existing_sensors"]) == 467
    for s in q["existing_sensors"]:
        assert {"lat", "lon"} <= set(s)
    assert {"coverage", "timing", "num_candidates", "best_energy"} <= set(qa)


def test_sensor_placement_cold_requests_share_one_solve(client, monkeypatch):
    calls = []
    release = threading.Event()

    async def fake_solve():
        calls.append(1)
        while not release.is_set():
            await asyncio.sleep(0.01)
        result = {"methods": {}, "existing_sensors": [], "num_existing_sensors": 0}
        main._quantum_cache = {"data": result,
                               "expires": datetime.now(timezone.utc) + timedelta(hours=1)}
        return result

    monkeypatch.setattr(main, "_quantum_cache", {"data": None, "expires": main._EPOCH})
    monkeypatch.setattr(main, "_solve_quantum_and_cache", fake_solve)

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(client.get, "/api/quantum/sensor-placement") for _ in range(5)]
        time.sleep(0.3)
        release.set()
        responses = [f.result(timeout=10) for f in futures]
    assert [r.status_code for r in responses] == [200] * 5
    assert len(calls) == 1


def test_solver_inputs_from_main(client, monkeypatch):
    """main.py hands the solver NaN (unknown) for tracts with no LOSO residual,
    the current network as plain {lat, lon}, and the documented parameters."""
    from backend.quantum import qubo_solver

    seen = {}

    def fake_solve(tracts, **kwargs):
        seen["tracts"], seen["kwargs"] = tracts, kwargs
        return {"method_display": "test", "weights": dict(qubo_solver.WEIGHTS), "feasible": True,
                "num_sensors": 0, "num_candidates": 0, "num_reads": 0, "best_energy": 0.0,
                "timing": {}, "selected_tracts": []}

    monkeypatch.setattr(qubo_solver, "solve_quantum", fake_solve)
    monkeypatch.setattr(qubo_solver, "compute_coverage", lambda *a, **k: {})
    result = asyncio.run(main._run_quantum_placement())

    kw = seen["kwargs"]
    assert (kw["k"], kw["num_reads"], kw["top_candidates"], kw["proximity_threshold_miles"]) == (
        25, 500, 120, 8.0)
    md = kw["model_disagreement"]
    residuals = main._load_loso_residuals()
    known = np.array([str(t["geoid"]) in residuals for t in seen["tracts"]])
    assert md.dtype == np.float64 and len(md) == len(seen["tracts"]) == 6896
    assert 0 < known.sum() < len(known)
    assert np.isnan(md[~known]).all()
    assert np.isfinite(md[known]).all()
    assert all({"lat", "lon"} <= set(s) for s in kw["existing_sensors"])
    assert result["num_existing_sensors"] == len(kw["existing_sensors"])
    assert result["methods"]["quantum_annealing"]["feasible"] is True


# ── EPA monitor calibration and /api/live/sensors ────────────────────────────

RAW_SEMANTICS = "Model-predicted 24-hour-average PM2.5 (µg/m³) per census tract"


def test_offline_map_is_uncalibrated_and_says_so(client, predictions):
    cal = predictions["calibration"]
    assert cal["applied"] is False
    assert cal["loo"] is None and cal["reused_from"] is None
    assert predictions["value_semantics"] == RAW_SEMANTICS
    assert predictions["data_sources"]["airnow_monitors"] == 0
    # NWS and Open-Meteo are both unreachable offline.
    assert predictions["data_sources"]["conditions_source"] == "fallback"
    assert predictions["weather"]["source"] == "fallback"
    assert all(t["pm25"] == t["pm25_model"] for t in predictions["tracts"])

    s = client.get("/api/live/sensors").json()
    assert s["generated_at"] == predictions["generated_at"]
    assert s["calibration_applied"] is False
    assert s["purpleair"]["count"] == 0 and s["purpleair"]["sensors"] == []
    assert s["monitors"]["count"] == 0 and s["monitors"]["monitors"] == []

    h = client.get("/api/health").json()
    assert h["calibration_applied"] is False and h["calibration_mae_loo"] is None
    assert h["airnow_monitors"] == 0 and h["conditions_source"] == "fallback"


@pytest.fixture
def restore_texas_state(monkeypatch):
    """Put the startup result back after a test installs its own computes."""
    for k in ("cache_texas", "cache_texas_by_geoid", "texas_payload", "city_cache",
              "live_points", "last_calibration"):
        monkeypatch.setitem(main.state, k, main.state.get(k))


# Out-of-state monitors: one far from any Texas tract, one beside El Paso.
OKC = {"id": "401090033", "name": "Oklahoma City test", "lat": 35.47, "lon": -97.52,
       "pm25": 6.0, "hours": 24}
ANTHONY_NM = {"id": "350130021", "name": "Anthony NM test", "lat": 31.997, "lon": -106.598,
              "pm25": 5.0, "hours": 24}


def _fake_airnow(lookup, n=28, step=211):
    """n synthetic monitors a little off tract centroids spread over Texas,
    plus the two out-of-state ones (30 in all)."""
    rows = lookup.iloc[::step].head(n)
    monitors = [{"id": f"48{i:07d}", "name": f"Test monitor {i}",
                 "lat": float(la) + 0.01, "lon": float(lo) - 0.01,
                 "pm25": 4.0 + (i % 5), "hours": 24}
                for i, (la, lo) in enumerate(zip(rows["lat"], rows["lon"]))]
    monitors += [dict(OKC), dict(ANTHONY_NM)]
    return {"usable": True, "window_start": "2026-11-01T19:00:00+00:00",
            "window_end": "2026-11-02T18:00:00+00:00", "monitors": monitors, "reason": None}


def test_calibrated_cycle_end_to_end(client, monkeypatch, no_snapshots, restore_texas_state):
    from backend import calibration as C

    airnow = _fake_airnow(main.state["tract_lookup"])
    monkeypatch.setattr(main, "_current_airnow_means", lambda now: airnow)
    monkeypatch.setitem(main.state, "last_calibration", None)
    result = asyncio.run(main._compute_texas_predictions())

    cal = result["calibration"]
    assert cal["applied"] is True and cal["method"] == C.METHOD_ID
    assert cal["monitors_used"] == 30 and cal["reused_from"] is None
    assert cal["loo"]["n"] == 30 and cal["uncalibrated"]["n"] == 30
    assert (cal["window_start"], cal["window_end"]) == (airnow["window_start"],
                                                        airnow["window_end"])
    assert result["data_sources"]["airnow_monitors"] == 30
    assert result["value_semantics"] == RAW_SEMANTICS + ", calibrated to EPA AirNow monitors"
    assert list(result) == TOP_LEVEL_KEYS

    # Displayed pm25 = the fit applied to the raw model; AQI and colors follow it.
    fit = C.Calibration.from_dict(main.state["last_calibration"]["calibration"])
    tr = result["tracts"]
    lat = np.array([t["lat"] for t in tr])
    lon = np.array([t["lon"] for t in tr])
    raw = np.array([t["pm25_model"] for t in tr])
    shown = np.array([t["pm25"] for t in tr])
    assert np.allclose(fit.apply(lat, lon, raw), shown, atol=0.011)
    assert (shown >= 0).all() and not np.array_equal(shown, raw)
    assert all(t["epa_aqi"] == main.pm25_to_epa_aqi(t["pm25"]) for t in tr[:500])
    assert all(t["color"] == main.pm25_info(t["pm25"])["color"] for t in tr[:500])

    # Every endpoint serves this same cycle.
    assert client.get("/api/texas/predictions").json()["calibration"] == cal
    t = client.get(f"/api/tract/{tr[0]['geoid']}").json()
    assert (t["pm25"], t["pm25_model"]) == (tr[0]["pm25"], tr[0]["pm25_model"])
    s = client.get("/api/live/sensors").json()
    assert s["generated_at"] == result["generated_at"] and s["calibration_applied"] is True
    mons = s["monitors"]["monitors"]
    assert s["monitors"]["count"] == len(mons) == 30
    assert {"id", "name", "lat", "lon", "pm25", "hours", "map_pm25", "map_pm25_loo",
            "model_pm25"} == set(mons[0])
    assert all(m["map_pm25_loo"] is not None and m["model_pm25"] is not None for m in mons)
    # The map draws no tract near Oklahoma City, so it has no "map value" there.
    by_id = {m["id"]: m for m in mons}
    assert by_id[OKC["id"]]["map_pm25"] is None
    assert by_id[ANTHONY_NM["id"]]["map_pm25"] is not None
    assert all(m["map_pm25"] is not None for m in mons if m["id"].startswith("48"))
    loo = np.array([m["map_pm25_loo"] for m in mons])
    truth = np.array([m["pm25"] for m in mons])
    assert abs(float(np.mean(np.abs(loo - truth))) - cal["loo"]["mae"]) < 0.06
    h = client.get("/api/health").json()
    assert h["calibration_applied"] is True and h["airnow_monitors"] == 30
    assert h["calibration_mae_loo"] == cal["loo"]["mae"]
    assert h["airnow_window_end"] == airnow["window_end"]

    # AirNow down next cycle: the fit is reused (no fresh held-out score).
    monkeypatch.setattr(main, "_current_airnow_means", lambda now: None)
    again = asyncio.run(main._compute_texas_predictions())
    cal2 = again["calibration"]
    assert cal2["applied"] is True and cal2["reused_from"] == cal["fitted_at"]
    assert cal2["loo"] is None and cal2["monitors_used"] == 30
    assert again["data_sources"]["airnow_monitors"] == 0
    assert any(t["pm25"] != t["pm25_model"] for t in again["tracts"])

    # Past CALIBRATION_REUSE_MAX_H the map goes back to raw values and says so.
    old = dict(main.state["last_calibration"])
    old["fitted_at"] = (datetime.now(timezone.utc)
                        - timedelta(hours=main.CALIBRATION_REUSE_MAX_H + 1)).isoformat()
    monkeypatch.setitem(main.state, "last_calibration", old)
    stale = asyncio.run(main._compute_texas_predictions())
    assert stale["calibration"]["applied"] is False
    assert stale["value_semantics"] == RAW_SEMANTICS
    assert all(t["pm25"] == t["pm25_model"] for t in stale["tracts"])


def test_calibration_error_serves_raw_map(client, monkeypatch, no_snapshots,
                                          restore_texas_state):
    from backend import calibration as C

    def boom(*a, **k):
        raise RuntimeError("calibration bug")

    monkeypatch.setattr(main, "_current_airnow_means",
                        lambda now: _fake_airnow(main.state["tract_lookup"]))
    monkeypatch.setitem(main.state, "last_calibration", None)
    monkeypatch.setattr(C, "fit", boom)
    result = asyncio.run(main._compute_texas_predictions())
    assert result["calibration"]["applied"] is False
    assert all(t["pm25"] == t["pm25_model"] for t in result["tracts"])
    s = client.get("/api/live/sensors").json()
    assert s["calibration_applied"] is False and s["monitors"]["count"] == 30


def test_airnow_means_keep_only_regulatory_monitors(monkeypatch):
    from backend.airnow import AirNowStore, target_hours

    now = datetime(2026, 11, 2, 20, 30, tzinfo=timezone.utc)
    ids = [f"48{i:07d}" for i in range(12)] + ["MMFS12345"]
    store = AirNowStore()
    store.sites = {a: {"id": a, "name": a, "lat": 30.0 + i * 0.1, "lon": -97.0,
                       "aliases": []} for i, a in enumerate(ids)}
    store.hours = {h: {"fetched_at": now, "etag": None, "values": {a: 6.0 for a in ids}}
                   for h in target_hours(now)[:24]}
    monkeypatch.setattr(main, "_airnow_store", store)
    m = main._current_airnow_means(now)
    assert m["usable"] is True
    assert [x["id"] for x in m["monitors"]] == ids[:12]

    monkeypatch.setattr(main, "_airnow_store", AirNowStore())
    assert main._current_airnow_means(now) is None


def test_airnow_refresh_never_blocks_the_compute(monkeypatch, no_snapshots):
    from backend.airnow import AirNowStore

    class Slow(AirNowStore):
        async def refresh(self, now=None):
            await asyncio.sleep(10)

    monkeypatch.setattr(main, "_airnow_store", Slow())
    monkeypatch.setattr(main, "AIRNOW_REFRESH_TIMEOUT_S", 0.2)
    t0 = time.monotonic()
    asyncio.run(main._refresh_airnow())        # times out quietly
    assert time.monotonic() - t0 < 2.0


def test_conditions_prefer_nws_then_open_meteo(client, monkeypatch, no_snapshots):
    from backend import airquality, nws

    austin = airquality.conditions_cell_key(30.2672, -97.7431)
    cell = {"temperature": 70.0, "humidity": 50.0, "pressure": 1015.0, "wind_speed": 5.0}

    async def nws_grid(lats, lons):
        return {"by_key": {austin: {**cell, "source": "nws", "station": "KATT",
                                    "station_name": "Austin Camp Mabry",
                                    "observed_at": "2026-11-02T20:00:00+00:00"}},
                "n_cells": 1, "usable": True, "source": "nws",
                "fetched_at": datetime.now(timezone.utc).isoformat()}

    async def nws_down(lats, lons):
        return {"by_key": {}, "n_cells": 0, "usable": False, "source": "nws"}

    async def open_meteo(lats, lons):
        return {"by_key": {austin: dict(cell)}, "n_cells": 1, "usable": True,
                "fetched_at": datetime.now(timezone.utc).isoformat()}

    monkeypatch.setattr(airquality, "fetch_conditions_grid", open_meteo)
    for grid, source in ((nws_grid, "nws"), (nws_down, "open-meteo")):
        monkeypatch.setattr(nws, "fetch_conditions_grid", grid)
        monkeypatch.setattr(main, "_conditions_cache", {"data": None, "expires": main._EPOCH})
        w = asyncio.run(main.get_display_weather(30.2672, -97.7431))
        assert w["source"] == source and w["temperature"] == 70.0
    assert w.get("station") is None


def test_click_weather_prefers_nws(monkeypatch):
    from backend import nws

    obs = {"temperature": 71.0, "humidity": 40.0, "pressure": 1012.0, "wind_speed": 3.0,
           "source": "nws", "station": "KAUS", "station_name": "Austin-Bergstrom",
           "observed_at": "2026-11-02T20:00:00+00:00"}

    async def point(lat, lon):
        return obs

    async def no_point(lat, lon):
        return None

    monkeypatch.setattr(nws, "fetch_point_current", point)
    assert asyncio.run(main.get_click_weather(30.2, -97.7)) == obs

    # NWS has nothing: the Open-Meteo path (offline here) then the grid fallback.
    monkeypatch.setattr(nws, "fetch_point_current", no_point)
    monkeypatch.setitem(main._click_weather_cooldown, "until", main._EPOCH)
    monkeypatch.setattr(main, "_conditions_cache", {"data": None, "expires": main._EPOCH})
    w = asyncio.run(main.get_click_weather(30.2, -97.7))
    assert w["source"] == "fallback"


def test_hanging_nws_does_not_hold_up_the_compute(client, monkeypatch, no_snapshots,
                                                  restore_texas_state):
    """Display weather is waited on for at most CONDITIONS_COMPUTE_WAIT_S; the
    Austin point lookup is skipped that cycle and the map still publishes."""
    from backend import nws

    async def hang(lats, lons):
        await asyncio.sleep(30)

    point_calls = []

    async def point(lat, lon):
        point_calls.append((lat, lon))
        return None

    monkeypatch.setattr(nws, "fetch_conditions_grid", hang)
    monkeypatch.setattr(nws, "fetch_point_current", point)
    monkeypatch.setattr(main, "CONDITIONS_COMPUTE_WAIT_S", 0.3)
    monkeypatch.setattr(main, "_conditions_cache", {"data": None, "expires": main._EPOCH})
    monkeypatch.setattr(main, "_current_airnow_means", lambda now: None)
    t0 = time.monotonic()
    result = asyncio.run(main._compute_texas_predictions())
    assert time.monotonic() - t0 < 15
    assert result["weather"]["source"] == "fallback"
    assert result["data_sources"]["conditions_source"] == "fallback"
    assert point_calls == []


def test_failed_conditions_fetch_backs_off_without_a_cache(client, monkeypatch, no_snapshots):
    from backend import airquality, nws

    calls = []

    async def down(lats, lons):
        calls.append(1)
        return {"by_key": {}, "n_cells": 0, "usable": False, "source": "nws"}

    async def om_down(lats, lons):
        return None

    monkeypatch.setattr(nws, "fetch_conditions_grid", down)
    monkeypatch.setattr(airquality, "fetch_conditions_grid", om_down)
    monkeypatch.setattr(main, "_conditions_cache", {"data": None, "expires": main._EPOCH})
    for _ in range(3):
        w = asyncio.run(main.get_display_weather(30.2672, -97.7431))
        assert w["source"] == "fallback"
    assert len(calls) == 1
    assert main._conditions_cache["expires"] > datetime.now(timezone.utc) + timedelta(minutes=14)

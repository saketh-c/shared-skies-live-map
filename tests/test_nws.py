"""NWS display weather: station catalog filtering and pagination, unit
conversions, stale/null fallback to the next station, the grid shape that
main.get_display_weather reads, and never raising on failures. Offline: every
api.weather.gov request goes through the conftest 'net' router."""
import asyncio
import re
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs

import httpx
import pytest

from backend import airquality, nws

HOST = "api.weather.gov"


@pytest.fixture(autouse=True)
def _fresh_nws(monkeypatch, tmp_path):
    nws._reset_state()
    monkeypatch.setattr(nws, "RETRY_DELAY_S", 0.0)
    monkeypatch.setattr(nws, "STATIONS_SNAPSHOT_PATH", str(tmp_path / "nws_stations.json"))
    yield
    nws._reset_state()


def _feature(sid, lat, lon, provider="ASOS-HFM", name=None):
    return {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"stationIdentifier": sid, "name": name or f"{sid} Airport",
                           "provider": provider, "subProvider": ""}}


def _q(value, unit, qc="V"):
    return {"unitCode": f"wmoUnit:{unit}", "value": value, "qualityControl": qc}


def _obs(sid, *, age_min=20, t=20.0, rh=50.0, slp=None, alt=101325.0, wind=18.52,
         t_unit="degC", wind_unit="km_h-1", p_unit="Pa", qc=None, elev=None):
    """An /observations/latest body; values in the given units (None = null).
    elev (m) is omitted by default, which keeps the altimeter setting as is."""
    qc = qc or {}
    ts = (datetime.now(timezone.utc) - timedelta(minutes=age_min)).isoformat()
    return {"type": "Feature", "properties": {
        "stationId": sid, "stationName": f"{sid} Field", "timestamp": ts,
        **({"elevation": {"unitCode": "wmoUnit:m", "value": elev}} if elev is not None else {}),
        "temperature": _q(t, t_unit, qc.get("temperature", "V" if t is not None else "Z")),
        "relativeHumidity": _q(rh, "percent", qc.get("relativeHumidity", "V" if rh is not None else "Z")),
        "seaLevelPressure": _q(slp, p_unit, "V" if slp is not None else "Z"),
        "barometricPressure": _q(alt, p_unit, "V" if alt is not None else "Z"),
        "windSpeed": _q(wind, wind_unit, qc.get("windSpeed", "V" if wind is not None else "Z")),
    }}


def _route(net, stations=None, obs=None, pages=None):
    """Serve a one-page catalog per state (stations = {state: [features]}),
    or explicit cursor pages (pages = {state: [[features], ...]}), and
    observations from obs = {sid: body | int status}."""
    stations = stations or {}
    obs = obs or {}

    def handler(request):
        path = request.url.path
        if path == "/stations":
            qs = parse_qs(request.url.query.decode())
            state = (qs.get("state") or [None])[0]
            cursor = int((qs.get("cursor") or ["0"])[0])
            if pages is not None:
                plist = pages.get(state, [[]])
                feats = plist[cursor] if cursor < len(plist) else []
            else:
                feats = stations.get(state, []) if cursor == 0 else []
            nxt = f"https://{HOST}/stations?state={state}&limit=500&cursor={cursor + 1}"
            return httpx.Response(200, json={"type": "FeatureCollection", "features": feats,
                                             "pagination": {"next": nxt}})
        m = re.fullmatch(r"/stations/([A-Z0-9]+)/observations/latest", path)
        if m:
            body = obs.get(m.group(1), 404)
            if isinstance(body, int):
                return httpx.Response(body, json={"status": body})
            return httpx.Response(200, json=body)
        return httpx.Response(404)

    net.route(HOST, handler)


def _obs_calls(net, sid=None):
    return [r for r in net.calls(HOST)
            if r.url.path.endswith("/observations/latest")
            and (sid is None or f"/stations/{sid}/" in r.url.path)]


# Two 1.0 deg cells: (30, -97) and (32, -96). Stations A/B/C step away from the
# first cell; D sits by the second.
CELL_LATS = [30.02, 29.95, 32.1]
CELL_LONS = [-97.03, -96.98, -96.05]
TX_STATIONS = [
    _feature("KAAA", 30.05, -97.0),
    _feature("KBBB", 30.30, -97.0, provider="OTHER-MTR"),
    _feature("KCCC", 30.60, -97.0, provider="NonFedAWOS"),
    _feature("KDDD", 32.00, -96.10, provider="ASOS"),
]


# ── Unit conversions and QC ───────────────────────────────────────────────

def test_parse_converts_wmo_units_to_display_units():
    o = nws.parse_observation(_obs("KAAA", t=29.0, rh=58.347, slp=101080.0, alt=101049.84, wind=20.376))
    f = o["fields"]
    assert f["temperature"] == 84.2          # 29 degC
    assert f["humidity"] == 58.3
    assert f["pressure"] == 1010.8           # sea-level pressure wins over the altimeter
    assert f["wind_speed"] == 12.7           # 20.376 km/h
    assert o["station_id"] == "KAAA" and o["timestamp"].tzinfo is not None


def test_parse_other_units_and_altimeter_fallback():
    o = nws.parse_observation(_obs("KAAA", t=68.0, t_unit="degF", wind=5.0, wind_unit="m_s-1",
                                   slp=None, alt=1013.2, p_unit="hPa"))
    f = o["fields"]
    assert f["temperature"] == 68.0
    assert f["wind_speed"] == 11.2           # 5 m/s
    assert f["pressure"] == 1013.2           # no SLP: altimeter setting (sea-level reduced)
    k = nws.parse_observation(_obs("KAAA", t=293.15, t_unit="K", wind=10.0, wind_unit="kn"))
    assert k["fields"]["temperature"] == 68.0 and k["fields"]["wind_speed"] == 11.5


def test_altimeter_is_reduced_to_sea_level_with_elevation():
    # Live KAMA 2026-11-03 00:53Z: A2978 (100850 Pa), 15.6 degC, 1099 m, METAR
    # SLP047 (1004.7 hPa). The raw altimeter reads 3.8 hPa high; the reduction
    # lands within 0.5 hPa of the reported SLP.
    o = nws.parse_observation(_obs("KAMA", t=15.6, slp=None, alt=100850.0, elev=1099))
    assert abs(o["fields"]["pressure"] - 1004.7) < 0.5
    # When SLP is present it still wins.
    o = nws.parse_observation(_obs("KAMA", t=15.6, slp=100470.0, alt=100850.0, elev=1099))
    assert o["fields"]["pressure"] == 1004.7
    # Near sea level the reduction barely moves the altimeter setting.
    o = nws.parse_observation(_obs("KGLS", t=25.0, slp=None, alt=101200.0, elev=2))
    assert abs(o["fields"]["pressure"] - 1012.0) < 0.6
    # No temperature: the altimeter setting as is (never dropped).
    o = nws.parse_observation(_obs("KAMA", t=None, slp=None, alt=100850.0, elev=1099))
    assert o["fields"]["pressure"] == 1008.5


def test_parse_drops_nulls_rejected_qc_and_implausible_values():
    o = nws.parse_observation(_obs("KAAA", t=None, rh=40.0, wind=12.0,
                                   qc={"relativeHumidity": "X", "windSpeed": "Q"}))
    assert o["fields"] == {"temperature": None, "humidity": None,
                           "pressure": 1013.2, "wind_speed": None}
    bad = nws.parse_observation(_obs("KAAA", t=80.0, rh=100.4, alt=50000.0))   # 176 degF, 500 hPa
    assert bad["fields"]["temperature"] is None
    assert bad["fields"]["humidity"] == 100.0   # rounding a hair over 100 is clipped
    assert bad["fields"]["pressure"] is None
    weird = nws.parse_observation(_obs("KAAA", t=20.0, t_unit="furlongs"))
    assert weird["fields"]["temperature"] is None
    assert nws.parse_observation({}) is None
    assert nws.parse_observation({"properties": {"timestamp": "not a time"}}) is None
    assert nws.parse_observation(None) is None


# ── Station catalog ──────────────────────────────────────────────────────

def test_station_filter_rule():
    keep = [_feature("KAUS", 30.19, -97.67, "ASOS-HFM"), _feature("KT20", 29.53, -97.46, "NonFedAWOS"),
            _feature("KATT", 30.32, -97.76, "OTHER-MTR"), _feature("KELP", 31.81, -106.38, "ASOS")]
    drop = [_feature("001CE", 29.90, -95.06, "MesoWest"),          # not an ICAO id
            _feature("KOCH", 31.58, -94.71, ""),                    # no provider: no data
            _feature("COOPFSM", 35.33, -94.37, None),
            _feature("KXYZ", 30.0, -97.0, "LCRA"),                  # K id, non-METAR network
            _feature("KDEN", 39.86, -104.67, "ASOS-HFM"),           # outside the bbox
            _feature("KAB", 30.0, -97.0, "ASOS"),                   # too short
            {"properties": {"stationIdentifier": "KAUS", "provider": "ASOS"}}]  # no geometry
    assert all(nws.is_reliable_station(f) for f in keep)
    assert not any(nws.is_reliable_station(f) for f in drop)


def test_fetch_stations_follows_pagination_and_filters(net):
    pages = {"TX": [[_feature("KAUS", 30.19, -97.67), _feature("001CE", 29.9, -95.06, "MesoWest")],
                    [_feature("KDFW", 32.90, -97.04)],
                    [_feature("KELP", 31.81, -106.38, "ASOS")]],
             "NM": [[_feature("KROW", 33.30, -104.53), _feature("KABQ", 35.04, -106.6)]],
             "OK": [[_feature("KLAW", 34.57, -98.42)]],
             "AR": [[_feature("KTXK", 33.45, -93.99)]],
             "LA": [[_feature("KSHV", 32.45, -93.82), _feature("KMSY", 29.99, -90.25)]]}  # KMSY: east of the bbox
    _route(net, pages=pages)
    st = asyncio.run(nws.fetch_stations())
    ids = sorted(s["id"] for s in st)
    assert ids == ["KABQ", "KAUS", "KDFW", "KELP", "KLAW", "KROW", "KSHV", "KTXK"]
    reqs = [r for r in net.calls(HOST) if r.url.path == "/stations"]
    states = [parse_qs(r.url.query.decode())["state"][0] for r in reqs]
    assert states.count("TX") == 4   # 3 pages + the empty one that ends the cursor walk
    assert sorted(set(states)) == sorted(nws.STATES)
    assert all("chebrolusaketh@gmail.com" in r.headers["user-agent"] for r in reqs)
    # Cached: a second call sends nothing.
    n = len(net.calls(HOST))
    assert asyncio.run(nws.fetch_stations()) == st
    assert len(net.calls(HOST)) == n


def test_fetch_stations_snapshot_survives_restart(net):
    _route(net, stations={"TX": TX_STATIONS})
    first = asyncio.run(nws.fetch_stations())
    assert len(first) == 4
    nws._reset_state()   # a restart: memory gone, snapshot file kept
    net.reset()          # and the network is down
    again = asyncio.run(nws.fetch_stations())
    assert [s["id"] for s in again] == [s["id"] for s in first]
    assert net.calls(HOST) == []


def test_fetch_stations_failure_never_raises_and_retries_5xx_once(net):
    net.route(HOST, lambda r: httpx.Response(503))
    assert asyncio.run(nws.fetch_stations()) == []
    per_state = [parse_qs(r.url.query.decode())["state"][0] for r in net.calls(HOST)]
    assert sorted(per_state) == sorted(nws.STATES * 2)   # one retry each
    # Backoff: an immediate second call does not hammer the API.
    n = len(net.calls(HOST))
    assert asyncio.run(nws.fetch_stations()) == []
    assert len(net.calls(HOST)) == n


def test_partial_catalog_failure_keeps_previous_list(net):
    _route(net, stations={"TX": TX_STATIONS})
    asyncio.run(nws.fetch_stations())
    nws._stations["fetched_at"] = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()

    def flaky(request):
        if "state=NM" in str(request.url):
            return httpx.Response(500)
        return httpx.Response(200, json={"features": [], "pagination": {}})
    net.route(HOST, flaky)
    st = asyncio.run(nws.fetch_stations())
    assert len(st) == 4   # the full old catalog, not today's partial one


# ── Conditions grid ──────────────────────────────────────────────────────

def _grid():
    return asyncio.run(nws.fetch_conditions_grid(CELL_LATS, CELL_LONS))


def test_grid_shape_and_one_call_per_cell_when_healthy(net):
    _route(net, stations={"TX": TX_STATIONS},
           obs={"KAAA": _obs("KAAA", t=25.0), "KDDD": _obs("KDDD", t=30.0)})
    g = _grid()
    assert g["usable"] and g["source"] == "nws" and g["n_cells"] == 2
    k1 = airquality.conditions_cell_key(30.0, -97.0)
    k2 = airquality.conditions_cell_key(32.0, -96.0)
    assert set(g["by_key"]) == {k1, k2}
    c = g["by_key"][k1]
    assert c["temperature"] == 77.0 and c["humidity"] == 50.0
    assert c["pressure"] == 1013.2 and c["wind_speed"] == 11.5
    assert c["station"] == "KAAA" and c["station_name"] == "KAAA Field" and c["source"] == "nws"
    datetime.fromisoformat(c["observed_at"])
    assert g["by_key"][k2]["temperature"] == 86.0
    assert len(_obs_calls(net)) == 2 and g["n_station_calls"] == 2
    datetime.fromisoformat(g["fetched_at"])


def test_stale_nearest_station_falls_back_to_next(net):
    _route(net, stations={"TX": TX_STATIONS},
           obs={"KAAA": _obs("KAAA", age_min=4 * 60, t=10.0),   # 4 h old
                "KBBB": _obs("KBBB", t=21.0),
                "KDDD": _obs("KDDD")})
    c = _grid()["by_key"][airquality.conditions_cell_key(30.0, -97.0)]
    assert c["station"] == "KBBB" and c["temperature"] == 69.8
    assert _obs_calls(net, "KCCC") == []   # resolved at rank 2, rank 3 never asked


def test_null_field_comes_from_next_station(net):
    _route(net, stations={"TX": TX_STATIONS},
           obs={"KAAA": _obs("KAAA", t=25.0, rh=None, wind=None),
                "KBBB": _obs("KBBB", t=15.0, rh=None, wind=9.26),
                "KCCC": _obs("KCCC", t=5.0, rh=77.0, wind=40.0),
                "KDDD": _obs("KDDD")})
    c = _grid()["by_key"][airquality.conditions_cell_key(30.0, -97.0)]
    assert c["temperature"] == 77.0 and c["station"] == "KAAA"   # nearest wins per field
    assert c["wind_speed"] == 5.8                                 # from KBBB
    assert c["humidity"] == 77.0                                  # from KCCC


def test_cell_missing_a_field_everywhere_is_left_out(net):
    _route(net, stations={"TX": TX_STATIONS},
           obs={"KAAA": _obs("KAAA", wind=None), "KBBB": 404, "KCCC": 500, "KDDD": _obs("KDDD")})
    g = _grid()
    assert list(g["by_key"]) == [airquality.conditions_cell_key(32.0, -96.0)]
    assert len(_obs_calls(net, "KCCC")) == 2   # 5xx retried once
    assert len(_obs_calls(net, "KBBB")) == 1   # 4xx not retried


def test_missing_field_borrowed_from_already_fetched_station(net):
    # Cell (30,-97)'s 3 nearest lack pressure; KEEE (about 80 km away) was
    # already fetched for its own cell, so it fills pressure with no extra call.
    stations = TX_STATIONS[:3] + [_feature("KEEE", 30.75, -97.0)]
    lats, lons = [30.02, 31.0], [-97.03, -97.0]
    _route(net, stations={"TX": stations},
           obs={"KAAA": _obs("KAAA", alt=None), "KBBB": _obs("KBBB", alt=None),
                "KCCC": _obs("KCCC", alt=None), "KEEE": _obs("KEEE", alt=100900.0)})
    g = asyncio.run(nws.fetch_conditions_grid(lats, lons))
    c = g["by_key"][airquality.conditions_cell_key(30.0, -97.0)]
    assert c["station"] == "KAAA" and c["pressure"] == 1009.0
    assert len(_obs_calls(net)) == 4


def test_grid_feeds_main_get_display_weather(net, monkeypatch):
    import backend.main as main
    _route(net, stations={"TX": TX_STATIONS},
           obs={"KAAA": _obs("KAAA", t=25.0), "KDDD": _obs("KDDD", t=30.0)})
    g = _grid()
    monkeypatch.setattr(main, "_conditions_cache",
                        {"data": g, "expires": datetime.now(timezone.utc) + timedelta(hours=1)})
    w = asyncio.run(main.get_display_weather(30.2, -97.1, allow_fetch=False))
    assert w["temperature"] == 77.0 and w["station"] == "KAAA" and w["source"] == "nws"
    assert set(main.DISPLAY_WEATHER_FALLBACK) <= set(w)
    # A point whose own cell is absent uses the nearest occupied cell.
    far = asyncio.run(main.get_display_weather(33.4, -95.6, allow_fetch=False))
    assert far["temperature"] == 86.0


def test_grid_never_raises(net):
    # Network down entirely.
    g = _grid()
    assert g["usable"] is False and g["by_key"] == {} and g["source"] == "nws"
    # Garbage bodies.
    net.route(HOST, lambda r: httpx.Response(200, content=b"<html>oops</html>"))
    nws._reset_state()
    assert _grid()["usable"] is False
    # Catalog fine, every observation broken in a different way.
    nws._reset_state()
    _route(net, stations={"TX": TX_STATIONS},
           obs={"KAAA": {"properties": None}, "KBBB": {"properties": {"timestamp": 5}},
                "KCCC": 503, "KDDD": {"type": "Feature"}})
    g = _grid()
    assert g["usable"] is False and g["by_key"] == {}
    # No tracts at all.
    assert asyncio.run(nws.fetch_conditions_grid([], []))["n_cells"] == 0


def test_grid_gives_up_when_no_station_answers(net):
    # Catalog fine, every observation 503: round 1 fails completely, so rounds
    # 2 and 3 (KBBB, KCCC) are never asked; each round-1 call retried once.
    _route(net, stations={"TX": TX_STATIONS}, obs={"KAAA": 503, "KDDD": 503})
    g = _grid()
    assert g["usable"] is False
    assert sorted(r.url.path.split("/")[2] for r in _obs_calls(net)) == ["KAAA", "KAAA", "KDDD", "KDDD"]


def test_grid_and_point_are_time_bounded(net, monkeypatch):
    _route(net, stations={"TX": TX_STATIONS})
    asyncio.run(nws.fetch_stations())

    async def hang(request):
        await asyncio.sleep(5)
        return httpx.Response(200, json=_obs("KAAA"))
    net.route(HOST, hang)
    monkeypatch.setattr(nws, "GRID_TIMEOUT_S", 0.2)
    monkeypatch.setattr(nws, "POINT_TIMEOUT_S", 0.2)
    t0 = time.monotonic()
    g = _grid()
    assert g["usable"] is False and g["by_key"] == {} and g["source"] == "nws"
    assert asyncio.run(nws.fetch_point_current(30.06, -97.01)) is None
    assert time.monotonic() - t0 < 2.0   # two 0.2 s budgets, not 5 s hangs


def test_point_timeout_trips_a_backoff(net, monkeypatch):
    """A timed-out point lookup never reaches the per-station negative cache,
    so the module backs off point lookups instead: during an NWS brownout one
    click waits POINT_TIMEOUT_S and the next ones return None at once."""
    _route(net, stations={"TX": TX_STATIONS})
    asyncio.run(nws.fetch_stations())

    async def hang(request):
        await asyncio.sleep(5)
        return httpx.Response(200, json=_obs("KAAA"))
    net.route(HOST, hang)
    monkeypatch.setattr(nws, "POINT_TIMEOUT_S", 0.3)
    t0 = time.monotonic()
    assert asyncio.run(nws.fetch_point_current(30.06, -97.01)) is None
    assert time.monotonic() - t0 >= 0.3
    calls = len(_obs_calls(net))
    t1 = time.monotonic()
    assert asyncio.run(nws.fetch_point_current(29.5, -98.0)) is None
    assert time.monotonic() - t1 < 0.1 and len(_obs_calls(net)) == calls

    # Once the backoff lapses, lookups resume.
    nws._point_backoff["until"] = 0.0
    _route(net, stations={"TX": TX_STATIONS}, obs={"KAAA": _obs("KAAA")})
    assert asyncio.run(nws.fetch_point_current(30.06, -97.01))["station"] == "KAAA"


# ── Exact point (clicked tract) ──────────────────────────────────────────

def test_point_current_uses_nearest_and_shared_cache(net):
    _route(net, stations={"TX": TX_STATIONS},
           obs={"KAAA": _obs("KAAA", t=25.0), "KDDD": _obs("KDDD", t=30.0)})
    w = asyncio.run(nws.fetch_point_current(30.06, -97.01))
    assert w["temperature"] == 77.0 and w["station"] == "KAAA" and w["source"] == "nws"
    assert set(w) == {"temperature", "humidity", "pressure", "wind_speed",
                      "source", "station", "station_name", "observed_at"}
    n = len(_obs_calls(net))
    assert n == 1
    # Same station again (another click, then the grid): served from cache.
    asyncio.run(nws.fetch_point_current(30.04, -96.99))
    assert len(_obs_calls(net)) == n
    _grid()
    assert len(_obs_calls(net, "KAAA")) == 1


def test_point_current_falls_back_and_returns_none(net):
    _route(net, stations={"TX": TX_STATIONS},
           obs={"KAAA": _obs("KAAA", age_min=200), "KBBB": _obs("KBBB", t=18.0)})
    w = asyncio.run(nws.fetch_point_current(30.06, -97.01))
    assert w["station"] == "KBBB" and w["temperature"] == 64.4
    # Far from every station (> 150 km): None, so the caller uses the grid.
    assert asyncio.run(nws.fetch_point_current(26.0, -106.0)) is None


def test_point_current_never_raises(net):
    assert asyncio.run(nws.fetch_point_current(30.0, -97.0)) is None   # no catalog, no network
    _route(net, stations={"TX": TX_STATIONS}, obs={})
    nws._reset_state()
    assert asyncio.run(nws.fetch_point_current(30.0, -97.0)) is None   # every station 404
    assert asyncio.run(nws.fetch_point_current(float("nan"), -97.0)) is None

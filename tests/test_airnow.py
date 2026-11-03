"""EPA AirNow monitor feed: file parsing, the incremental hourly store, the
24-hour window and 75% rule, late-report re-fetch and snapshots. All network
goes through the offline router (files.airnowtech.org)."""
import asyncio
import hashlib
import json
from datetime import datetime, timedelta, timezone

import httpx

from backend import airnow

HOST = airnow.AIRNOW_HOST
UTC = timezone.utc
# 19:20 UTC: the 18:00 hour is the newest attemptable one.
NOW = datetime(2026, 10, 29, 19, 20, tzinfo=UTC)
NEWEST = datetime(2026, 10, 29, 18, tzinfo=UTC)

SITES_HEADER = ("StationID|AQSID|FullAQSID|Parameter|MonitorType|SiteCode|SiteName|Status|"
                "AgencyID|AgencyName|EPARegion|Latitude|Longitude|Elevation|GMTOffset|"
                "CountryFIPS|CBSA_ID|CBSA_Name|StateAQSCode|StateAbbreviation|"
                "CountyAQSCode|CountyName")


def _site_row(aqsid, name, lat, lon, param="PM2.5", status="Active", station=None, full=None):
    return "|".join([
        station or aqsid, aqsid, full or f"840{aqsid}", param, "Permanent", aqsid[-4:], name,
        status, "TX1", "Texas Commission on Environmental Quality", "R6", str(lat), str(lon),
        "10.0", "-6.00", "US", "", "", "48", "TX", "201", "HARRIS"])


# Twelve Texas monitors: S00..S11 (AQSIDs 480000100..480000111).
IDS = [f"4800001{i:02d}" for i in range(12)]


def _sites_file(ids=IDS, extra=()):
    rows = [SITES_HEADER]
    for i, a in enumerate(ids):
        rows.append(_site_row(a, f"Site {i:02d}", 29.0 + i * 0.1, -97.0 + i * 0.1))
    rows.extend(extra)
    return ("\r\n".join(rows) + "\r\n").encode()


def _hour_rows(hour, values, noise=True):
    """CRLF hourly file body for {id: value}, plus rows we must ignore."""
    d, t = f"{hour:%m/%d/%y}", f"{hour:%H}:00"
    rows = [f"{d}|{t}|{a}|Name {a}|-6|PM2.5|UG/M3|{v}|TCEQ" for a, v in values.items()]
    if noise:
        rows += [f"{d}|{t}|{a}|Name {a}|-6|OZONE|PPB|31|TCEQ" for a in list(values)[:3]]
        rows += [f"{d}|{t}|060410001|San Rafael|-8|PM2.5|UG/M3|99|BAAQMD"]
    return ("\r\n".join(rows) + "\r\n").encode()


def _value(site_idx, hour):
    """A deterministic value with some hour-to-hour variation."""
    return round(site_idx + 1 + (hour.hour % 4) * 0.25, 2)


class FakeAirNow:
    """AirNow's file layout on the offline router: the site file, hourly
    files keyed by hour ({hour: {id: value}}), ETag / If-None-Match -> 304,
    anything else 404 (with S3's NoSuchKey XML body, as the real server)."""

    def __init__(self, net, sites=None):
        self.sites = sites if sites is not None else _sites_file()
        self.hours = {}
        self.sites_status = 200
        net.route(HOST, self.handle)
        self.net = net

    def body(self, hour):
        return _hour_rows(hour, self.hours[hour])

    def handle(self, request):
        path = request.url.path
        if path == "/airnow/today/Monitoring_Site_Locations_V2.dat":
            return httpx.Response(self.sites_status, content=self.sites)
        for hour in self.hours:
            if path == httpx.URL(airnow.hour_url(hour)).path:
                body = self.body(hour)
                etag = '"' + hashlib.md5(body).hexdigest() + '"'
                if request.headers.get("If-None-Match") == etag:
                    return httpx.Response(304, headers={"ETag": etag})
                return httpx.Response(200, content=body, headers={"ETag": etag})
        return httpx.Response(404, content=b'<?xml version="1.0"?><Error><Code>NoSuchKey</Code></Error>')

    def hour_calls(self):
        return [r for r in self.net.calls(HOST) if "HourlyData_" in r.url.path]

    def requested_hours(self):
        return sorted(r.url.path.rsplit("_", 1)[1][:10] for r in self.hour_calls())

    def site_calls(self):
        return [r for r in self.net.calls(HOST) if r.url.path.endswith("V2.dat")]


def _full_day(fake, newest_published=NEWEST - timedelta(hours=1)):
    """Hours from the oldest stored hour to newest_published: S00..S09 report
    every hour, S10 misses 6 window hours (18 left), S11 misses 7 (17 left)."""
    end = NEWEST - timedelta(hours=1)
    window = {end - timedelta(hours=i) for i in range(24)}
    s10_gaps = sorted(window)[2:8]
    s11_gaps = sorted(window)[2:9]
    h = NEWEST - timedelta(hours=airnow.STORE_HOURS - 1)
    while h <= newest_published:
        vals = {IDS[i]: _value(i, h) for i in range(10)}
        if h not in s10_gaps:
            vals[IDS[10]] = _value(10, h)
        if h not in s11_gaps:
            vals[IDS[11]] = _value(11, h)
        fake.hours[h] = vals
        h += timedelta(hours=1)
    return sorted(window)


def _run(coro):
    return asyncio.run(coro)


# ── Parsing ──────────────────────────────────────────────────────────────────

def test_hour_url_is_utc_hour_beginning_layout():
    assert airnow.hour_url(datetime(2026, 10, 29, 5, tzinfo=UTC)) == (
        "https://files.airnowtech.org/airnow/2026/20261029/HourlyData_2026102905.dat")
    targets = airnow.target_hours(NOW)
    assert len(targets) == airnow.STORE_HOURS == 30
    assert targets[0] == NEWEST and targets[-1] == NEWEST - timedelta(hours=29)
    # Each hour lives in its own UTC day (and year) folder across midnight.
    urls = [airnow.hour_url(h) for h in airnow.target_hours(datetime(2027, 1, 1, 1, 5, tzinfo=UTC))]
    assert urls[0].endswith("/2027/20270101/HourlyData_2027010100.dat")
    assert urls[1].endswith("/2026/20261231/HourlyData_2026123123.dat")
    # A naive or non-UTC `now` is read as / converted to UTC.
    cdt = timezone(timedelta(hours=-5))
    assert airnow.target_hours(datetime(2026, 10, 29, 14, 20, tzinfo=cdt))[0] == NEWEST


def test_parse_sites_filters_parameter_status_and_bbox():
    extra = [
        _site_row("480000900", "Ozone only", 30.0, -97.0, param="O3"),
        _site_row("480000901", "Retired", 30.0, -97.0, status="Inactive"),
        _site_row("060410001", "San Rafael", 37.97, -122.52),
        _site_row("350450001", "Farmington", 36.7, -108.2),          # north and west of bbox
        _site_row("480000902", "Bad coords", "N/A", -97.0),
        _site_row("480000903", "Short row", 30.0, -97.0)[:40],
        _site_row("MMFS11042", "Smoke monitor", 36.3, -106.4, full="840MMFS11042",
                  station="840MMFS11042"),
        _site_row("480000100", "Duplicate of S00", 30.0, -97.0),
    ]
    sites = airnow.parse_sites(_sites_file(extra=extra))
    assert set(sites) == set(IDS) | {"MMFS11042"}
    assert sites["480000100"]["name"] == "Site 00"   # first row wins
    assert sites["480000100"]["aliases"] == ["840480000100"]
    assert sites["MMFS11042"]["aliases"] == ["840MMFS11042"]
    s = sites["480000105"]
    assert (s["lat"], s["lon"]) == (29.5, -96.5)

    index = airnow.build_alias_index(sites)
    assert index["840480000100"] == "480000100"
    assert index["840MMFS11042"] == "MMFS11042"


def test_parse_sites_rejects_unknown_header():
    try:
        airnow.parse_sites(b"foo|bar\r\n1|2\r\n")
    except ValueError as e:
        assert "AQSID" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_parse_hourly_ids_values_and_malformed_rows():
    sites = airnow.parse_sites(_sites_file(
        extra=[_site_row("MMFS10328", "Unit328", 35.6, -105.9, station="ST10328")]))
    index = airnow.build_alias_index(sites)
    h = datetime(2026, 10, 29, 17, tzinfo=UTC)
    lines = [
        "10/29/26|17:00|480000100|Site 00|-6|PM2.5|UG/M3|7.4|TCEQ",
        "10/29/26|17:00|840480000101|Site 01|-6|PM2.5|UG/M3|12|TCEQ",     # FullAQSID
        "10/29/26|17:00|ST10328|Unit328|-7|PM2.5|UG/M3|3.0|USFS",          # StationID
        "10/29/26|17:00|480000102|Site 02|-6|PM2.5|UG/M3|-3.2|TCEQ",       # valid negative
        "10/29/26|17:00|480000103|Site 03|-6|PM2.5|UG/M3|-45|TCEQ",        # implausible
        "10/29/26|17:00|480000104|Site 04|-6|PM2.5|UG/M3|5000|TCEQ",       # implausible
        "10/29/26|17:00|480000105|Site 05|-6|PM2.5|UG/M3|N/A|TCEQ",        # non-numeric
        "10/29/26|17:00|480000106|Site 06|-6|PM2.5|PPB|4|TCEQ",            # wrong units
        "10/29/26|16:00|480000107|Site 07|-6|PM2.5|UG/M3|4|TCEQ",          # wrong hour
        "10/29/26|17:00|480000108|Site 08|-6|PM2.5|UG/M3",                 # truncated
        "10/29/26|17:00|480000100|Site 00|-6|PM2.5|UG/M3|99|TCEQ",         # duplicate
        "10/29/26|17:00|480000109|Site 09|-6|OZONE|PPB|40|TCEQ",           # other parameter
        "10/29/26|17:00|480000109|Salmon PM2.5|-6|TEMP|C|10.1|IDEQ",       # name says PM2.5
        "10/29/26|17:00|060410001|San Rafael|-8|PM2.5|UG/M3|9|BAAQMD",     # not our site
        "garbage without pipes",
        "",
        "10/29/26|17:00|480000110|Site 10|-6|PM2.5|UG/M3|nan|TCEQ",        # NaN
        "10/29/26|17:00|480000111|Site 11|-6| PM2.5 |UG/M3| 8.25 |TCEQ",   # padded fields
    ]
    values, stats = airnow.parse_hourly("\r\n".join(lines).encode(), h, index)
    assert values == {"480000100": 7.4, "480000101": 12.0, "MMFS10328": 3.0,
                      "480000102": -3.2, "480000111": 8.25}
    assert stats["duplicate"] == 1
    assert stats["wrong_hour"] == 1
    assert stats["malformed"] == 1
    assert stats["bad_value"] == 5


def test_parse_hourly_date_without_leading_zero():
    index = airnow.build_alias_index(airnow.parse_sites(_sites_file()))
    h = datetime(2026, 5, 3, 7, tzinfo=UTC)
    values, _ = airnow.parse_hourly(b"5/3/26|7:00|480000100|S|-6|PM2.5|UG/M3|6|TCEQ\n", h, index)
    assert values == {"480000100": 6.0}


# ── Means: window selection and the 75% rule ─────────────────────────────────

def _sites_dict(n=12):
    return {a: {"id": a, "name": f"Site {i:02d}", "lat": 30.0, "lon": -97.0, "aliases": []}
            for i, a in enumerate(IDS[:n])}


def _hours_dict(spec):
    """{hour: n_sites_reporting} -> store hours with the first n sites at 5.0."""
    return {h: {"fetched_at": NOW, "etag": None, "values": {a: 5.0 for a in IDS[:n]}}
            for h, n in spec.items()}


def test_window_ends_at_newest_hour_with_half_the_sites():
    sites = _sites_dict()
    spec = {NEWEST - timedelta(hours=i): 12 for i in range(airnow.STORE_HOURS)}
    spec[NEWEST] = 5                                   # under half of 12: skipped
    spec[NEWEST - timedelta(hours=1)] = 6              # exactly half: qualifies
    m = airnow.compute_means(sites, _hours_dict(spec), NOW)
    assert m["window_end"] == (NEWEST - timedelta(hours=1)).isoformat()
    assert m["window_start"] == (NEWEST - timedelta(hours=24)).isoformat()
    assert m["usable"] is True and m["reason"] is None
    hours = {x["id"]: x["hours"] for x in m["monitors"]}
    assert hours[IDS[0]] == 24 and hours[IDS[11]] == 23

    # Sites that never report are not in the denominator: 12 listed, 6 report.
    spec2 = {h: 6 for h in spec}
    spec2[NEWEST] = 3
    m2 = airnow.compute_means(sites, _hours_dict(spec2), NOW)
    assert m2["window_end"] == NEWEST.isoformat()


def test_stale_store_is_not_usable():
    sites = _sites_dict()
    # Nothing newer than 7 h before the newest attemptable hour.
    spec = {NEWEST - timedelta(hours=i): 12 for i in range(7, airnow.STORE_HOURS)}
    m = airnow.compute_means(sites, _hours_dict(spec), NOW)
    assert m["usable"] is False and m["monitors"] == [] and m["window_end"] is None
    assert "at least half" in m["reason"]
    # An hour older than the stored span is ignored outright.
    old = {NEWEST - timedelta(hours=40): {"fetched_at": NOW, "etag": None,
                                          "values": {a: 5.0 for a in IDS}}}
    assert airnow.compute_means(sites, old, NOW)["reason"].startswith("no AirNow hourly data")


def test_means_need_ten_monitors_with_eighteen_hours():
    sites = _sites_dict()
    spec = {NEWEST - timedelta(hours=i): 12 for i in range(airnow.STORE_HOURS)}
    hours = _hours_dict(spec)
    window = [NEWEST - timedelta(hours=i) for i in range(24)]
    # Gaps inside the window (the newest hour keeps everyone, so the window
    # still ends there): 9 sites keep 18 hours, 3 keep 17.
    for i, a in enumerate(IDS):
        drop = 6 if i < 9 else 7
        for h in window[3:3 + drop]:
            hours[h]["values"].pop(a)
    m = airnow.compute_means(sites, hours, NOW)
    assert m["window_end"] == NEWEST.isoformat()
    assert len(m["monitors"]) == 9 and {x["hours"] for x in m["monitors"]} == {18}
    assert m["usable"] is False and "only 9 monitors" in m["reason"]


# ── Refresh against the fake server ──────────────────────────────────────────

def test_cold_start_404_newest_and_24h_means(net):
    fake = FakeAirNow(net)
    window = _full_day(fake)                          # 18:00 not published yet
    store = airnow.AirNowStore()
    s = _run(store.refresh(NOW))
    assert s["ok"] is True and s["errors"] == []
    assert s["sites_refreshed"] is True and s["sites"] == 12
    assert s["not_published"] == [NEWEST.isoformat()]
    assert len(s["fetched"]) == 29 and s["hours_stored"] == 29
    assert len(fake.hour_calls()) == 30 and len(fake.site_calls()) == 1
    assert s["bytes"] > 0 and s["rows_skipped"] == 0

    m = store.latest_means(NOW)
    assert m["usable"] is True
    assert m["window_end"] == "2026-10-29T17:00:00+00:00"
    assert m["window_start"] == "2026-10-28T18:00:00+00:00"
    by_id = {x["id"]: x for x in m["monitors"]}
    assert set(by_id) == set(IDS[:11])                # S11 has only 17 hours
    assert by_id[IDS[10]]["hours"] == 18 and by_id[IDS[0]]["hours"] == 24
    for i in (0, 4, 10):
        hrs = [h for h in window if IDS[i] in fake.hours[h]]
        expect = round(sum(_value(i, h) for h in hrs) / len(hrs), 2)
        assert by_id[IDS[i]]["pm25"] == expect
    x = by_id[IDS[3]]
    assert set(x) == {"id", "name", "lat", "lon", "pm25", "hours"}
    assert x["name"] == "Site 03" and x["lat"] == 29.3 and x["lon"] == -96.7


def test_newest_hour_retried_after_404(net):
    fake = FakeAirNow(net)
    _full_day(fake)
    store = airnow.AirNowStore()
    _run(store.refresh(NOW))
    net.requests.clear()

    # Five minutes later: the 404 is not retried yet, nothing else is due.
    s = _run(store.refresh(NOW + timedelta(minutes=5)))
    assert fake.hour_calls() == [] and s["fetched"] == [] and s["changed"] is False

    fake.hours[NEWEST] = {a: 4.0 for a in IDS}
    s = _run(store.refresh(NOW + timedelta(minutes=15)))
    assert s["fetched"] == [NEWEST.isoformat()]
    assert fake.requested_hours() == ["2026102918"]
    assert store.latest_means(NOW + timedelta(minutes=15))["window_end"] == NEWEST.isoformat()


def test_partial_newest_hour_is_skipped_then_refetched(net):
    fake = FakeAirNow(net)
    _full_day(fake)
    fake.hours[NEWEST] = {a: 4.0 for a in IDS[:3]}    # 3 of 12 so far
    store = airnow.AirNowStore()
    s = _run(store.refresh(NOW))
    assert NEWEST.isoformat() in s["fetched"]
    m = store.latest_means(NOW)
    assert m["window_end"] == (NEWEST - timedelta(hours=1)).isoformat()

    # Late reports arrive; 31 min later the provisional hour is re-fetched.
    fake.hours[NEWEST] = {a: 4.0 for a in IDS}
    net.requests.clear()
    later = NOW + timedelta(minutes=31)
    s = _run(store.refresh(later))
    assert NEWEST.isoformat() in s["refetched"]
    assert store.latest_means(later)["window_end"] == NEWEST.isoformat()


def _provisional_after_cold_start():
    """Hours a cold start at NOW leaves provisional: those that ended less
    than LATE_REPORT_H before NOW (17:00, 16:00 and 15:00 with 4 h)."""
    return {NEWEST - timedelta(hours=i) for i in range(1, airnow.LATE_REPORT_H)}


def _labels(hours):
    return {f"{h:%Y%m%d%H}" for h in hours}


def test_late_report_refetch_policy_and_etag(net):
    fake = FakeAirNow(net)
    _full_day(fake)
    store = airnow.AirNowStore()
    _run(store.refresh(NOW))                          # 19:20
    h17 = NEWEST - timedelta(hours=1)
    provisional = _provisional_after_cold_start()
    assert h17 in provisional and NEWEST - timedelta(hours=airnow.LATE_REPORT_H) not in provisional
    # 29 min on: only the 404'd newest hour is due (its 10 min retry).
    assert store.plan(NOW + timedelta(minutes=29)) == [(NEWEST, "new")]
    due = store.plan(NOW + timedelta(minutes=30))
    assert {h for h, kind in due if kind == "refetch"} == provisional

    # Unchanged files: conditional GETs answered 304.
    net.requests.clear()
    t1 = NOW + timedelta(minutes=31)                  # 19:51
    s = _run(store.refresh(t1))
    assert set(s["unchanged"]) == {h.isoformat() for h in provisional}
    call = [r for r in fake.hour_calls() if r.url.path.endswith("2026102917.dat")][0]
    assert call.headers.get("If-None-Match")
    assert set(fake.requested_hours()) == _labels(provisional) | {"2026102918"}

    # Still provisional (fetched 19:51, before 17:00 + 1 h + LATE_REPORT_H);
    # S11's value is revised.
    fake.hours[h17] = {**fake.hours[h17], IDS[11]: 2.5}
    net.requests.clear()
    t2 = NOW + timedelta(minutes=70)                  # 20:30
    s = _run(store.refresh(t2))
    assert h17.isoformat() in s["refetched"]
    assert store.hours[h17]["values"][IDS[11]] == 2.5

    # Re-fetched every 30 min until one fetch lands LATE_REPORT_H after the
    # hour ended; after that it is final and never fetched again.
    final_at = h17 + timedelta(hours=1 + airnow.LATE_REPORT_H)
    t = t2
    while t < final_at:
        t += timedelta(minutes=15)
        _run(store.refresh(t))
    assert store.hours[h17]["fetched_at"] >= final_at
    net.requests.clear()
    for i in range(1, 9):
        _run(store.refresh(t + timedelta(minutes=15 * i)))
    assert "2026102917" not in fake.requested_hours()


def test_hour_completed_late_is_not_frozen_partial(net):
    """Real AirNow (2026-11-03): an hour's file appears soon after the hour
    without most TCEQ monitors, which join it in one batch, usually ~1 h 20
    min after the hour ends but for some hours past +3 h. Here every hour's
    batch comes at +2.5 h (the old 2 h finality froze all of them partial).
    On a 15-minute cycle every stored hour must end up complete."""
    fake = FakeAirNow(net)
    clock = {}
    early = {a: 6.0 for a in IDS[:5]}
    full = {a: 6.0 for a in IDS}

    def handler(request):
        if "HourlyData_" in request.url.path:
            stamp = request.url.path.rsplit("_", 1)[1][:10]
            hour = datetime.strptime(stamp, "%Y%m%d%H").replace(tzinfo=UTC)
            age = clock["now"] - (hour + timedelta(hours=1))
            if age < timedelta(minutes=31):
                fake.hours.pop(hour, None)
            else:
                fake.hours[hour] = full if age >= timedelta(minutes=151) else early
        return fake.handle(request)
    net.route(HOST, handler)

    store = airnow.AirNowStore()
    t = NOW
    for _ in range(4 * airnow.STORE_HOURS):           # 30 h of 15-minute cycles
        clock["now"] = t
        _run(store.refresh(t))
        t += timedelta(minutes=15)
    now = t - timedelta(minutes=15)
    m = store.latest_means(now)
    assert m["usable"] is True and len(m["monitors"]) == 12
    assert {x["hours"] for x in m["monitors"]} == {24}
    # Only the hours still inside the 2.5 h fill-in are partial.
    partial = [h for h, r in store.hours.items() if len(r["values"]) < 12]
    assert partial and all(now - (h + timedelta(hours=1)) < timedelta(hours=3) for h in partial)


def test_sparse_final_hour_is_rechecked_hourly(net):
    fake = FakeAirNow(net)
    _full_day(fake)
    h10 = datetime(2026, 10, 29, 10, tzinfo=UTC)
    fake.hours[h10] = {a: 4.0 for a in IDS[:3]}       # an agency's data is late
    store = airnow.AirNowStore()
    _run(store.refresh(NOW))                          # 10:00 is final by age
    gap = timedelta(minutes=airnow.SPARSE_REFETCH_MIN)
    assert (h10, "refetch") not in store.plan(NOW + gap - timedelta(minutes=1))
    assert (h10, "refetch") in store.plan(NOW + gap)

    # Unchanged: a 304, then nothing until another SPARSE_REFETCH_MIN.
    s = _run(store.refresh(NOW + gap))
    assert h10.isoformat() in s["unchanged"]
    assert (h10, "refetch") not in store.plan(NOW + 2 * gap - timedelta(minutes=1))

    # The late data arrives; once complete the hour is not re-fetched again.
    fake.hours[h10] = {a: 4.0 for a in IDS}
    s = _run(store.refresh(NOW + 2 * gap))
    assert h10.isoformat() in s["refetched"] and len(store.hours[h10]["values"]) == 12
    assert (h10, "refetch") not in store.plan(NOW + timedelta(hours=5))


def test_window_moves_and_old_hours_are_pruned(net):
    fake = FakeAirNow(net)
    _full_day(fake)
    store = airnow.AirNowStore()
    _run(store.refresh(NOW))
    oldest = NEWEST - timedelta(hours=29)
    assert oldest in store.hours
    fake.hours[NEWEST] = {a: 4.0 for a in IDS}
    fake.hours[NEWEST + timedelta(hours=1)] = {a: 4.0 for a in IDS}
    net.requests.clear()
    s = _run(store.refresh(NOW + timedelta(hours=1)))
    assert oldest not in store.hours and len(store.hours) == 30
    # Only the two new hours and the provisional ones were downloaded.
    assert set(fake.requested_hours()) == (_labels(_provisional_after_cold_start())
                                           | {"2026102918", "2026102919"})
    assert s["hours_stored"] == 30


def test_site_metadata_cached_for_a_day(net):
    fake = FakeAirNow(net)
    _full_day(fake)
    store = airnow.AirNowStore()
    _run(store.refresh(NOW))
    _run(store.refresh(NOW + timedelta(hours=6)))
    assert len(fake.site_calls()) == 1
    _run(store.refresh(NOW + timedelta(hours=24, minutes=1)))
    assert len(fake.site_calls()) == 2


def test_site_file_failure_skips_hours_and_backs_off(net):
    fake = FakeAirNow(net)
    _full_day(fake)
    fake.sites_status = 500
    store = airnow.AirNowStore()
    s = _run(store.refresh(NOW))
    assert s["ok"] is False and any(e.startswith("sites:") for e in s["errors"])
    assert fake.hour_calls() == [] and store.hours == {}
    assert store.latest_means(NOW)["usable"] is False

    _run(store.refresh(NOW + timedelta(minutes=10)))
    assert len(fake.site_calls()) == 1                # retry waits SITES_RETRY_MIN
    fake.sites_status = 200
    s = _run(store.refresh(NOW + timedelta(minutes=31)))
    assert s["ok"] is True and s["sites"] == 12 and len(s["fetched"]) == 29


def test_unreachable_airnow_never_raises(net):
    store = airnow.AirNowStore()
    s = _run(store.refresh(NOW))                      # no route: connection errors
    assert s["ok"] is False and s["errors"] and s["changed"] is False
    assert store.latest_means(NOW)["usable"] is False

    # Sites reachable, hourly files not: stop after a few transport failures.
    fake = FakeAirNow(net)

    def handler(request):
        if "HourlyData_" in request.url.path:
            raise httpx.ConnectTimeout("timed out", request=request)
        return fake.handle(request)
    net.route(HOST, handler)
    t = NOW + timedelta(minutes=airnow.SITES_RETRY_MIN)
    s = _run(store.refresh(t))
    assert s["ok"] is False and s["sites"] == 12
    hour_calls = [r for r in net.calls(HOST) if "HourlyData_" in r.url.path]
    assert len(hour_calls) <= airnow.MAX_TRANSPORT_FAILURES + airnow.MAX_CONCURRENCY - 1
    assert any("unreachable" in e for e in s["errors"])

    # A handler bug (non-transport exception) is reported, not raised.
    def broken(request):
        if "HourlyData_" in request.url.path:
            raise RuntimeError("boom")
        return fake.handle(request)
    net.route(HOST, broken)
    s = _run(store.refresh(t + timedelta(minutes=airnow.MISSING_RETRY_MIN)))
    assert s["ok"] is False and any("boom" in e for e in s["errors"])

    # A caller bug (not a datetime) is reported too.
    s = _run(store.refresh("not a datetime"))
    assert s["ok"] is False and s["errors"][0].startswith("refresh:")


def test_server_errors_are_reported_not_stored(net):
    fake = FakeAirNow(net)
    _full_day(fake)

    def handler(request):
        if request.url.path.endswith("2026102910.dat"):
            return httpx.Response(503)
        return fake.handle(request)
    net.route(HOST, handler)
    store = airnow.AirNowStore()
    s = _run(store.refresh(NOW))
    assert s["ok"] is False and s["errors"] == ["2026-10-29T10:00:00+00:00: HTTP 503"]
    assert datetime(2026, 10, 29, 10, tzinfo=UTC) not in store.hours
    assert store.latest_means(NOW)["usable"] is True


def test_refetch_that_loses_most_sites_keeps_old_values(net):
    fake = FakeAirNow(net)
    _full_day(fake)
    store = airnow.AirNowStore()
    _run(store.refresh(NOW))
    h17 = NEWEST - timedelta(hours=1)
    fake.hours[h17] = {IDS[0]: 50.0}                  # a bad, near-empty file
    _run(store.refresh(NOW + timedelta(minutes=31)))
    vals = store.hours[h17]["values"]
    assert vals[IDS[0]] == 50.0 and len(vals) == 12


# ── Snapshots ────────────────────────────────────────────────────────────────

def test_snapshot_round_trip(net):
    fake = FakeAirNow(net)
    _full_day(fake)
    store = airnow.AirNowStore()
    _run(store.refresh(NOW))
    snap = json.loads(json.dumps(store.to_snapshot()))
    assert snap["schema"] == airnow.SNAPSHOT_SCHEMA

    restored = airnow.AirNowStore.from_snapshot(snap)
    assert restored.latest_means(NOW) == store.latest_means(NOW)
    assert restored.sites == store.sites
    later = NOW + timedelta(minutes=35)         # 19:55: newest is still 18:00
    assert restored.plan(later) == store.plan(later)

    # The restored store skips the site file (still fresh) and only fetches
    # what is due.
    net.requests.clear()
    fake.hours[NEWEST] = {a: 4.0 for a in IDS}
    s = _run(restored.refresh(later))
    assert fake.site_calls() == [] and s["ok"] is True
    assert set(fake.requested_hours()) == _labels(_provisional_after_cold_start()) | {"2026102918"}


def test_bad_snapshots_give_an_empty_store():
    for bad in (None, {}, {"schema": 99}, "nope", {"schema": airnow.SNAPSHOT_SCHEMA, "sites": 5}):
        st = airnow.AirNowStore.from_snapshot(bad)
        assert st.sites == {} and st.hours == {}
        assert st.latest_means(NOW)["usable"] is False

    snap = {"schema": airnow.SNAPSHOT_SCHEMA, "sites_fetched_at": NOW.isoformat(),
            "sites": {"480000100": {"name": "A", "lat": 30, "lon": -97},
                      "480000101": {"name": "B", "lat": "x", "lon": -97}},
            "hours": {NEWEST.isoformat(): {"fetched_at": NOW.isoformat(),
                                           "values": {"480000100": 5}},
                      "not a date": {"fetched_at": NOW.isoformat(), "values": {}}}}
    st = airnow.AirNowStore.from_snapshot(snap)
    assert set(st.sites) == {"480000100"} and list(st.hours) == [NEWEST]
    assert st.hours[NEWEST]["values"] == {"480000100": 5.0}

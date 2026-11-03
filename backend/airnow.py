"""EPA AirNow live monitor feed: 24-hour PM2.5 means at the regulatory-grade
monitors in and around Texas, for calibrating the map (backend/calibration.py).

Source: AirNow's public file server, no API key.
  Sites:  https://files.airnowtech.org/airnow/today/Monitoring_Site_Locations_V2.dat
  Hours:  https://files.airnowtech.org/airnow/YYYY/YYYYMMDD/HourlyData_YYYYMMDDHH.dat

What the fact sheets say (AirNow API "Hourly Data File" fact sheet, Aug 2025,
and the "Monitoring Site Information File" fact sheet), checked against real
files from 2026-10-28/29:
- Hourly file: pipe-delimited, no header, 9 fields:
  valid_date|valid_time|aqsid|sitename|gmt_offset|parameter|units|value|agency.
  Date and hour (in the file name and in every row) are GMT (UTC) and mark the
  BEGINNING of the hour: "17:00" is the sample measured 17:00 to 17:59 UTC.
  valid_date is mm/dd/yy (the fact sheet sample also shows "5/30/19"
  without a leading zero). Files are CRLF, UTF-8, ~740 KB / ~11k rows.
- Missing data is not coded: "Only valid data are reported in the data file",
  so a monitor that did not report an hour simply has no row (no -999 style
  sentinel was seen in 25 files). Negative values ARE valid ("AirNow stores
  negative raw concentrations, typically greater than -5", used in NowCast and
  daily averages), so they are kept like EPA keeps them. Seen nationwide: min
  -9.1; in our bbox min -4.3, max 47.0. Values below VALUE_MIN or above
  VALUE_MAX, non-numeric values and units other than UG/M3 are dropped.
- Files are updated twice an hour (:25 and :55) and "all hourly files for the
  preceding 48 hours will be updated every hour": an hour first appears
  partial (the newest file had 28 of ~66 Texas-area monitors at +1 h 16 min;
  on 2026-11-03 at 01:36 UTC the 23:00 and 00:00 files had 32 and 31 of ~72)
  and fills in later, hence the late-report re-fetch below. The early ~32
  are mostly the neighbor states plus 11 TCEQ sites; the other ~40 TCEQ
  monitors (TCEQ is 51 of the 72) arrive in one batch, watched on
  2026-11-03: the 00:00 and 01:00 UTC hours got it 1 h 16 min and 1 h 20 min
  after they ended, but the 23:00 hour was still without it at +3 h 21 min
  (older hours were all complete, so such hours are backfilled later).
  Files were rewritten at irregular minutes (:02 to :36), not only at the
  :25/:55 of the fact sheet. So the last hour of the 24 h window usually
  ended 1.5 to 2.5 hours before now. The file for an hour not yet
  published is a 404 (S3 NoSuchKey XML body), which is normal. Files carry an
  ETag and a conditional GET (If-None-Match) of an unchanged file is a 304.
- A few non-PM2.5 rows contain "PM2.5" in the site name ("Salmon PM2.5"), so
  rows are filtered on the parameter field, not on a substring.
- Sites file (V2): pipe-delimited WITH a header row, one row per
  site x parameter, 22 columns including StationID, AQSID, FullAQSID,
  Parameter, SiteName, Status, Latitude, Longitude. Hourly rows identify a
  monitor by any of the three ids: in one file 944 PM2.5 rows used the 9-char
  AQSID, 360 the 12-char FullAQSID ("840..."), 32 the StationID. All three are
  mapped to the AQSID, which is the public id here. Hourly sitename is
  truncated to 20 chars, so names come from the sites file.
- Of 105 Active PM2.5 sites in the bbox, only ~68 reported in a 25 h span (11
  are US Forest Service temporary smoke monitors, "MMFS..."; ~26 are TCEQ
  sites listed Active but silent). So "at least half of the sites reported"
  is measured against the sites that reported at least once in the stored
  hours, not against the metadata count, which would make the rule fragile.
- No duplicate (site, hour) rows were seen; if one appears the first is kept.

Rules implemented here:
- Sites: Parameter == 'PM2.5', Status == 'Active', inside the PurpleAir bbox
  (BBOX, Texas plus border neighbors). Metadata cached SITES_TTL_H.
- Rolling store of STORE_HOURS hours of PM2.5 values for those sites only.
- 24-hour mean: the WINDOW_HOURS consecutive hours ending at the newest hour
  in which at least half of the reporting sites reported; a site needs at
  least MIN_VALID_HOURS valid hours (the EPA 75% completeness rule). The
  result is usable with at least MIN_MONITORS monitors.

Integrator API (one long-lived AirNowStore; everything else is pure):

    from backend.airnow import AirNowStore
    store = AirNowStore.from_snapshot(_load_snapshot(AIRNOW_SNAPSHOT_PATH))
        # at startup; None or a bad snapshot gives an empty store
    summary = await store.refresh(now)
        # every refresh cycle; never raises. Cold start downloads the site
        # file (~4.3 MB) plus 30 hourly files (~740 KB each): measured
        # 26.2 MB in 3.3 s from a home connection, peak ~18 MB of Python
        # allocations. A warm cycle sends ~2 to 8 conditional GETs and
        # downloads 0 to ~4 files (re-fetches of unchanged files are 304s);
        # each hour is fetched until one fetch lands LATE_REPORT_H after
        # it ended. summary["changed"] says whether the
        # stored data changed (then persist the snapshot, ~55 KB).
    if summary["changed"]:
        await _save_snapshot_async(AIRNOW_SNAPSHOT_PATH, store.to_snapshot())
    means = store.latest_means(now)
        # {"usable", "window_start", "window_end", "monitors": [...], "reason"}
        # cheap (pure Python over ~3k numbers), safe to call on the event loop.

refresh() and latest_means() must be called from the same event loop thread
(the store is mutated only there; parsing runs in worker threads on copies).
"""
import asyncio
import math
import time
from datetime import datetime, timedelta, timezone

import httpx

AIRNOW_HOST = "files.airnowtech.org"
SITES_URL = f"https://{AIRNOW_HOST}/airnow/today/Monitoring_Site_Locations_V2.dat"
HOURLY_URL = f"https://{AIRNOW_HOST}/airnow/{{y}}/{{ymd}}/HourlyData_{{ymdh}}.dat"

# PurpleAir fetch bbox (Texas plus border neighbors), inclusive.
BBOX = {"lat_min": 25.84, "lat_max": 36.5, "lon_min": -106.65, "lon_max": -93.5}

SITES_TTL_H = 24
# Retry a failed site-file download no more often than this.
SITES_RETRY_MIN = 30
# Hours kept: the newest attemptable hour (the one that began an hour before
# the current hour) and the 29 before it. Long enough for a 24 h window that
# ends up to MAX_WINDOW_LAG_H hours before the newest hour.
STORE_HOURS = 30
WINDOW_HOURS = 24
MIN_VALID_HOURS = 18
MIN_MONITORS = 10
MAX_WINDOW_LAG_H = STORE_HOURS - WINDOW_HOURS
# An hour fetched less than this long after it ended is provisional (late
# reports are still arriving) and is re-fetched, at most every REFETCH_MIN.
# TCEQ's ~50 monitors usually land about 1 h 20 min after the hour ends, but
# not always (see the module docstring), so this leaves a wide margin.
LATE_REPORT_H = 4
REFETCH_MIN = 30
# A batch that arrives later than that (TCEQ's 23:00 UTC hour on 2026-11-02
# was still missing at +3 h 21 min) would freeze the hour without this: a
# stored hour with fewer than SPARSE_FRACTION of the sites in the fullest
# stored hour is re-fetched every SPARSE_REFETCH_MIN for as long as it is
# stored (AirNow rewrites the last 48 h hourly; a conditional GET of an
# unchanged file is a cheap 304).
SPARSE_FRACTION = 0.75
SPARSE_REFETCH_MIN = 60
# A 404 or failed hour is retried no more often than this.
MISSING_RETRY_MIN = 10
MAX_CONCURRENCY = 4
# After this many transport failures (timeouts, refused connections) in one
# refresh the remaining downloads are skipped: AirNow is down, don't queue 30.
MAX_TRANSPORT_FAILURES = 4
HTTP_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
# Plausible hourly PM2.5 range (ug/m3). Negatives down to about -5 are valid
# instrument noise per AirNow; anything beyond these is a glitch.
VALUE_MIN = -10.0
VALUE_MAX = 1000.0
SNAPSHOT_SCHEMA = 1


def _utc(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def _floor_hour(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)


def hour_url(hour: datetime) -> str:
    """URL of the hourly file for the UTC hour that BEGINS at `hour`."""
    return HOURLY_URL.format(y=f"{hour:%Y}", ymd=f"{hour:%Y%m%d}", ymdh=f"{hour:%Y%m%d%H}")


def target_hours(now: datetime) -> list[datetime]:
    """The STORE_HOURS hour-beginning UTC labels the store should hold, newest
    first. The current hour is still being measured, so the newest is the one
    before it."""
    newest = _floor_hour(_utc(now)) - timedelta(hours=1)
    return [newest - timedelta(hours=i) for i in range(STORE_HOURS)]


def _in_bbox(lat: float, lon: float) -> bool:
    return (BBOX["lat_min"] <= lat <= BBOX["lat_max"]
            and BBOX["lon_min"] <= lon <= BBOX["lon_max"])


# ── Parsing (pure; runs in worker threads) ───────────────────────────────────

def parse_sites(content: bytes) -> dict[str, dict]:
    """Active PM2.5 sites inside BBOX from the V2 site-locations file, keyed by
    AQSID: {aqsid: {"id", "name", "lat", "lon", "aliases": [ids]}}. The aliases
    are the other ids an hourly row may use (FullAQSID, StationID).
    Raises ValueError when the header lacks a required column."""
    text = content.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if not lines:
        raise ValueError("empty site file")
    header = [h.strip() for h in lines[0].split("|")]
    col = {h: i for i, h in enumerate(header)}
    need = ("AQSID", "Parameter", "SiteName", "Status", "Latitude", "Longitude")
    missing = [c for c in need if c not in col]
    if missing:
        raise ValueError(f"site file header lacks {missing}")
    i_aqs, i_par, i_name = col["AQSID"], col["Parameter"], col["SiteName"]
    i_stat, i_lat, i_lon = col["Status"], col["Latitude"], col["Longitude"]
    i_full, i_station = col.get("FullAQSID"), col.get("StationID")
    width = max(col.values()) + 1

    sites: dict[str, dict] = {}
    for line in lines[1:]:
        if "|PM2.5|" not in line:
            continue
        p = line.split("|")
        if len(p) < width:
            continue
        if p[i_par].strip() != "PM2.5" or p[i_stat].strip() != "Active":
            continue
        try:
            lat, lon = float(p[i_lat]), float(p[i_lon])
        except ValueError:
            continue
        if not (math.isfinite(lat) and math.isfinite(lon)) or not _in_bbox(lat, lon):
            continue
        aqsid = p[i_aqs].strip()
        if not aqsid or aqsid in sites:
            continue
        aliases = []
        for i in (i_full, i_station):
            if i is not None:
                a = p[i].strip()
                if a and a != aqsid and a not in aliases:
                    aliases.append(a)
        sites[aqsid] = {"id": aqsid, "name": p[i_name].strip() or aqsid,
                        "lat": lat, "lon": lon, "aliases": aliases}
    return sites


def build_alias_index(sites: dict[str, dict]) -> dict[str, str]:
    """Every id an hourly row may carry for one of our sites -> its AQSID."""
    index: dict[str, str] = {}
    for aqsid, s in sites.items():
        index[aqsid] = aqsid
        for a in s.get("aliases") or ():
            index.setdefault(a, aqsid)
    return index


def _row_hour(date_s: str, time_s: str) -> datetime | None:
    """UTC hour-beginning datetime of a row's mm/dd/yy + hh:mm, or None."""
    try:
        m, d, y = (int(x) for x in date_s.strip().split("/"))
        hh = int(time_s.strip().split(":")[0])
        return datetime(2000 + y if y < 100 else y, m, d, hh, tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def parse_hourly(content: bytes, hour: datetime, alias_index: dict[str, str]) -> tuple[dict, dict]:
    """PM2.5 values for our sites from one hourly file.

    Returns ({aqsid: value}, stats). Rows for other hours, malformed rows,
    non-numeric or implausible values and non-UG/M3 units are skipped and
    counted in stats; a duplicate (site, hour) keeps the first value."""
    stats = {"rows": 0, "pm25_rows": 0, "kept": 0, "malformed": 0, "wrong_hour": 0,
             "bad_value": 0, "duplicate": 0}
    values: dict[str, float] = {}
    text = content.decode("utf-8", errors="replace")
    for line in text.splitlines():
        if not line:
            continue
        stats["rows"] += 1
        if "PM2.5" not in line:  # cheap pre-filter: ~1 row in 8 is PM2.5
            continue
        p = line.split("|")
        if len(p) != 9:
            stats["malformed"] += 1
            continue
        if p[5].strip() != "PM2.5":  # e.g. TEMP at a site named "Salmon PM2.5"
            continue
        stats["pm25_rows"] += 1
        aqsid = alias_index.get(p[2].strip())
        if aqsid is None:
            continue
        if _row_hour(p[0], p[1]) != hour:
            stats["wrong_hour"] += 1
            continue
        if p[6].strip().upper() != "UG/M3":
            stats["bad_value"] += 1
            continue
        try:
            v = float(p[7])
        except ValueError:
            stats["bad_value"] += 1
            continue
        if not math.isfinite(v) or v < VALUE_MIN or v > VALUE_MAX:
            stats["bad_value"] += 1
            continue
        if aqsid in values:
            stats["duplicate"] += 1
            continue
        values[aqsid] = v
        stats["kept"] += 1
    return values, stats


# ── 24-hour means (pure) ─────────────────────────────────────────────────────

def compute_means(sites: dict[str, dict], hours: dict[datetime, dict], now: datetime) -> dict:
    """The latest_means result from stored hours ({hour: {"values": {...}}}).
    See AirNowStore.latest_means."""
    now = _utc(now)
    targets = target_hours(now)
    newest_allowed, oldest_allowed = targets[0], targets[-1]
    present = {h: rec["values"] for h, rec in hours.items()
               if oldest_allowed <= h <= newest_allowed and rec.get("values") is not None}
    out = {"usable": False, "window_start": None, "window_end": None,
           "monitors": [], "reason": None}
    reporting = set()
    for vals in present.values():
        reporting.update(k for k in vals if k in sites)
    if not reporting:
        out["reason"] = "no AirNow hourly data for our sites in the last %d h" % STORE_HOURS
        return out

    # Newest hour in which at least half of the reporting sites reported. It
    # must leave room for a full window inside the stored span.
    half = len(reporting) / 2.0
    end = None
    for h in sorted(present, reverse=True):
        if h < newest_allowed - timedelta(hours=MAX_WINDOW_LAG_H):
            break
        if sum(1 for k in present[h] if k in sites) >= half:
            end = h
            break
    if end is None:
        out["reason"] = (f"no hour in the last {MAX_WINDOW_LAG_H + 1} h with at least half "
                         f"of the {len(reporting)} reporting monitors")
        return out
    start = end - timedelta(hours=WINDOW_HOURS - 1)
    out["window_start"] = start.isoformat()
    out["window_end"] = end.isoformat()

    window = [present[h] for h in sorted(present) if start <= h <= end]
    monitors = []
    for aqsid in sorted(reporting):
        vals = [w[aqsid] for w in window if aqsid in w]
        if len(vals) < MIN_VALID_HOURS:
            continue
        s = sites[aqsid]
        monitors.append({"id": aqsid, "name": s["name"], "lat": s["lat"], "lon": s["lon"],
                         "pm25": round(sum(vals) / len(vals), 2), "hours": len(vals)})
    out["monitors"] = monitors
    out["usable"] = len(monitors) >= MIN_MONITORS
    if not out["usable"]:
        out["reason"] = (f"only {len(monitors)} monitors with >= {MIN_VALID_HOURS} valid hours "
                         f"(need {MIN_MONITORS})")
    return out


# ── The store ────────────────────────────────────────────────────────────────

class AirNowStore:
    """Site metadata plus a rolling STORE_HOURS store of hourly PM2.5 values.

    sites: {aqsid: {"id","name","lat","lon","aliases"}}
    hours: {hour-beginning UTC datetime: {"fetched_at": datetime,
            "etag": str | None, "values": {aqsid: float}}}
    """

    def __init__(self):
        self.sites: dict[str, dict] = {}
        self.sites_fetched_at: datetime | None = None
        self.hours: dict[datetime, dict] = {}
        self._alias: dict[str, str] = {}
        # Not persisted: when a site download / an hour last failed or 404'd.
        self._sites_failed_at: datetime | None = None
        self._missing: dict[datetime, datetime] = {}
        self._lock: asyncio.Lock | None = None

    # ---- reading ----
    def latest_means(self, now: datetime | None = None) -> dict:
        """24-hour mean PM2.5 per monitor:
        {"usable": bool, "window_start": ISO | None, "window_end": ISO | None,
         "monitors": [{"id","name","lat","lon","pm25","hours"}], "reason": str | None}.
        window_start / window_end are the hour-beginning UTC labels of the first
        and last hour (AirNow's convention): the data cover window_start up to
        window_end + 1 h. pm25 is rounded to 2 dp. Monitors are listed even when
        usable is False; reason says why it is not usable."""
        try:
            return compute_means(self.sites, self.hours, _utc(now))
        except Exception as e:  # never break the caller's compute
            return {"usable": False, "window_start": None, "window_end": None,
                    "monitors": [], "reason": f"{type(e).__name__}: {e}"}

    # ---- planning ----
    def _sites_due(self, now: datetime) -> bool:
        if self.sites and self.sites_fetched_at \
                and now - self.sites_fetched_at < timedelta(hours=SITES_TTL_H):
            return False
        if self._sites_failed_at and now - self._sites_failed_at < timedelta(minutes=SITES_RETRY_MIN):
            return False
        return True

    def plan(self, now: datetime) -> list[tuple[datetime, str]]:
        """Hours to download now, newest first, as (hour, "new" | "refetch").
        A stored hour is re-fetched every REFETCH_MIN until one fetch lands
        at least LATE_REPORT_H after the hour ended, and every
        SPARSE_REFETCH_MIN while it holds fewer than SPARSE_FRACTION of the
        sites in the fullest stored hour. Both are bounded: an hour leaves
        the store after STORE_HOURS."""
        now = _utc(now)
        targets = target_hours(now)
        fullest = max((len(self.hours[h]["values"]) for h in targets if h in self.hours), default=0)
        todo = []
        for h in targets:
            failed = self._missing.get(h)
            if failed is not None and now - failed < timedelta(minutes=MISSING_RETRY_MIN):
                continue
            rec = self.hours.get(h)
            if rec is None:
                todo.append((h, "new"))
                continue
            fetched = rec["fetched_at"]
            since = now - fetched
            provisional = fetched < h + timedelta(hours=1 + LATE_REPORT_H)
            sparse = len(rec["values"]) < SPARSE_FRACTION * fullest
            if (provisional and since >= timedelta(minutes=REFETCH_MIN)) \
                    or (sparse and since >= timedelta(minutes=SPARSE_REFETCH_MIN)):
                todo.append((h, "refetch"))
        return todo

    def _prune(self, now: datetime) -> None:
        oldest = target_hours(now)[-1]
        for h in [h for h in self.hours if h < oldest]:
            del self.hours[h]
        for h in [h for h in self._missing if h < oldest]:
            del self._missing[h]

    # ---- fetching ----
    async def refresh(self, now: datetime | None = None) -> dict:
        """Bring the store up to date for `now` (UTC). Downloads the site file
        when older than SITES_TTL_H, every target hour not yet stored, and
        provisional or sparse hours due for a late-report re-fetch (see
        plan). Never raises.

        Returns {"ok", "changed", "sites", "sites_refreshed", "fetched": [ISO],
        "refetched": [ISO], "unchanged": [ISO], "not_published": [ISO],
        "errors": [str], "bytes", "rows_skipped", "seconds", "hours_stored"}.
        rows_skipped counts malformed PM2.5 rows in the files parsed plus
        implausible, wrong-hour and duplicate rows for our sites.
        ok is False only when something other than a not-yet-published hour
        failed."""
        t0 = time.monotonic()
        summary = {"ok": True, "changed": False, "sites": len(self.sites),
                   "sites_refreshed": False, "fetched": [], "refetched": [],
                   "unchanged": [], "not_published": [], "errors": [], "bytes": 0,
                   "rows_skipped": 0, "seconds": 0.0, "hours_stored": len(self.hours)}
        if self._lock is None:
            self._lock = asyncio.Lock()
        try:
            now = _utc(now)
            async with self._lock:
                await self._refresh_locked(now, summary)
        except Exception as e:
            summary["errors"].append(f"refresh: {type(e).__name__}: {e}")
        summary["ok"] = not summary["errors"]
        summary["sites"] = len(self.sites)
        summary["hours_stored"] = len(self.hours)
        summary["seconds"] = round(time.monotonic() - t0, 2)
        return summary

    async def _refresh_locked(self, now: datetime, summary: dict) -> None:
        self._prune(now)
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
            if self._sites_due(now):
                await self._refresh_sites(client, now, summary)
            if not self.sites:
                summary["errors"].append("no site metadata; hourly files skipped")
                return
            todo = self.plan(now)
            if not todo:
                return
            sem = asyncio.Semaphore(MAX_CONCURRENCY)
            alias = dict(self._alias)
            failures = {"transport": 0}
            results = await asyncio.gather(
                *(self._fetch_hour(client, sem, alias, h, kind, now, summary, failures)
                  for h, kind in todo),
                return_exceptions=True)
            for h_kind, r in zip(todo, results):
                if isinstance(r, BaseException):
                    summary["errors"].append(f"{h_kind[0].isoformat()}: {type(r).__name__}: {r}")

    async def _refresh_sites(self, client: httpx.AsyncClient, now: datetime, summary: dict) -> None:
        try:
            resp = await client.get(SITES_URL)
            summary["bytes"] += len(resp.content)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}")
            sites = await asyncio.to_thread(parse_sites, resp.content)
            if not sites:
                raise RuntimeError("no active PM2.5 sites in the bbox")
        except Exception as e:
            self._sites_failed_at = now
            summary["errors"].append(f"sites: {type(e).__name__}: {e}")
            return
        self.sites = sites
        self._alias = build_alias_index(sites)
        self.sites_fetched_at = now
        self._sites_failed_at = None
        summary["sites_refreshed"] = True
        summary["changed"] = True

    async def _fetch_hour(self, client, sem, alias, hour, kind, now, summary, failures) -> None:
        label = hour.isoformat()
        async with sem:
            if failures["transport"] >= MAX_TRANSPORT_FAILURES:
                self._missing[hour] = now
                summary["errors"].append(f"{label}: skipped (AirNow unreachable)")
                return
            prev = self.hours.get(hour)
            headers = {}
            if prev is not None and prev.get("etag"):
                headers["If-None-Match"] = prev["etag"]
            try:
                resp = await client.get(hour_url(hour), headers=headers)
            except Exception as e:
                if isinstance(e, httpx.TransportError):
                    failures["transport"] += 1
                self._missing[hour] = now
                summary["errors"].append(f"{label}: {type(e).__name__}: {e}")
                return
            summary["bytes"] += len(resp.content)
            if resp.status_code == 304 and prev is not None:
                prev["fetched_at"] = now
                summary["unchanged"].append(label)
                summary["changed"] = True
                return
            if resp.status_code in (403, 404):
                # Not published yet (normal for the newest hour), or a gap.
                self._missing[hour] = now
                summary["not_published"].append(label)
                return
            if resp.status_code != 200:
                self._missing[hour] = now
                summary["errors"].append(f"{label}: HTTP {resp.status_code}")
                return
            etag = resp.headers.get("ETag")
            try:
                values, stats = await asyncio.to_thread(parse_hourly, resp.content, hour, alias)
            except Exception as e:
                self._missing[hour] = now
                summary["errors"].append(f"{label}: parse {type(e).__name__}: {e}")
                return
            del resp  # the ~740 KB body is not kept past its parse
            summary["rows_skipped"] += (stats["malformed"] + stats["wrong_hour"]
                                        + stats["bad_value"] + stats["duplicate"])
        # The store is only mutated here, on the event loop thread.
        if prev is not None and len(values) < len(prev["values"]) / 2:
            # A re-fetch that lost more than half the sites looks like a bad
            # file, not late reports: keep the old values, take any new ones.
            values = {**prev["values"], **values}
            etag = None
        self.hours[hour] = {"fetched_at": now, "etag": etag, "values": values}
        self._missing.pop(hour, None)
        summary["refetched" if kind == "refetch" else "fetched"].append(label)
        summary["changed"] = True

    # ---- snapshot ----
    def to_snapshot(self) -> dict:
        """JSON-serializable state for SNAPSHOTS_DIR (~100 KB)."""
        return {
            "schema": SNAPSHOT_SCHEMA,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "sites_fetched_at": self.sites_fetched_at.isoformat() if self.sites_fetched_at else None,
            "sites": self.sites,
            "hours": {h.isoformat(): {"fetched_at": r["fetched_at"].isoformat(),
                                      "etag": r.get("etag"), "values": r["values"]}
                      for h, r in sorted(self.hours.items())},
        }

    @classmethod
    def from_snapshot(cls, data: dict | None) -> "AirNowStore":
        """A store hydrated from to_snapshot() output. None, a different schema
        or a corrupt snapshot gives an empty store; bad entries are skipped.
        Stale hours are pruned on the next refresh and ignored by latest_means."""
        store = cls()
        if not isinstance(data, dict) or data.get("schema") != SNAPSHOT_SCHEMA:
            return store
        try:
            sites = {}
            for k, s in (data.get("sites") or {}).items():
                try:
                    sites[str(k)] = {"id": str(k), "name": str(s.get("name") or k),
                                     "lat": float(s["lat"]), "lon": float(s["lon"]),
                                     "aliases": [str(a) for a in s.get("aliases") or []]}
                except (KeyError, TypeError, ValueError, AttributeError):
                    continue
            hours = {}
            for k, r in (data.get("hours") or {}).items():
                try:
                    h = _utc(datetime.fromisoformat(k))
                    vals = {str(a): float(v) for a, v in (r.get("values") or {}).items()
                            if v is not None and math.isfinite(float(v))}
                    hours[h] = {"fetched_at": _utc(datetime.fromisoformat(r["fetched_at"])),
                                "etag": r.get("etag"), "values": vals}
                except (KeyError, TypeError, ValueError, AttributeError):
                    continue
            fetched = data.get("sites_fetched_at")
            store.sites = sites
            store._alias = build_alias_index(sites)
            store.sites_fetched_at = _utc(datetime.fromisoformat(fetched)) if fetched and sites else None
            store.hours = hours
        except Exception as e:
            print(f"[airnow] ignoring snapshot ({type(e).__name__}: {e})")
            return cls()
        return store

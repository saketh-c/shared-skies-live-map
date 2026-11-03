#!/usr/bin/env python3
"""Log one prediction snapshot of the live map to a GitHub Release.

Render's disk is ephemeral, so the backend keeps no history. This script runs
from .github/workflows/prediction-log.yml every 3 hours: it fetches
/api/health, /api/texas/predictions and /api/live/sensors, packs them into
snapshot_<YYYYMMDDTHHMMZ>.tar.gz (named for the predictions' generated_at),
and uploads that as an asset of the monthly release prediction-log-YYYY-MM.
scripts/fetch_prediction_log.py downloads and merges them for analysis.

Standard library only (no pip installs on the runner). Upload uses the gh CLI
with GH_TOKEN set.

Tarball contents:
  tracts.csv    geoid, pm25 (displayed), pm25_model (raw model; empty on an
                older backend), category
  monitors.csv  EPA AirNow monitors with the map values at each one (only
                when /api/live/sensors exists)
  sensors.csv   live PurpleAir sensors, raw and calibrated (same condition)
  meta.json     generated_at, calibration, data_sources, value_semantics,
                health, and fetch details

Failure policy: a degraded backend is still logged, since that is data. The
script exits non-zero only when the backend never answered during the whole
retry window, or when the upload itself fails.

Local use (macOS python.org builds may need SSL_CERT_FILE=/etc/ssl/cert.pem):
  python3 scripts/log_snapshot.py --dry-run --out-dir /tmp/snap
"""
from __future__ import annotations

import argparse
import csv
import gzip
import http.client
import io
import json
import os
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

DEFAULT_BASE_URL = "https://shared-skies-backend.onrender.com"
SCHEMA_VERSION = 1
USER_AGENT = "shared-skies-prediction-log/1"

TRACT_FIELDS = ["geoid", "pm25", "pm25_model", "category"]
MONITOR_FIELDS = ["id", "name", "lat", "lon", "pm25", "hours",
                  "map_pm25", "map_pm25_loo", "model_pm25"]
SENSOR_FIELDS = ["lat", "lon", "pm25_raw", "pm25"]

RELEASE_BODY = """\
Automated prediction log for the Shared Skies live PM2.5 map. Not a software release.

Every 3 hours `.github/workflows/prediction-log.yml` saves what the map was serving as one asset, `snapshot_<YYYYMMDDTHHMMZ>.tar.gz`, named for the predictions' `generated_at` time (UTC). Each tarball holds:

- `tracts.csv`: geoid, pm25 (displayed value), pm25_model (raw model value), category
- `monitors.csv`: EPA AirNow monitors (24-h mean) with the calibrated, leave-one-out and raw map values at each monitor
- `sensors.csv`: live PurpleAir sensors, raw and calibrated 24-h averages
- `meta.json`: generated_at, calibration block, data_sources, value_semantics and /api/health at fetch time

`monitors.csv` and `sensors.csv` are missing from snapshots taken before the backend had /api/live/sensors. Use `scripts/fetch_prediction_log.py` to download and merge every snapshot.
"""


def log(msg: str) -> None:
    print(msg, flush=True)


def gh_warning(msg: str) -> None:
    # Shows up as an annotation on the Actions run; plain text locally.
    prefix = "::warning::" if os.environ.get("GITHUB_ACTIONS") else "warning: "
    print(prefix + msg, flush=True)


class Fetcher:
    """GETs JSON with retries until a shared deadline. A free Render instance
    can take a minute or more to wake, then answers 503 until its first
    compute lands."""

    def __init__(self, base_url: str, deadline: float, retry_sleep: float = 20.0):
        self.base_url = base_url.rstrip("/")
        self.deadline = deadline
        self.retry_sleep = retry_sleep
        self.ever_answered = False

    def _once(self, path: str) -> tuple[int, object | None, str]:
        """One request. Returns (status, parsed JSON or None, error text).
        status 0 means no HTTP answer at all (DNS, connect, timeout)."""
        remaining = self.deadline - time.monotonic()
        timeout = max(10.0, min(90.0, remaining))
        req = urllib.request.Request(
            self.base_url + path,
            headers={"Accept": "application/json", "Accept-Encoding": "gzip",
                     "User-Agent": USER_AGENT},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                if (resp.headers.get("Content-Encoding") or "").lower() == "gzip":
                    raw = gzip.decompress(raw)
                body = json.loads(raw.decode("utf-8"))
                self.ever_answered = True
                return resp.status, body, ""
        except urllib.error.HTTPError as e:
            # Only a JSON error body (FastAPI's {"detail": ...}) proves the app
            # itself answered. Render's proxy answers 502/503 with HTML while
            # the instance is down or a deploy failed: that is unreachable.
            try:
                raw = e.read()
                if (e.headers.get("Content-Encoding") or "").lower() == "gzip":
                    raw = gzip.decompress(raw)
                json.loads(raw.decode("utf-8"))
                self.ever_answered = True
            except (OSError, ValueError, AttributeError, EOFError, http.client.HTTPException):
                pass
            return e.code, None, f"HTTP {e.code}"
        except ValueError as e:
            # 200 but not JSON: some proxy page, not the app.
            return 0, None, f"bad JSON: {e}"
        except (urllib.error.URLError, TimeoutError, OSError, EOFError,
                http.client.HTTPException) as e:
            # HTTPException covers a body cut short (IncompleteRead) when the
            # instance restarts mid-response; EOFError, a truncated gzip body.
            return 0, None, f"{type(e).__name__}: {e}"

    def get(self, path: str, *, give_up_on: tuple[int, ...] = ()) -> tuple[int, object | None]:
        """Retries until a 200 with a JSON object, a status in give_up_on, or
        the deadline. Returns the last (status, body)."""
        attempt = 0
        while True:
            attempt += 1
            status, body, err = self._once(path)
            if status == 200 and isinstance(body, dict):
                return status, body
            if status in give_up_on:
                return status, None
            left = self.deadline - time.monotonic()
            log(f"  {path}: attempt {attempt} failed ({err or f'HTTP {status}'}); "
                f"{max(0, int(left))} s left in retry window")
            if left <= self.retry_sleep:
                return status, None
            time.sleep(self.retry_sleep)


def parse_iso(value: str | None) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def csv_bytes(fields: list[str], rows: list[dict]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for row in rows:
        w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fields})
    return buf.getvalue().encode("utf-8")


def build_tarball(path: str, members: dict[str, bytes], mtime: float) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = int(mtime)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))


def gh(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["gh", *args], capture_output=True, text=True, check=check)


def upload(repo: str, tag: str, month: str, tar_path: str) -> str:
    """Uploads tar_path to release `tag`, creating the release if needed.
    Returns "uploaded" or "exists"."""
    asset = os.path.basename(tar_path)
    view = gh("release", "view", tag, "--repo", repo, "--json", "assets",
              "--jq", ".assets[].name", check=False)
    if view.returncode != 0:
        if "not found" not in (view.stderr or "").lower():
            raise RuntimeError(f"gh release view failed: {view.stderr.strip()}")
        log(f"Creating release {tag}")
        # Prerelease and never "latest", so it cannot be mistaken for a
        # software release. The tag is created by GITHUB_TOKEN, and events
        # made with that token do not start other workflows.
        create = gh("release", "create", tag, "--repo", repo, "--prerelease",
                    "--latest=false", "--title", f"Prediction log {month} (automated)",
                    "--notes", RELEASE_BODY, check=False)
        if create.returncode != 0:
            raise RuntimeError(f"gh release create failed: {create.stderr.strip()}")
        existing: set[str] = set()
    else:
        existing = {line.strip() for line in view.stdout.splitlines() if line.strip()}
    if asset in existing:
        return "exists"
    up = gh("release", "upload", tag, tar_path, "--repo", repo, check=False)
    if up.returncode != 0:
        # A concurrent upload of the same name is not an error.
        if "already exists" in (up.stderr or "").lower():
            return "exists"
        raise RuntimeError(f"gh release upload failed: {up.stderr.strip()}")
    return "uploaded"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-url", default=os.environ.get("SSI_BACKEND_URL", DEFAULT_BASE_URL),
                    help="backend base URL (env SSI_BACKEND_URL)")
    ap.add_argument("--out-dir", default=".", help="where the tarball is written")
    ap.add_argument("--dry-run", action="store_true",
                    help="write the tarball but do not upload it")
    ap.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", "saketh-c/shared-skies-live-map"),
                    help="owner/name of the repo that holds the releases")
    ap.add_argument("--max-wait", type=float,
                    default=float(os.environ.get("SSI_LOG_MAX_WAIT_S", "480")),
                    help="seconds to keep retrying a sleeping or warming backend")
    ap.add_argument("--retry-sleep", type=float, default=20.0)
    args = ap.parse_args(argv)

    started = datetime.now(timezone.utc)
    f = Fetcher(args.base_url, time.monotonic() + args.max_wait, args.retry_sleep)
    log(f"Backend: {f.base_url} (retry window {int(args.max_wait)} s)")

    _, health = f.get("/api/health")
    if health is None and not f.ever_answered:
        print(f"::error::Backend unreachable for the whole {int(args.max_wait)} s retry window",
              flush=True)
        return 1
    if health is not None:
        log(f"Health: status={health.get('status')} live_neighbors={health.get('live_neighbors')} "
            f"predictions_age_min={health.get('predictions_age_min')}")

    # Predictions and sensors come from the same compute only if no refresh
    # lands between the two requests. The backend swaps both at once, so on a
    # mismatch the sensors are the newer cycle: re-fetch predictions once.
    sensors_status, sensors = None, None
    _, preds = f.get("/api/texas/predictions")
    if preds is not None:
        sensors_status, sensors = f.get("/api/live/sensors", give_up_on=(404,))
        if sensors is not None and sensors.get("generated_at") != preds.get("generated_at"):
            log("  sensors and predictions straddle a refresh; fetching predictions again")
            _, again = f.get("/api/texas/predictions")
            if again is not None:
                preds = again

    if preds is None:
        if not f.ever_answered:
            print("::error::Backend unreachable for the whole retry window", flush=True)
            return 1
        gh_warning("Backend answered but /api/texas/predictions never returned data; "
                   "nothing to log this run")
        return 0
    if sensors_status == 404:
        log("/api/live/sensors: 404 (older backend); logging tracts only")
    elif sensors is None:
        gh_warning("/api/live/sensors did not answer in time; logging tracts only")

    gen = parse_iso(preds.get("generated_at"))
    if gen is None:
        gh_warning(f"Unparseable generated_at {preds.get('generated_at')!r}; naming by fetch time")
        gen = started
    stamp = gen.strftime("%Y%m%dT%H%MZ")
    month = gen.strftime("%Y-%m")
    tag = f"prediction-log-{month}"

    tracts = preds.get("tracts") or []
    members: dict[str, bytes] = {"tracts.csv": csv_bytes(TRACT_FIELDS, tracts)}
    pa = (sensors or {}).get("purpleair") or {}
    mon = (sensors or {}).get("monitors") or {}
    if sensors is not None:
        members["monitors.csv"] = csv_bytes(MONITOR_FIELDS, mon.get("monitors") or [])
        members["sensors.csv"] = csv_bytes(SENSOR_FIELDS, pa.get("sensors") or [])

    meta = {
        "schema_version": SCHEMA_VERSION,
        "snapshot": stamp,
        "generated_at": preds.get("generated_at"),
        "fetched_at": started.isoformat(),
        "base_url": f.base_url,
        "calibration": preds.get("calibration"),
        "data_sources": preds.get("data_sources"),
        "value_semantics": preds.get("value_semantics"),
        "tract_count": len(tracts),
        "avg_pm25": preds.get("avg_pm25"),
        "weather": preds.get("weather"),
        "health": health,
        "live_sensors": None if sensors is None else {
            "generated_at": sensors.get("generated_at"),
            "matches_predictions": sensors.get("generated_at") == preds.get("generated_at"),
            "calibration_applied": sensors.get("calibration_applied"),
            "purpleair": {k: v for k, v in pa.items() if k != "sensors"},
            "monitors": {k: v for k, v in mon.items() if k != "monitors"},
        },
        "live_sensors_status": sensors_status,
    }
    members["meta.json"] = (json.dumps(meta, indent=2, ensure_ascii=False) + "\n").encode("utf-8")

    os.makedirs(args.out_dir, exist_ok=True)
    tar_path = os.path.join(args.out_dir, f"snapshot_{stamp}.tar.gz")
    build_tarball(tar_path, members, gen.timestamp())
    log(f"Wrote {tar_path} ({os.path.getsize(tar_path)} bytes): {len(tracts)} tracts, "
        f"{len(mon.get('monitors') or [])} monitors, {len(pa.get('sensors') or [])} sensors")

    if args.dry_run:
        log(f"Dry run: would upload to release {tag} of {args.repo}")
        return 0
    result = upload(args.repo, tag, month, tar_path)
    if result == "exists":
        log(f"{os.path.basename(tar_path)} is already in {tag} (backend has not refreshed); skipped")
    else:
        log(f"Uploaded to {tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

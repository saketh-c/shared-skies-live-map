#!/usr/bin/env python3
"""Download the prediction log and merge it into analysis-ready CSVs.

scripts/log_snapshot.py (run by .github/workflows/prediction-log.yml) uploads
one snapshot_<YYYYMMDDTHHMMZ>.tar.gz every 3 hours to the monthly GitHub
Release prediction-log-YYYY-MM. This script downloads every such asset
(cached under <out-dir>/snapshots, so re-runs only fetch new ones) and writes:

  tracts.csv     snapshot_time, geoid, pm25, pm25_model, category
  monitors.csv   snapshot_time, id, name, lat, lon, pm25, hours, map_pm25,
                 map_pm25_loo, model_pm25
  sensors.csv    snapshot_time, lat, lon, pm25_raw, pm25
  snapshots.csv  one row per snapshot: snapshot_time, calibration fields,
                 health status, row counts

snapshot_time is the predictions' generated_at (ISO, UTC). Monitors and
sensors exist only for snapshots taken after the backend gained
/api/live/sensors. Example analysis: per snapshot, compare monitors.csv
map_pm25_loo (and model_pm25) against pm25 to track the calibrated map's
accuracy over weeks.

Standard library only. Lists releases with the GitHub REST API. Without a
token, snapshots download from the public release URLs, which do not count
against the API's 60-requests-an-hour limit; with GITHUB_TOKEN (or GH_TOKEN)
set they download through the API, which also works for a private repo.
--list only prints what is available. Exits 1 if any snapshot failed to
download (the merged CSVs are then incomplete; re-run to fetch the rest).
(macOS python.org builds may need SSL_CERT_FILE=/etc/ssl/cert.pem.)

  python3 scripts/fetch_prediction_log.py --out-dir prediction_log
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import tarfile
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_REPO = "saketh-c/shared-skies-live-map"
TAG_PREFIX = "prediction-log-"
ASSET_RE = re.compile(r"^snapshot_\d{8}T\d{4}Z\.tar\.gz$")
USER_AGENT = "shared-skies-prediction-log/1"

TRACT_FIELDS = ["geoid", "pm25", "pm25_model", "category"]
MONITOR_FIELDS = ["id", "name", "lat", "lon", "pm25", "hours",
                  "map_pm25", "map_pm25_loo", "model_pm25"]
SENSOR_FIELDS = ["lat", "lon", "pm25_raw", "pm25"]
SNAPSHOT_FIELDS = ["snapshot_time", "snapshot", "fetched_at", "health_status",
                   "calibration_applied", "calibration_method", "monitors_used",
                   "loo_mae", "loo_bias", "uncalibrated_mae", "uncalibrated_bias",
                   "tract_count", "monitor_rows", "sensor_rows", "value_semantics"]


def _token() -> str | None:
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or None


def _headers(accept: str) -> dict:
    h = {"Accept": accept, "User-Agent": USER_AGENT,
         "X-GitHub-Api-Version": "2022-11-28"}
    token = _token()
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


class _NoAuthAcrossHosts(urllib.request.HTTPRedirectHandler):
    """Asset downloads redirect to a signed URL on another host; urllib would
    forward the GitHub token there. Drop it when the host changes."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and (urllib.parse.urlsplit(newurl).hostname
                                != urllib.parse.urlsplit(req.full_url).hostname):
            new.remove_header("Authorization")
        return new


_OPENER = urllib.request.build_opener(_NoAuthAcrossHosts)


def _get(url: str, accept: str = "application/vnd.github+json") -> bytes:
    req = urllib.request.Request(url, headers=_headers(accept))
    with _OPENER.open(req, timeout=120) as resp:
        return resp.read()


def list_assets(repo: str) -> list[dict]:
    """All snapshot assets of prediction-log-* releases, oldest first."""
    assets = []
    page = 1
    while True:
        url = f"https://api.github.com/repos/{repo}/releases?per_page=100&page={page}"
        releases = json.loads(_get(url))
        if not releases:
            break
        for rel in releases:
            if not (rel.get("tag_name") or "").startswith(TAG_PREFIX):
                continue
            for a in rel.get("assets") or []:
                if ASSET_RE.match(a.get("name") or ""):
                    assets.append({"tag": rel["tag_name"], "name": a["name"],
                                   "size": a.get("size"), "api_url": a["url"],
                                   "browser_url": a.get("browser_download_url")})
        if len(releases) < 100:
            break
        page += 1
    assets.sort(key=lambda a: a["name"])
    return assets


def download(asset: dict, cache_dir: str) -> str:
    path = os.path.join(cache_dir, asset["name"])
    if os.path.exists(path) and (asset.get("size") is None
                                 or os.path.getsize(path) == asset["size"]):
        return path
    # Unauthenticated API asset downloads each spend one of the 60 API
    # requests an hour, so a month of snapshots (up to 248) would stall. The
    # public download URL has no such limit. With a token, use the API URL
    # (5,000 an hour, and it works for a private repo).
    if _token() or not asset.get("browser_url"):
        data = _get(asset["api_url"], accept="application/octet-stream")
    else:
        data = _get(asset["browser_url"], accept="application/octet-stream")
    tmp = path + ".part"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)
    return path


def read_snapshot(path: str) -> dict[str, bytes]:
    """Members by name. Only regular files with the expected names are read;
    nothing is extracted to disk."""
    wanted = {"tracts.csv", "monitors.csv", "sensors.csv", "meta.json"}
    out = {}
    with tarfile.open(path, "r:gz") as tar:
        for m in tar.getmembers():
            if m.isfile() and m.name in wanted:
                fh = tar.extractfile(m)
                if fh is not None:
                    out[m.name] = fh.read()
    return out


def _rows(data: bytes | None) -> list[dict]:
    if not data:
        return []
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"))))


def assemble(paths: list[str], out_dir: str) -> dict[str, int]:
    writers = {}
    files = []
    specs = {"tracts.csv": TRACT_FIELDS, "monitors.csv": MONITOR_FIELDS,
             "sensors.csv": SENSOR_FIELDS}
    for name, fields in specs.items():
        fh = open(os.path.join(out_dir, name), "w", newline="", encoding="utf-8")
        files.append(fh)
        w = csv.DictWriter(fh, fieldnames=["snapshot_time", *fields],
                           extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        writers[name] = w
    snap_fh = open(os.path.join(out_dir, "snapshots.csv"), "w", newline="", encoding="utf-8")
    files.append(snap_fh)
    snap_w = csv.DictWriter(snap_fh, fieldnames=SNAPSHOT_FIELDS, lineterminator="\n")
    snap_w.writeheader()

    counts = {name: 0 for name in specs}
    counts["snapshots"] = 0
    try:
        for path in paths:
            try:
                members = read_snapshot(path)
                meta = json.loads(members.get("meta.json") or b"{}")
            except (tarfile.TarError, OSError, ValueError) as e:
                print(f"warning: skipping unreadable {os.path.basename(path)}: {e}",
                      file=sys.stderr)
                continue
            t = meta.get("generated_at") or meta.get("snapshot") or os.path.basename(path)
            n = {}
            for name in specs:
                rows = _rows(members.get(name))
                for r in rows:
                    r["snapshot_time"] = t
                    writers[name].writerow(r)
                n[name] = len(rows)
                counts[name] += len(rows)
            cal = meta.get("calibration") or {}
            loo = cal.get("loo") or {}
            unc = cal.get("uncalibrated") or {}
            snap_w.writerow({
                "snapshot_time": t,
                "snapshot": meta.get("snapshot"),
                "fetched_at": meta.get("fetched_at"),
                "health_status": (meta.get("health") or {}).get("status"),
                "calibration_applied": cal.get("applied"),
                "calibration_method": cal.get("method"),
                "monitors_used": cal.get("monitors_used"),
                "loo_mae": loo.get("mae"), "loo_bias": loo.get("bias"),
                "uncalibrated_mae": unc.get("mae"), "uncalibrated_bias": unc.get("bias"),
                "tract_count": n["tracts.csv"],
                "monitor_rows": n["monitors.csv"] if "monitors.csv" in members else None,
                "sensor_rows": n["sensors.csv"] if "sensors.csv" in members else None,
                "value_semantics": meta.get("value_semantics"),
            })
            counts["snapshots"] += 1
    finally:
        for fh in files:
            fh.close()
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", default=DEFAULT_REPO, help="owner/name holding the releases")
    ap.add_argument("--out-dir", default="prediction_log", help="where merged CSVs go")
    ap.add_argument("--list", action="store_true", help="only list available snapshots")
    ap.add_argument("--local-dir",
                    help="merge snapshot tarballs from this folder instead of downloading")
    args = ap.parse_args(argv)

    failed = 0
    if args.local_dir:
        paths = sorted(os.path.join(args.local_dir, n) for n in os.listdir(args.local_dir)
                       if ASSET_RE.match(n))
        print(f"{len(paths)} local snapshot(s) in {args.local_dir}")
    else:
        try:
            assets = list_assets(args.repo)
        except urllib.error.HTTPError as e:
            print(f"error: GitHub API {e.code} listing releases of {args.repo}", file=sys.stderr)
            return 1
        tags = sorted({a["tag"] for a in assets})
        print(f"{len(assets)} snapshot(s) in {len(tags)} release(s) of {args.repo}")
        if args.list:
            for a in assets:
                print(f"  {a['tag']}  {a['name']}  {a['size']} bytes")
            return 0
        cache = os.path.join(args.out_dir, "snapshots")
        os.makedirs(cache, exist_ok=True)
        paths = []
        for i, a in enumerate(assets, 1):
            try:
                paths.append(download(a, cache))
            except (urllib.error.URLError, OSError) as e:
                failed += 1
                print(f"warning: could not download {a['name']}: {e}", file=sys.stderr)
            if i % 50 == 0:
                print(f"  downloaded {i}/{len(assets)}")

    os.makedirs(args.out_dir, exist_ok=True)
    counts = assemble(paths, args.out_dir)
    print(f"Wrote {args.out_dir}/: {counts['snapshots']} snapshots, "
          f"{counts['tracts.csv']} tract rows, {counts['monitors.csv']} monitor rows, "
          f"{counts['sensors.csv']} sensor rows")
    if failed:
        print(f"error: {failed} snapshot(s) failed to download, so the merged CSVs are "
              "incomplete; re-run to fetch them (downloaded ones are cached)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

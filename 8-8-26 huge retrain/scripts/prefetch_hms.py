"""Prefetch NOAA HMS smoke shapefile zips for 2021-01-01 .. 2026-08-07 into the
staged repo's HMS cache, so the post-download run of 10_build_hms_history.py is
compute-only. Same cache layout and empty-marker convention as production.
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

REPO = os.path.expanduser("~/scratch/livemap_retrain/repo")
sys.path.insert(0, os.path.join(REPO, "pipeline"))
CACHE_DIR = os.path.join(REPO, "pipeline", "data_pull_cache", "hms")
HMS_BASE_URL = ("https://satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/"
                "Smoke_Polygons/Shapefile")

import httpx


def cache_path(yyyymmdd):
    d = os.path.join(CACHE_DIR, yyyymmdd[:4])
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "hms_smoke%s.zip" % yyyymmdd)


def download_one(yyyymmdd):
    cp = cache_path(yyyymmdd)
    if os.path.exists(cp):
        return "cached"
    url = "%s/%s/%s/hms_smoke%s.zip" % (HMS_BASE_URL, yyyymmdd[:4],
                                        yyyymmdd[4:6], yyyymmdd)
    try:
        r = httpx.get(url, timeout=60.0)
        if r.status_code != 200 or not r.content:
            with open(cp, "wb") as f:
                f.write(b"")
            return "missing"
        with open(cp, "wb") as f:
            f.write(r.content)
        return "ok"
    except Exception as e:
        print("[hms-prefetch] %s error: %s" % (yyyymmdd, e), flush=True)
        return "error"


def main():
    dates = pd.date_range("2021-01-01", "2026-08-07", freq="D")
    labels = [d.strftime("%Y%m%d") for d in dates]
    print("[hms-prefetch] %d days" % len(labels), flush=True)
    counts = {}
    done = 0
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(download_one, lbl): lbl for lbl in labels}
        for fut in as_completed(futs):
            st = fut.result()
            counts[st] = counts.get(st, 0) + 1
            done += 1
            if done % 200 == 0:
                print("[hms-prefetch] %d/%d %s" % (done, len(labels), counts),
                      flush=True)
    print("[hms-prefetch] complete: %s" % counts, flush=True)
    if counts.get("error", 0) > len(labels) * 0.05:
        raise SystemExit("too many download errors")


if __name__ == "__main__":
    main()

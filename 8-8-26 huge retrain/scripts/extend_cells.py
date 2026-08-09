"""Rebuild the 0.5-degree grid-cell covariate parquets for the huge retrain.

Mirrors pipeline/11_pull_airquality.py (CAMS via Open-Meteo air-quality API,
hourly aerosol_optical_depth/pm2_5/dust collapsed to plain UTC-day means) and
pipeline/12_pull_met_extra.py (ERA5 daily shortwave/et0/cloud_cover). Cell set =
cells of the existing parquets (covers the old sensor network) union cells of
the pa_v4 TX sensors. Full date range pulled fresh for every cell so the new
Aug-2026 tail is covered. Batch JSONs cached; resumable.

Outputs (staged repo): pipeline/airquality_by_cell.parquet
                       pipeline/met_extra_by_cell.parquet
"""
import json
import os
import time
import urllib.parse
import urllib.request

import pandas as pd

BASE = os.path.expanduser("~/scratch/livemap_retrain")
REPO = os.path.join(BASE, "repo")
SEL = os.path.expanduser("~/scratch/aqnet/pa_selection.parquet")
CACHE = os.path.join(BASE, "cells_cache")
GRID_DEG = 0.5
BATCH = 20
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
MET_URL = "https://archive-api.open-meteo.com/v1/archive"
AQ_START = "2022-08-03"
MET_START = "2021-01-01"
END = "2026-08-07"


def cell_key(lat, lon):
    return (round(lat / GRID_DEG) * GRID_DEG, round(lon / GRID_DEG) * GRID_DEG)


def get_json(url, params, timeout=180):
    q = urllib.parse.urlencode(params)
    req = urllib.request.Request(url + "?" + q,
                                 headers={"User-Agent": "shared-skies-retrain"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def fetch_batch(url, cells, params_extra, cache_file):
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            j = json.load(f)
        if isinstance(j, list):
            return j
    params = dict(params_extra)
    params["latitude"] = ",".join("%.4f" % c[0] for c in cells)
    params["longitude"] = ",".join("%.4f" % c[1] for c in cells)
    for attempt in range(40):
        try:
            j = get_json(url, params)
            if isinstance(j, dict) and j.get("error"):
                time.sleep(60 if attempt < 10 else 300)
                continue
            if isinstance(j, dict):
                j = [j]
            with open(cache_file, "w") as f:
                json.dump(j, f)
            time.sleep(2.0)
            return j
        except Exception as e:
            code = getattr(e, "code", None)
            time.sleep(60 if code == 429 and attempt < 10
                       else 300 if code == 429 else 15)
    raise SystemExit("batch failed after 40 attempts: %s" % cache_file)


def daily_mean(times, values):
    s = pd.Series(values, index=pd.to_datetime(times))
    return s.groupby(s.index.normalize()).mean()


def target_cells():
    cells = set()
    for name in ("airquality_by_cell.parquet", "met_extra_by_cell.parquet"):
        p = os.path.join(REPO, "pipeline", name)
        if os.path.exists(p):
            d = pd.read_parquet(p)
            cells |= set(zip(d.cell_lat.round(4), d.cell_lon.round(4)))
    sel = pd.read_parquet(SEL)
    tx = sel[sel.box == "TX"]
    for la, lo in zip(tx.latitude, tx.longitude):
        c = cell_key(float(la), float(lo))
        cells.add((round(c[0], 4), round(c[1], 4)))
    return sorted(cells)


def main():
    os.makedirs(os.path.join(CACHE, "aq"), exist_ok=True)
    os.makedirs(os.path.join(CACHE, "met"), exist_ok=True)
    cells = target_cells()
    print("[cells] %d target cells" % len(cells), flush=True)

    aq_rows, met_rows = [], []
    for b in range(0, len(cells), BATCH):
        chunk = cells[b:b + BATCH]
        i = b // BATCH

        j = fetch_batch(AQ_URL, chunk, {
            "hourly": "aerosol_optical_depth,pm2_5,dust",
            "start_date": AQ_START, "end_date": END, "timezone": "UTC",
        }, os.path.join(CACHE, "aq", "batch_%03d.json" % i))
        for c, r in zip(chunk, j):
            h = r.get("hourly", {})
            t = h.get("time", [])
            if not t:
                continue
            aod = daily_mean(t, h.get("aerosol_optical_depth", []))
            pm = daily_mean(t, h.get("pm2_5", []))
            du = daily_mean(t, h.get("dust", []))
            for d in aod.index:
                aq_rows.append((c[0], c[1], d, aod.get(d), pm.get(d),
                                du.get(d)))
        print("[cells] aq batch %d done (%d rows)" % (i, len(aq_rows)),
              flush=True)

        j = fetch_batch(MET_URL, chunk, {
            "daily": ("shortwave_radiation_sum,et0_fao_evapotranspiration,"
                      "cloud_cover_mean"),
            "start_date": MET_START, "end_date": END, "timezone": "UTC",
        }, os.path.join(CACHE, "met", "batch_%03d.json" % i))
        for c, r in zip(chunk, j):
            d = r.get("daily", {})
            t = d.get("time", [])
            sw = d.get("shortwave_radiation_sum", [])
            et = d.get("et0_fao_evapotranspiration", [])
            cc = d.get("cloud_cover_mean", [])
            for k, day in enumerate(t):
                met_rows.append((c[0], c[1], pd.Timestamp(day),
                                 sw[k] if k < len(sw) else None,
                                 et[k] if k < len(et) else None,
                                 cc[k] if k < len(cc) else None))
        print("[cells] met batch %d done (%d rows)" % (i, len(met_rows)),
              flush=True)

    aq = pd.DataFrame(aq_rows, columns=["cell_lat", "cell_lon", "date", "aod",
                                        "cams_pm25", "dust"])
    aq["date"] = pd.to_datetime(aq["date"]).dt.normalize()
    aq.to_parquet(os.path.join(REPO, "pipeline", "airquality_by_cell.parquet"),
                  index=False)
    met = pd.DataFrame(met_rows, columns=["cell_lat", "cell_lon", "date",
                                          "shortwave", "et0", "cloud_cover"])
    met["date"] = pd.to_datetime(met["date"]).dt.normalize()
    met.to_parquet(os.path.join(REPO, "pipeline", "met_extra_by_cell.parquet"),
                   index=False)
    print("[cells] wrote %d aq rows, %d met rows" % (len(aq), len(met)),
          flush=True)


if __name__ == "__main__":
    main()

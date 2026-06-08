"""Pull historical CAMS air-quality (AOD, PM2.5, dust) on a 0.5deg grid.

CAMS is ~40km resolution, so pulling per-sensor wastes the rate-limit quota —
a 0.5deg grid (~55km, ~108 cells over TX) carries identical information at ~4x
lower cost AND matches the inference grid in backend/airquality.py exactly.

Output: pipeline/airquality_by_cell.parquet [cell_lat, cell_lon, date, aod,
cams_pm25, dust]. The training merge maps each sensor to its cell.

Robust to Open-Meteo's weighted quota: backs off and waits (never skips) so the
pull completes even if it spans the hourly/daily limit. Caches per batch.

Run:  python pipeline/11_pull_airquality.py
"""
import os, time, json, math
import numpy as np
import pandas as pd
import httpx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(ROOT, "pipeline", "purpleair_full_dataset.parquet")
CACHE_DIR = os.path.join(ROOT, "pipeline", "data_pull_cache", "airquality_grid")
OUT = os.path.join(ROOT, "pipeline", "airquality_by_cell.parquet")
URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
ARCHIVE_START = "2022-08-03"
GRID_DEG = 0.5
BATCH = 30
os.makedirs(CACHE_DIR, exist_ok=True)


def cell_key(lat, lon):
    return (round(lat / GRID_DEG) * GRID_DEG, round(lon / GRID_DEG) * GRID_DEG)


def daily_mean(times, values):
    if not values:
        return {}
    s = pd.Series(values, index=pd.to_datetime(times))
    d = s.groupby(s.index.normalize()).mean()
    return {k.strftime("%Y-%m-%d"): (None if pd.isna(v) else float(v)) for k, v in d.items()}


def fetch_batch(b, cells, end_date):
    cache = os.path.join(CACHE_DIR, f"batch_{b:03d}.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    params = {
        "latitude": ",".join(f"{c[0]:.4f}" for c in cells),
        "longitude": ",".join(f"{c[1]:.4f}" for c in cells),
        "hourly": "aerosol_optical_depth,pm2_5,dust",
        "start_date": ARCHIVE_START, "end_date": end_date, "timezone": "UTC",
    }
    for attempt in range(40):
        try:
            r = httpx.get(URL, params=params, timeout=180)
            if r.status_code == 429:
                wait = 60 if attempt < 10 else 300  # wait out hourly, then daily
                print(f"  batch {b}: 429, waiting {wait}s (attempt {attempt})")
                time.sleep(wait); continue
            r.raise_for_status()
            payload = r.json()
            if isinstance(payload, dict):
                payload = [payload]
            with open(cache, "w") as f:
                json.dump(payload, f)
            time.sleep(2.0)
            return payload
        except Exception as e:
            print(f"  batch {b} attempt {attempt}: {e}")
            time.sleep(15)
    return None


def main():
    df = pd.read_parquet(DATASET, columns=["latitude", "longitude", "date"])
    end_date = pd.to_datetime(df["date"]).max().strftime("%Y-%m-%d")
    cells = sorted({cell_key(la, lo) for la, lo in zip(df["latitude"], df["longitude"])})
    print(f"{len(cells)} grid cells, archive {ARCHIVE_START}..{end_date}, batch={BATCH}")

    rows = []
    n_batches = math.ceil(len(cells) / BATCH)
    for b in range(n_batches):
        chunk = cells[b*BATCH:(b+1)*BATCH]
        payload = fetch_batch(b, chunk, end_date)
        if not payload:
            print(f"  batch {b}/{n_batches} FAILED")
            continue
        for cell, loc in zip(chunk, payload):
            h = loc.get("hourly", {})
            t = h.get("time", [])
            a = daily_mean(t, h.get("aerosol_optical_depth", []))
            p = daily_mean(t, h.get("pm2_5", []))
            du = daily_mean(t, h.get("dust", []))
            for d in set(a) | set(p) | set(du):
                rows.append((cell[0], cell[1], d, a.get(d), p.get(d), du.get(d)))
        print(f"  batch {b+1}/{n_batches} done, {len(rows):,} rows")

    aq = pd.DataFrame(rows, columns=["cell_lat", "cell_lon", "date", "aod", "cams_pm25", "dust"])
    aq["date"] = pd.to_datetime(aq["date"]).dt.normalize()
    aq.to_parquet(OUT, index=False)
    print(f"\nSaved {OUT}: {len(aq):,} cell-day rows")
    for c in ["aod", "cams_pm25", "dust"]:
        nn = aq[c].notna().sum()
        print(f"  {c}: {nn:,} non-null, mean={aq[c].mean():.3f}, max={aq[c].max():.2f}")


if __name__ == "__main__":
    main()

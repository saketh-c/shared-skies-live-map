"""Pull extra keyless met PBL-proxy features on a 0.5deg grid (ERA5 archive).

Daily mixing/dilution proxies for boundary-layer height (no keyless historical
archive): shortwave_radiation_sum, et0_fao_evapotranspiration, cloud_cover_mean.
Grid-based (matches backend/airquality.py inference grid). Full 2021+ history.

Output: pipeline/met_extra_by_cell.parquet [cell_lat, cell_lon, date, shortwave,
et0, cloud_cover].

Run:  python pipeline/12_pull_met_extra.py
"""
import os, time, json, math
import numpy as np
import pandas as pd
import httpx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(ROOT, "pipeline", "purpleair_full_dataset.parquet")
CACHE_DIR = os.path.join(ROOT, "pipeline", "data_pull_cache", "met_extra_grid")
OUT = os.path.join(ROOT, "pipeline", "met_extra_by_cell.parquet")
URL = "https://archive-api.open-meteo.com/v1/archive"
GRID_DEG = 0.5
BATCH = 30
DAILY = "shortwave_radiation_sum,et0_fao_evapotranspiration,cloud_cover_mean"
os.makedirs(CACHE_DIR, exist_ok=True)


def cell_key(lat, lon):
    return (round(lat / GRID_DEG) * GRID_DEG, round(lon / GRID_DEG) * GRID_DEG)


def fetch_batch(b, cells, start, end):
    cache = os.path.join(CACHE_DIR, f"batch_{b:03d}.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    params = {
        "latitude": ",".join(f"{c[0]:.4f}" for c in cells),
        "longitude": ",".join(f"{c[1]:.4f}" for c in cells),
        "daily": DAILY, "start_date": start, "end_date": end, "timezone": "UTC",
    }
    for attempt in range(40):
        try:
            r = httpx.get(URL, params=params, timeout=180)
            if r.status_code == 429:
                wait = 60 if attempt < 10 else 300
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
    start = pd.to_datetime(df["date"]).min().strftime("%Y-%m-%d")
    end = pd.to_datetime(df["date"]).max().strftime("%Y-%m-%d")
    cells = sorted({cell_key(la, lo) for la, lo in zip(df["latitude"], df["longitude"])})
    print(f"{len(cells)} grid cells, archive {start}..{end}, batch={BATCH}")

    rows = []
    n_batches = math.ceil(len(cells) / BATCH)
    for b in range(n_batches):
        chunk = cells[b*BATCH:(b+1)*BATCH]
        payload = fetch_batch(b, chunk, start, end)
        if not payload:
            print(f"  batch {b}/{n_batches} FAILED")
            continue
        for cell, loc in zip(chunk, payload):
            d = loc.get("daily", {})
            t = d.get("time", [])
            sw = d.get("shortwave_radiation_sum", [])
            et = d.get("et0_fao_evapotranspiration", [])
            cc = d.get("cloud_cover_mean", [])
            for i, day in enumerate(t):
                rows.append((cell[0], cell[1], day,
                             sw[i] if i < len(sw) else None,
                             et[i] if i < len(et) else None,
                             cc[i] if i < len(cc) else None))
        print(f"  batch {b+1}/{n_batches} done, {len(rows):,} rows")

    met = pd.DataFrame(rows, columns=["cell_lat", "cell_lon", "date", "shortwave", "et0", "cloud_cover"])
    met["date"] = pd.to_datetime(met["date"]).dt.normalize()
    met.to_parquet(OUT, index=False)
    print(f"\nSaved {OUT}: {len(met):,} cell-day rows")
    for c in ["shortwave", "et0", "cloud_cover"]:
        nn = met[c].notna().sum()
        print(f"  {c}: {nn:,} non-null, mean={met[c].mean():.3f}")


if __name__ == "__main__":
    main()

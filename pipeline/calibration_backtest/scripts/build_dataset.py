"""Historical monitor-day dataset: EPA AQS truth + raw production-model value at
each monitor's exact coordinates, computed with the PRODUCTION code path.

usage: python -I build_dataset.py <repo> <monitor_truth.parquet> <out monitor_days.parquet>

Per UTC day d with >= 50 PurpleAir sensors in the pool:
  pool          all p2_processed_v3 sensors that day (in and out of Texas),
                minus the legacy 2025 rows whose pm25 is not raw ATM (README
                'Mixed PM2.5 basis'); live serving pools raw ATM only.
  neighbors     backend.purpleair.compute_neighbor_features(monitor pts, pool)
  dist_to_nearest_sensor  haversine to models/training_sensors.json, clipped
  dist_to_coast           M._min_dist_to_points(.., M.TX_COAST_POINTS)
  weather       humidity/temperature/pressure/wind_speed/precipitation and the
                two interactions copied from the nearest pool sensor that day
  hms_smoke     nearest pool sensor's (sensor_id, date) HMS tier, else 0
  aod/cams_pm25 0.5 deg cell (training rounding) and date, else NaN (median fill)
  calendar      M.get_temporal(d 12:00 UTC) -> M._compute_v3_shared, per row
One batched M.run_predictions_batch over all rows. A second batch uses the
serve-time outlier-QC pool (report only) for model_qc.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

REPO = sys.argv[1]
TRUTH = sys.argv[2]
OUT = sys.argv[3]
sys.path.insert(0, REPO)
os.chdir(REPO)

import backend.main as M  # noqa: E402
from backend.purpleair import compute_neighbor_features  # noqa: E402

MIN_POOL = 50
START = pd.Timestamp("2023-01-01")
QC_RADIUS_KM = 50.0
QC_MIN_OTHERS = 3
WEATHER_COLS = ["humidity", "temperature", "pressure", "wind_speed", "precipitation",
                "temp_x_humidity", "wind_x_temp"]
TEMPORAL_COLS = ["month", "dow", "day_of_year", "month_sin", "month_cos",
                 "dow_sin", "dow_cos", "doy_sin", "doy_cos"]

REGION_ANCHORS = [
    ("El Paso/far West", 31.76, -106.49), ("El Paso/far West", 31.85, -102.37),
    ("El Paso/far West", 29.30, -103.20),
    ("Panhandle/South Plains", 35.20, -101.83), ("Panhandle/South Plains", 33.58, -101.85),
    ("DFW", 32.78, -96.80), ("DFW", 32.75, -97.33),
    ("Houston/Gulf", 29.76, -95.37), ("Houston/Gulf", 30.08, -94.10), ("Houston/Gulf", 29.30, -94.79),
    ("Austin/San Antonio/Central", 30.27, -97.74), ("Austin/San Antonio/Central", 29.42, -98.49),
    ("Austin/San Antonio/Central", 31.55, -97.15), ("Austin/San Antonio/Central", 31.46, -100.44),
    ("East Texas", 32.35, -95.30), ("East Texas", 32.50, -94.74), ("East Texas", 33.43, -94.05),
    ("South Texas/RGV", 27.80, -97.40), ("South Texas/RGV", 26.20, -98.23),
    ("South Texas/RGV", 25.90, -97.50), ("South Texas/RGV", 27.53, -99.49),
]


def hav(lat1, lon1, lat2, lon2):
    return M._haversine_km_np(lat1, lon1, lat2, lon2)


def region_of(lat, lon, state):
    if state != "Texas":
        return "Outside Texas"
    d = [hav(np.array([lat]), np.array([lon]), a, b)[0] for _, a, b in REGION_ANCHORS]
    return REGION_ANCHORS[int(np.argmin(d))][0]


def qc_mask(lat, lon, pm):
    """True = keep. Drop a sensor whose value > max(3 x median of others within
    50 km, that median + 20), when it has >= 3 others within 50 km."""
    D = hav(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    np.fill_diagonal(D, np.inf)
    keep = np.ones(len(pm), dtype=bool)
    for i in range(len(pm)):
        nb = D[i] <= QC_RADIUS_KM
        if nb.sum() >= QC_MIN_OTHERS:
            med = float(np.median(pm[nb]))
            if pm[i] > max(3.0 * med, med + 20.0):
                keep[i] = False
    return keep


def main():
    t0 = time.time()
    truth = pd.read_parquet(TRUTH)
    truth["date"] = pd.to_datetime(truth["date"]).dt.normalize()
    truth = truth[truth["date"] >= START]

    pa = pd.read_parquet(os.path.join(REPO, "8-8-26 huge retrain/dataset/p2_processed_v3.parquet"))
    pa["date"] = pd.to_datetime(pa["date"]).dt.normalize()
    pa = pa[pa["date"] >= START].copy()
    # Legacy 2025 rows (p2_processed.xls copies, ~0.54 x ATM + 1.6): flag by
    # exact (sensor_id, date, pm25) match within the old part (city is None).
    leg = pd.read_csv(os.path.join(REPO, "p2_processed.xls"), encoding="utf-8-sig",
                      low_memory=False, usecols=["sensor_id", "date", "pm25"])
    leg["date"] = pd.to_datetime(leg["date"], errors="coerce").dt.normalize()
    leg = leg.dropna().rename(columns={"pm25": "pm_leg"})
    pa = pa.merge(leg, on=["sensor_id", "date"], how="left")
    is_leg = pa["city"].isna() & pa["pm_leg"].notna() & ((pa["pm25"] - pa["pm_leg"]).abs() < 1e-9)
    print(f"pool rows since {START.date()}: {len(pa):,}; legacy non-ATM rows dropped: {int(is_leg.sum()):,}")
    pa = pa[~is_leg].drop(columns=["pm_leg"])
    pa = pa.dropna(subset=["pm25", "latitude", "longitude"])

    hms = pd.read_parquet(os.path.join(REPO, "pipeline/hms_smoke_by_sensor.parquet"))
    hms["date"] = pd.to_datetime(hms["date"]).dt.normalize()
    hms_map = {(str(s), d): int(v) for s, d, v in zip(hms["sensor_id"], hms["date"], hms["hms_smoke"])}
    aq = pd.read_parquet(os.path.join(REPO, "pipeline/airquality_by_cell.parquet"))
    aq["date"] = pd.to_datetime(aq["date"]).dt.normalize()
    aq = aq.set_index(["cell_lat", "cell_lon", "date"])[["aod", "cams_pm25"]]
    hms_last = hms["date"].max()
    aq_last = aq.index.get_level_values("date").max()

    with open(M.TRAINING_SENSORS_PATH) as f:
        ts = json.load(f)
    ts_lat = np.array([s["lat"] for s in ts], float)
    ts_lon = np.array([s["lon"] for s in ts], float)

    sites = truth.drop_duplicates("site_id").set_index("site_id")
    site_region = {sid: region_of(r.lat, r.lon, r.state) for sid, r in sites.iterrows()}

    rows = []
    pools_by_day = {}
    skipped_small = 0
    by_day_truth = dict(tuple(truth.groupby("date")))
    for d, pday in pa.groupby("date"):
        if d not in by_day_truth:
            continue
        n_pool = len(pday)
        if n_pool < MIN_POOL:
            skipped_small += 1
            continue
        tday = by_day_truth[d]
        mlat = tday["lat"].to_numpy(float)
        mlon = tday["lon"].to_numpy(float)
        plat = pday["latitude"].to_numpy(float)
        plon = pday["longitude"].to_numpy(float)
        ppm = pday["pm25"].to_numpy(float)
        pool = [{"lat": a, "lon": b, "pm25": c} for a, b, c in zip(plat, plon, ppm)]
        nb = compute_neighbor_features(mlat, mlon, pool)
        keep = qc_mask(plat, plon, ppm)
        pool_qc = [p for p, k in zip(pool, keep) if k]
        nb_qc = compute_neighbor_features(mlat, mlon, pool_qc)

        D = hav(mlat[:, None], mlon[:, None], plat[None, :], plon[None, :])
        j = np.argmin(D, axis=1)
        near_km = D[np.arange(len(mlat)), j]
        sid = pday["sensor_id"].to_numpy()
        temporal = M.get_temporal(datetime(d.year, d.month, d.day, 12, tzinfo=timezone.utc))
        shared = M._compute_v3_shared({}, temporal)

        df = pd.DataFrame({
            "site_id": tday["site_id"].to_numpy(), "date": d,
            "lat": mlat, "lon": mlon, "truth": tday["truth"].to_numpy(float),
            "n_hours": tday["n_hours"].to_numpy(), "n_pa": n_pool,
            "n_pa_qc_dropped": int((~keep).sum()), "nearest_pa_km": near_km,
        })
        for c in WEATHER_COLS:
            df[c] = pday[c].to_numpy(float)[j]
        df["rh"] = df["humidity"]
        df["hms_smoke"] = [hms_map.get((str(sid[k]), d), 0) for k in j]
        cl = (pd.Series(mlat) / 0.5).round() * 0.5
        cn = (pd.Series(mlon) / 0.5).round() * 0.5
        idx = pd.MultiIndex.from_arrays([cl.to_numpy(), cn.to_numpy(), [d] * len(mlat)])
        aqv = aq.reindex(idx)
        df["aod"] = aqv["aod"].to_numpy()
        df["cams_pm25"] = aqv["cams_pm25"].to_numpy()
        tsd = np.min(hav(mlat[:, None], mlon[:, None], ts_lat[None, :], ts_lon[None, :]), axis=1)
        df["dist_to_nearest_sensor"] = np.clip(tsd, 0.0, M.DIST_TO_SENSOR_MAX_KM)
        df["dist_to_coast"] = M._min_dist_to_points(mlat, mlon, M.TX_COAST_POINTS)
        for c in TEMPORAL_COLS:
            df[c] = float(shared[c])
        for k, v in nb.items():
            df[k] = v
        for k, v in nb_qc.items():
            df[k + "__qc"] = v
        rows.append(df)
        pools_by_day[d] = n_pool
    data = pd.concat(rows, ignore_index=True)
    print(f"days used {data.date.nunique()} (skipped <{MIN_POOL} sensors: {skipped_small}); rows {len(data):,}")

    # One batch, production code path. Every model feature is a per-row column
    # here, so the shared weather/temporal scalars only matter for NaN fills.
    M.state["bundle"] = M._load_bundle()
    feats = M.state["bundle"]["feature_names"]
    missing = [f for f in feats if f not in data.columns and f not in ("latitude", "longitude")]
    print("features not supplied per row (training fill):", missing)
    wfb = M._weather_fallback()
    tmp0 = M.get_temporal(datetime(2024, 6, 15, 12, tzinfo=timezone.utc))
    t1 = time.time()
    X = data[["lat", "lon"] + [f for f in feats if f in data.columns]].copy()
    data["model"] = M.run_predictions_batch(X, wfb, tmp0)
    Xq = X.copy()
    for k in nb.keys():
        Xq[k] = data[k + "__qc"].to_numpy()
    data["model_qc"] = M.run_predictions_batch(Xq, wfb, tmp0)
    print(f"batch predictions: {2 * len(data):,} rows in {time.time() - t1:.1f}s")

    # Equivalence check: per-day calls with temporal=get_temporal(day 12:00)
    # and NO per-row calendar columns must reproduce the batch exactly.
    rng = np.random.default_rng(0)
    days = rng.choice(data["date"].unique(), size=8, replace=False)
    maxdiff = 0.0
    for d in days:
        sel = data["date"] == d
        Xd = X[sel.to_numpy()].drop(columns=TEMPORAL_COLS).copy()
        dd = pd.Timestamp(d)
        tmp = M.get_temporal(datetime(dd.year, dd.month, dd.day, 12, tzinfo=timezone.utc))
        p = M.run_predictions_batch(Xd, wfb, tmp)
        maxdiff = max(maxdiff, float(np.max(np.abs(p - data.loc[sel, "model"].to_numpy()))))
    print(f"per-day production call vs batch: max |diff| = {maxdiff:.2e} over {len(days)} days")
    assert maxdiff < 1e-9

    data["region"] = data["site_id"].map(site_region)
    for c in ["name", "city", "county", "state", "utc_offset_h", "frac_88101"]:
        if c in truth.columns:
            data = data.merge(truth[["site_id", "date", c]] if c == "frac_88101"
                              else sites[[c]].reset_index(), on=["site_id", "date"] if c == "frac_88101"
                              else ["site_id"], how="left")
    data = data.drop(columns=[c for c in data.columns if c.endswith("__qc")])
    data["hms_beyond_archive"] = data["date"] > hms_last
    data["aq_beyond_archive"] = data["date"] > aq_last
    data.to_parquet(OUT, index=False)
    print(f"saved {OUT}: {len(data):,} rows, {data.site_id.nunique()} sites, "
          f"{data.date.nunique()} days, {time.time() - t0:.0f}s total")
    print(f"aod coverage {data.aod.notna().mean():.3f}; hms archive ends {hms_last.date()}, "
          f"aq archive ends {aq_last.date()}")


if __name__ == "__main__":
    main()

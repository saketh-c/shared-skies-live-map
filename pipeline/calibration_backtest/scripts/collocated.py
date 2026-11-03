"""Does applying the calibration to RAW PurpleAir values (the live sensor dots)
make sense? Collocated pairs: each Texas monitor-day with a PurpleAir sensor
within 1.5 km. The calibration for that monitor-day is fitted WITHOUT the
monitor (leave-one-site-out, shipped fit()), then applied to the sensor's raw
daily mean at the sensor's coordinates.

usage: python -I collocated.py <repo> <monitor_days.parquet> <out.json>
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO, SRC, OUT = sys.argv[1:4]
sys.path.insert(0, REPO)
import bt_common as B  # noqa: E402
from backend import calibration as C  # noqa: E402

MAX_KM = 1.5


def main():
    d = B.load(SRC, "model")
    pa = pd.read_parquet(os.path.join(REPO, "8-8-26 huge retrain/dataset/p2_processed_v3.parquet"),
                         columns=["sensor_id", "date", "pm25", "latitude", "longitude", "city"])
    pa["date"] = pd.to_datetime(pa["date"]).dt.normalize()
    pa = pa[pa["date"] >= d["date"].min()]
    leg = pd.read_csv(os.path.join(REPO, "p2_processed.xls"), encoding="utf-8-sig",
                      low_memory=False, usecols=["sensor_id", "date", "pm25"])
    leg["date"] = pd.to_datetime(leg["date"], errors="coerce").dt.normalize()
    leg = leg.dropna().rename(columns={"pm25": "pm_leg"})
    pa = pa.merge(leg, on=["sensor_id", "date"], how="left")
    is_leg = pa["city"].isna() & pa["pm_leg"].notna() & ((pa["pm25"] - pa["pm_leg"]).abs() < 1e-9)
    pa = pa[~is_leg].dropna(subset=["pm25", "latitude", "longitude"])
    pools = dict(tuple(pa.groupby("date")))

    rows = []
    for day, g in d.groupby("date"):
        p = pools.get(day)
        if p is None:
            continue
        lat, lon = g["lat"].to_numpy(float), g["lon"].to_numpy(float)
        y, m = g["truth"].to_numpy(float), g["model"].to_numpy(float)
        Dp = B.hav(lat[:, None], lon[:, None], p["latitude"].to_numpy()[None, :],
                   p["longitude"].to_numpy()[None, :])
        j = Dp.argmin(1)
        near = Dp[np.arange(len(lat)), j] <= MAX_KM
        for i in np.where(near & g["is_tx"].to_numpy())[0]:
            use = np.arange(len(y)) != i
            cal = C.fit(lat[use], lon[use], y[use], m[use])
            if cal is None:
                continue
            s_lat, s_lon = float(p["latitude"].iloc[j[i]]), float(p["longitude"].iloc[j[i]])
            raw = float(p["pm25"].iloc[j[i]])
            rows.append({"date": day, "site_id": g["site_id"].iloc[i], "truth": y[i], "sensor_raw": raw,
                         "sensor_cal": float(cal.apply([s_lat], [s_lon], [raw])[0]),
                         "model": m[i], "model_cal": float(cal.apply([lat[i]], [lon[i]], [m[i]])[0]),
                         "period": "train" if day <= B.TRAIN_END else "test"})
    r = pd.DataFrame(rows)
    out = {"max_km": MAX_KM, "pairs": int(len(r)), "sites": int(r["site_id"].nunique())}
    for per in ("train", "test"):
        x = r[r["period"] == per]
        out[per] = {c: B.core(x[c], x["truth"]) for c in ("sensor_raw", "sensor_cal", "model", "model_cal")}
        out[per]["n_sites"] = int(x["site_id"].nunique())
    print(json.dumps(out, indent=1))
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()

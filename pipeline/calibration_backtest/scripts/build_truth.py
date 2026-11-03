"""EPA AQS hourly PM2.5 -> one UTC-day mean per monitor site.

usage: python -I build_truth.py <aqs_filtered_dir> <aqs_sites.zip> <out.parquet>

Rules (backtest protocol):
  * keep sites in Texas or inside the PurpleAir bbox (done by filter_aqs.py)
  * one value per site-hour: average the POCs of a parameter, and prefer 88101
    (FRM/FEM) over 88502 (non-regulatory continuous) when both exist that hour
  * the hour key is Date GMT + Time GMT (true UTC); UTC-day mean needs >= 18
    valid hours
"""
import glob
import os
import sys
import zipfile

import numpy as np
import pandas as pd

MIN_HOURS = 18


def main(src_dir, sites_zip, dst):
    parts = []
    for p in sorted(glob.glob(os.path.join(src_dir, "hourly_*.parquet"))):
        d = pd.read_parquet(p)
        parts.append(d)
        print(f"  {os.path.basename(p)}: {len(d):,} rows")
    h = pd.concat(parts, ignore_index=True)
    h["site_id"] = h["State Code"].str.zfill(2) + h["County Code"].str.zfill(3) + h["Site Num"].str.zfill(4)
    h["ts_utc"] = pd.to_datetime(h["Date GMT"] + " " + h["Time GMT"], format="%Y-%m-%d %H:%M")
    h["ts_local"] = pd.to_datetime(h["Date Local"] + " " + h["Time Local"], format="%Y-%m-%d %H:%M")
    off = (h["ts_utc"] - h["ts_local"]).dt.total_seconds() / 3600.0
    # UTC sanity: every site must have ONE constant offset (AQS local = standard time).
    per_site = off.groupby(h["site_id"]).agg(["min", "max"])
    bad = per_site[per_site["min"] != per_site["max"]]
    print(f"UTC check: offsets by site {sorted(off.unique().tolist())}; sites with a varying offset: {len(bad)}")
    h = h[np.isfinite(h["Sample Measurement"])]

    # POC average per (site, param, hour)
    g = (h.groupby(["site_id", "Parameter Code", "ts_utc"], as_index=False)
           .agg(val=("Sample Measurement", "mean"), npoc=("POC", "nunique")))
    p1 = g[g["Parameter Code"] == "88101"].set_index(["site_id", "ts_utc"])
    p2 = g[g["Parameter Code"] == "88502"].set_index(["site_id", "ts_utc"])
    p2 = p2[~p2.index.isin(p1.index)]  # prefer 88101 at a site-hour
    hv = pd.concat([p1, p2]).reset_index()
    print(f"site-hours: {len(hv):,} (88101 {len(p1):,}, 88502-only {len(p2):,})")

    hv["date"] = hv["ts_utc"].dt.normalize()
    day = (hv.groupby(["site_id", "date"], as_index=False)
             .agg(truth=("val", "mean"), n_hours=("val", "size"),
                  frac_88101=("Parameter Code", lambda s: float((s == "88101").mean()))))
    day = day[day["n_hours"] >= MIN_HOURS].reset_index(drop=True)

    coords = (h.groupby("site_id").agg(lat=("Latitude", "median"), lon=("Longitude", "median"),
                                        state=("State Name", "first"),
                                        utc_offset_h=("ts_utc", "size")))
    coords["utc_offset_h"] = off.groupby(h["site_id"]).median()
    with zipfile.ZipFile(sites_zip) as zf:
        with zf.open(zf.namelist()[0]) as fh:
            s = pd.read_csv(fh, dtype=str, usecols=["State Code", "County Code", "Site Number",
                                                    "Local Site Name", "City Name", "County Name"])
    s["site_id"] = s["State Code"].str.zfill(2) + s["County Code"].str.zfill(3) + s["Site Number"].str.zfill(4)
    s = s.drop_duplicates("site_id").set_index("site_id")
    coords = coords.join(s[["Local Site Name", "City Name", "County Name"]], how="left")
    coords = coords.rename(columns={"Local Site Name": "name", "City Name": "city", "County Name": "county"})
    day = day.merge(coords.reset_index(), on="site_id", how="left")
    day.to_parquet(dst, index=False)
    print(f"monitor-days: {len(day):,}; sites {day.site_id.nunique()}; "
          f"dates {day.date.min().date()}..{day.date.max().date()}")
    print(day.groupby(day.date.dt.year).agg(n=("truth", "size"), sites=("site_id", "nunique"),
                                            mean=("truth", "mean")))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])

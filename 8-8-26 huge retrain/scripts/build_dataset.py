"""Build the 8-8-26 huge-retrain training dataset for the Shared Skies live map.

Converts the AQNet v4 PurpleAir archival pull (pa_v4: per-sensor parquet files,
tier A at 6-hour averages, tier B at daily averages, fields pm2.5_cf_1_a/b,
pm2.5_atm_a/b, humidity, temperature) into the exact 41-column schema of
p2_processed_v2.xls produced by pipeline/08_finish_pull.py, then unions it with
the existing shipped dataset so sensors that have died since the v4 selection
(alive-only) keep their full history.

Production semantics mirrored exactly:
  - target pm25 = raw PurpleAir ATM channel, UTC-day daily mean, NO correction
    (here reconstructed as the row-wise mean of pm2.5_atm_a and pm2.5_atm_b,
    which is PurpleAir's own blend when both channels are healthy)
  - QC: dropna, 0 <= pm25 <= 200, per-sensor MAD z < 20, sensors >= 60 days
  - weather: per-sensor Open-Meteo archive (temperature_2m_mean,
    relative_humidity_2m_mean, surface_pressure_mean, wind_speed_10m_max,
    wind_gusts_10m_max, precipitation_sum; wind requested in m/s so NO /3.6),
    NASA POWER fallback (T2M, RH2M, PS kPa*10 -> hPa, WS10M_MAX, PRECTOTCORR)
  - GEOID: nearest tract centroid by haversine from tract_lookup.parquet
  - temporal features, season, city="", hour=12, identical column order

Outputs:
  $REPO/p2_processed_v2.xls                      (CSV; what training loads)
  $REPO/pipeline/purpleair_full_dataset.parquet  (same frame; HMS builder input)
  $BASE/build_report.json
"""
import glob
import json
import os

import numpy as np
import pandas as pd

BASE = os.path.expanduser("~/scratch/livemap_retrain")
REPO = os.path.join(BASE, "repo")
OLD_CSV = os.path.join(BASE, "old", "p2_processed_v2.xls")
PA_V4 = os.path.expanduser("~/scratch/aqnet/repo/research/aqnet2/data/pa_v4")
SEL = os.path.expanduser("~/scratch/aqnet/pa_selection.parquet")
TRACTS = os.path.join(REPO, "backend", "static", "tract_lookup.parquet")
WEATHER_DIR = os.path.join(BASE, "weather")

W0 = pd.Timestamp("2021-01-01")
W_LAST_FULL = pd.Timestamp("2026-08-07")   # skip the partial pull day
PM25_MIN, PM25_MAX = 0.0, 200.0
MAD_Z_MAX = 20.0
MIN_DAYS = 60

SEASON_MAP = {12: "Winter", 1: "Winter", 2: "Winter",
              3: "Spring", 4: "Spring", 5: "Spring",
              6: "Summer", 7: "Summer", 8: "Summer",
              9: "Fall", 10: "Fall", 11: "Fall"}

COLUMNS = [
    "sensor_id", "date", "pm25", "latitude", "longitude", "GEOID", "lat",
    "lon", "POPULATION", "STATE_NAME", "CNTY_NAME", "ejf_score",
    "pct_people_of_color", "pct_low_income", "traffic_proximity",
    "superfund_proximity", "rmp_proximity", "diesel_pm_proximity",
    "pct_ling_isolated", "population", "temperature", "humidity", "pressure",
    "wind_speed", "precipitation", "wind_gusts", "weather_source", "month",
    "dow", "day_of_year", "hour", "month_sin", "month_cos", "dow_sin",
    "dow_cos", "doy_sin", "doy_cos", "temp_x_humidity", "wind_x_temp",
    "season", "city",
]

OM_RENAME = {
    "temperature_2m_mean": "temperature",
    "relative_humidity_2m_mean": "humidity",
    "surface_pressure_mean": "pressure",
    "wind_speed_10m_max": "wind_speed",
    "wind_gusts_10m_max": "wind_gusts",
    "precipitation_sum": "precipitation",
}

report = {}


def say(msg):
    print("[build] %s" % msg, flush=True)


def load_pa_v4_daily():
    """Read every per-sensor parquet in pa_v4 (TX sensors only) and reduce to
    daily UTC rows with pm25 = mean(atm_a, atm_b)."""
    sel = pd.read_parquet(SEL)
    tx = sel[sel.box == "TX"].set_index("sensor_index")
    say("TX sensors in selection: %d" % len(tx))

    frames = []
    n_files = 0
    bins_per_day = []
    for tier in ("A", "B"):
        for p in sorted(glob.glob(os.path.join(PA_V4, tier, "*.parquet"))):
            si = int(os.path.basename(p).split(".")[0])
            if si not in tx.index:
                continue
            d = pd.read_parquet(p)
            n_files += 1
            ts_col = "time_stamp" if "time_stamp" in d.columns else d.columns[0]
            d["date"] = (pd.to_datetime(d[ts_col], unit="s", utc=True)
                         .dt.tz_convert(None).dt.normalize())
            a = pd.to_numeric(d.get("pm2.5_atm_a"), errors="coerce")
            b = pd.to_numeric(d.get("pm2.5_atm_b"), errors="coerce")
            d["pm25"] = np.nanmean(np.column_stack([a, b]), axis=1)
            d = d.dropna(subset=["pm25"])
            if tier == "A":
                g = d.groupby("date")["pm25"].agg(["mean", "size"])
                bins_per_day.extend(g["size"].tolist())
                daily = g["mean"].reset_index().rename(columns={"mean": "pm25"})
            else:
                daily = (d.groupby("date")["pm25"].mean().reset_index())
            daily["sensor_id"] = si
            daily["latitude"] = float(tx.loc[si, "latitude"])
            daily["longitude"] = float(tx.loc[si, "longitude"])
            frames.append(daily)
    say("read %d sensor files" % n_files)
    out = pd.concat(frames, ignore_index=True)
    out = out[(out["date"] >= W0) & (out["date"] <= W_LAST_FULL)]
    report["pa_v4_files"] = n_files
    report["pa_v4_raw_rows"] = int(len(out))
    if bins_per_day:
        report["tier_a_bins_per_day_mean"] = float(np.mean(bins_per_day))
    return out


def quality_filter(df):
    """Production QC from 08_finish_pull.py: range, per-sensor MAD z, min days."""
    n0 = len(df)
    df = df.dropna(subset=["pm25"])
    df = df[(df["pm25"] >= PM25_MIN) & (df["pm25"] <= PM25_MAX)]
    n_range = len(df)

    med = df.groupby("sensor_id")["pm25"].transform("median")
    abs_dev = (df["pm25"] - med).abs()
    sensor_mad = abs_dev.groupby(df["sensor_id"]).median()
    pos = sensor_mad[sensor_mad > 0]
    fallback = float(pos.median()) if len(pos) else 1.0
    sensor_mad = sensor_mad.where(sensor_mad > 0, fallback)
    mad = df["sensor_id"].map(sensor_mad)
    z = abs_dev / (1.4826 * mad)
    df = df[z < MAD_Z_MAX]
    n_mad = len(df)

    counts = df.groupby("sensor_id")["pm25"].transform("size")
    df = df[counts >= MIN_DAYS]
    say("QC: %d -> %d (range) -> %d (MAD z<%g) -> %d (>=%d days)"
        % (n0, n_range, n_mad, MAD_Z_MAX, len(df), MIN_DAYS))
    report["qc"] = {"raw": n0, "after_range": n_range, "after_mad": n_mad,
                    "after_min_days": int(len(df))}
    return df.reset_index(drop=True)


def assign_geoid(df_points, tracts):
    """Nearest-tract-centroid GEOID, exact port of 06_pull_purpleair_full.py."""
    pts_lat = df_points["latitude"].to_numpy()
    pts_lon = df_points["longitude"].to_numpy()
    tr_lat = tracts["lat"].to_numpy()
    tr_lon = tracts["lon"].to_numpy()
    tr_geoid = tracts["GEOID"].to_numpy()
    R = 6371.0
    geoids = np.empty(len(pts_lat), dtype=tr_geoid.dtype)
    p_lat_r = np.radians(pts_lat)
    p_lon_r = np.radians(pts_lon)
    t_lat_r = np.radians(tr_lat)
    t_lon_r = np.radians(tr_lon)
    for i in range(0, len(pts_lat), 200):
        a = slice(i, i + 200)
        dlat = p_lat_r[a, None] - t_lat_r[None, :]
        dlon = p_lon_r[a, None] - t_lon_r[None, :]
        h = (np.sin(dlat / 2) ** 2
             + np.cos(p_lat_r[a, None]) * np.cos(t_lat_r[None, :])
             * np.sin(dlon / 2) ** 2)
        d = 2 * R * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
        geoids[a] = tr_geoid[np.argmin(d, axis=1)]
    df_points = df_points.copy()
    df_points["GEOID"] = geoids
    return df_points


def load_weather():
    """Per-sensor daily weather from the pull_weather.py caches."""
    frames = []
    n_om = n_np = 0
    for p in glob.glob(os.path.join(WEATHER_DIR, "om_*.json")):
        sid = int(os.path.basename(p)[3:-5])
        with open(p) as f:
            j = json.load(f)
        if "_error" in j or "daily" not in j:
            continue
        daily = j["daily"]
        cols = {"time": daily["time"]}
        for k in OM_RENAME:
            if k in daily:
                cols[k] = daily[k]
        w = pd.DataFrame(cols).rename(columns=OM_RENAME)
        w["date"] = pd.to_datetime(w["time"]).dt.normalize()
        w = w.drop(columns=["time"])
        units = j.get("daily_units", {})
        wsu = str(units.get("wind_speed_10m_max", "")).lower()
        if "km" in wsu:   # defensive: only convert if the API ignored unit=ms
            w["wind_speed"] = w["wind_speed"] / 3.6
            if "wind_gusts" in w:
                w["wind_gusts"] = w["wind_gusts"] / 3.6
        w["sensor_id"] = sid
        w["weather_source"] = "open_meteo"
        frames.append(w)
        n_om += 1
    for p in glob.glob(os.path.join(WEATHER_DIR, "np_*.json")):
        sid = int(os.path.basename(p)[3:-5])
        with open(p) as f:
            j = json.load(f)
        try:
            par = j["properties"]["parameter"]
        except (KeyError, TypeError):
            continue
        dates = sorted(par.get("T2M", {}).keys())
        if not dates:
            continue

        def col(name, scale=1.0):
            vals = [par.get(name, {}).get(d) for d in dates]
            return [None if (v is None or v == -999) else v * scale
                    for v in vals]

        w = pd.DataFrame({
            "date": pd.to_datetime(dates, format="%Y%m%d"),
            "temperature": col("T2M"),
            "humidity": col("RH2M"),
            "pressure": col("PS", 10.0),
            "wind_speed": col("WS10M_MAX"),
            "precipitation": col("PRECTOTCORR"),
        })
        w["wind_gusts"] = np.nan
        w["sensor_id"] = sid
        w["weather_source"] = "nasa_power"
        frames.append(w)
        n_np += 1
    say("weather caches: %d open_meteo, %d nasa_power" % (n_om, n_np))
    report["weather_sensors"] = {"open_meteo": n_om, "nasa_power": n_np}
    if not frames:
        return pd.DataFrame(columns=["sensor_id", "date"])
    return pd.concat(frames, ignore_index=True)


def add_engineered(df):
    df["month"] = df["date"].dt.month
    df["dow"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear
    df["hour"] = 12
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["temp_x_humidity"] = df["temperature"] * df["humidity"] / 100.0
    df["wind_x_temp"] = df["wind_speed"] * df["temperature"] / 100.0
    df["season"] = df["month"].map(SEASON_MAP)
    df["city"] = ""
    return df


def main():
    new = load_pa_v4_daily()
    new = quality_filter(new)
    say("new part: %d rows, %d sensors" % (len(new), new.sensor_id.nunique()))

    tracts = pd.read_parquet(TRACTS)
    new = assign_geoid(new, tracts)
    new["GEOID"] = new["GEOID"].astype(str).str.zfill(11)
    tr = tracts.copy()
    tr["GEOID"] = tr["GEOID"].astype(str).str.zfill(11)
    new = new.merge(tr, on="GEOID", how="left")

    weather = load_weather()
    new = new.merge(weather, on=["sensor_id", "date"], how="left")
    cov = new["temperature"].notna().mean() * 100
    say("weather coverage on new part: %.1f%%" % cov)
    report["weather_coverage_pct"] = round(float(cov), 2)

    new = add_engineered(new)

    old = pd.read_csv(OLD_CSV, encoding="utf-8-sig", low_memory=False)
    old["date"] = pd.to_datetime(old["date"], errors="coerce")
    old = old.dropna(subset=["date", "pm25"])
    old["GEOID"] = old["GEOID"].astype(str).str.zfill(11)
    say("old dataset: %d rows, %d sensors" % (len(old), old.sensor_id.nunique()))

    # Overlap agreement check before the union (same sensor, same date).
    ov = new.merge(old[["sensor_id", "date", "pm25"]], on=["sensor_id", "date"],
                   suffixes=("", "_old"))
    if len(ov):
        report["overlap"] = {
            "pairs": int(len(ov)),
            "corr": round(float(ov["pm25"].corr(ov["pm25_old"])), 4),
            "mae": round(float((ov["pm25"] - ov["pm25_old"]).abs().mean()), 3),
            "bias_new_minus_old": round(float((ov["pm25"] - ov["pm25_old"]).mean()), 3),
        }
        say("overlap: %d pairs, corr=%.4f, mae=%.3f, bias=%.3f"
            % (len(ov), report["overlap"]["corr"], report["overlap"]["mae"],
               report["overlap"]["bias_new_minus_old"]))

    new_ids = set(new["sensor_id"].astype(int))
    old["sensor_id"] = old["sensor_id"].astype(int)
    old_only = old[~old["sensor_id"].isin(new_ids)]
    old_fill = old[old["sensor_id"].isin(new_ids)]
    report["old_only_sensors"] = int(old_only["sensor_id"].nunique())

    for c in COLUMNS:
        if c not in new.columns:
            new[c] = np.nan
    new = new[COLUMNS]
    old_only = old_only[COLUMNS]
    old_fill = old_fill[COLUMNS]

    final = pd.concat([new, old_fill, old_only], ignore_index=True)
    n_before = len(final)
    final = final.drop_duplicates(subset=["sensor_id", "date"], keep="first")
    say("union dedupe: %d -> %d rows" % (n_before, len(final)))
    final = final.sort_values(["sensor_id", "date"]).reset_index(drop=True)

    report["final_rows"] = int(len(final))
    report["final_sensors"] = int(final["sensor_id"].nunique())
    report["date_min"] = str(final["date"].min().date())
    report["date_max"] = str(final["date"].max().date())
    say("FINAL: %d rows, %d sensors, %s .. %s"
        % (len(final), final["sensor_id"].nunique(),
           report["date_min"], report["date_max"]))

    out_csv = os.path.join(REPO, "p2_processed_v2.xls")
    out_parq = os.path.join(REPO, "pipeline", "purpleair_full_dataset.parquet")
    final.to_csv(out_csv, index=False)
    final.to_parquet(out_parq, index=False)
    with open(os.path.join(BASE, "build_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    say("wrote %s and %s" % (out_csv, out_parq))


if __name__ == "__main__":
    main()
